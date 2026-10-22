"""Phase 3: graph neural network (plain PyTorch), the purely learned
structure-aware model.

VERSION v3 (final). The GNN learns a correction on top of the intact-grid ridge
model (see linear_base); training starts at that model and keeps it unless the
correction improves validation error. Residual targets use one scale per
quantity (see Scalers). History: v1 had no linear path (max 200 epochs); v2
added the linear path but scaled each bus separately, which let training
degrade accuracy on case118. Results from different versions must not be mixed.

Graph: nodes = buses, edges = all branches (both directions). An outaged
branch is masked out of message passing, so an unseen outage is not an unseen
INPUT CODE (as with branch-status indicators) but a known graph with one
edge removed. The model never sees 1P1Q or the pre-outage state.

  node inputs  p_load, q_load, p_gen, v_set (per bus, standardized per channel),
               bus type one-hot (slack / PV / PQ), shunt susceptance
  edge inputs  r, x, b, tap ratio (standardized), direction flag
  model        encoder -> K residual edge-gated message-passing layers
               (message along j->i = MLP_gate(e_ji) * W h_j, summed over
               in-service edges; update = MLP[h, aggregate], LayerNorm)
               -> per-bus [vm, va]
  targets      vm, va per bus, standardized per bus

Protocol identical to run_experiments.py: same folds and seeds (3), non-
converged samples excluded from training and predicted for case analysis,
predictions written to results/preds/gnn_n1__gnn__fold<k>__seed<s>.npz.
Depth K is chosen PER FOLD by outage-grouped validation (a third of the
training outages held out), like the MLPs. Training uses Adam with early
stopping on the fold's validation set.

Run:  python gnn.py                 all folds, seeds 0-2 (resumable)
      python gnn.py --seeds 0        one seed only
      python gnn.py --budget 600     stop after ~600 s, checkpointing mid-training;
                                     rerun to continue exactly where it stopped
Fit times differ between runs, so "fit_seconds" is only comparable for runs
that were not interrupted.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from pandapower.pypower.idx_brch import BR_B, BR_R, BR_X, TAP
from pandapower.pypower.idx_bus import BS, BUS_TYPE

import config as C
from metrics import evaluate, evaluate_by_topology
from mlp_torch import DEVICE        # CUDA if available; GRID_DEVICE=cpu overrides
from physics import GridPhysics

torch.set_num_threads(max(1, os.cpu_count() or 1))

D = np.load(f"{C.OUT_DIR}/dataset.npz")
S = np.load(f"{C.OUT_DIR}/splits.npz")
CONV = D["converged"]
NB = D["vm"].shape[1]
GP = GridPhysics()
N_STATUS = D["branch_status"].shape[1]          # 38 enumerated branches
N_BR = GP.branch.shape[0]                       # 41 incl. impedance elements

FOLDS, SEEDS = range(5), [0, 1, 2]
DEPTHS = [4, 8]                                 # tuned per fold
HIDDEN, LR, BATCH, MAX_EPOCHS, PATIENCE = 64, 1e-3, 128, 400, 20
# MAX_EPOCHS was 200; several tuning fits hit it while still improving.
TEST_SETS = ["test_id", "test_seen", "test_unseen"]
VAL_OUTAGE_FRAC = 1 / 3

# ------------------------------------------------------------ graph -----
f, t = GP.f, GP.t
SRC = torch.tensor(np.r_[f, t], dtype=torch.long, device=DEVICE)
DST = torch.tensor(np.r_[t, f], dtype=torch.long, device=DEVICE)
br = GP.branch.real
tap = np.where(br[:, TAP] == 0, 1.0, br[:, TAP])
e = np.stack([br[:, BR_R], br[:, BR_X], br[:, BR_B], tap], 1)
e = (e - e.mean(0)) / (e.std(0) + 1e-12)
EDGE = torch.tensor(np.r_[np.c_[e, np.ones(N_BR)], np.c_[e, -np.ones(N_BR)]], dtype=torch.float32,
                    device=DEVICE)
types = GP.bus[:, BUS_TYPE].real.astype(int)
bs = GP.bus[:, BS].real
STATIC = torch.tensor(np.c_[types == 3, types == 2, types == 1, bs / (np.abs(bs).max() + 1e-12)],
                      dtype=torch.float32, device=DEVICE)
EVAL_CHUNK = 1024   # samples per forward pass when evaluating (keeps GPU memory low)


def idx(name):
    i = S[name]
    return i[CONV[i]]


def edge_mask(i):
    """(n, 2*N_BR) in-service mask; impedance elements are always in service."""
    st = np.c_[D["branch_status"][i], np.ones((len(i), N_BR - N_STATUS))]
    return torch.tensor(np.r_["1", st, st], dtype=torch.float32, device=DEVICE)


# ------------------------------------------------------ linear path -----
# The GNN predicts a CORRECTION on top of a fixed linear model: the ridge
# regression trained on the intact grid only (identical to the blind_intact
# linear model: same features, same leave-one-out (GCV) penalty search, so it
# is recomputed here rather than read from results). It predicts the state as
# if nothing happened; message passing learns how the outage changes it.
# Global quantities (each bus's angle relative to the slack) come from the
# linear part, which sees every injection; an 8-layer GNN only sees 8 hops,
# which on case118 left it 13x worse than linear on the intact grid.
_BASE = None


def linear_base():
    """[vm | va] predicted by the intact-grid ridge model, for every sample."""
    global _BASE
    if _BASE is None:
        path = f"{C.OUT_DIR}/gnn_linear_base_v1.npz"
        if not os.path.exists(path):
            import run_experiments as R
            tr = R.idx("intact_only/train")
            feats = R.Features("blind_intact", tr)
            sy = R.target_scaler(R.train_targets(tr, "blind_intact"))
            lin = R.RidgeCV(alphas=R.RIDGE_ALPHAS).fit(feats(tr), sy.transform(R.train_targets(tr, "blind_intact")))
            every = np.arange(len(D["topo_id"]))
            base = sy.inverse_transform(lin.predict(feats(every)))
            np.savez_compressed(path, base=base.astype(np.float64), alpha=lin.alpha_)
            print(f"  linear path: intact-grid ridge, alpha {lin.alpha_:.0e} (cached in {path})", flush=True)
        _BASE = np.load(path)["base"]
    return _BASE


def base_stack(i):
    b = linear_base()[i]
    return np.stack([b[:, :NB], b[:, NB:]], -1)            # (n, NB, 2)


class Scalers:
    """Inputs standardized per channel; targets are the RESIDUAL (truth minus
    the linear path), standardized per bus."""

    def __init__(self, tr):
        X = self.raw(tr)
        self.xm, self.xs = X.mean((0, 1)), X.std((0, 1)) + 1e-12
        Y = np.stack([D["vm"][tr], D["va"][tr]], -1) - base_stack(tr)
        # v3: ONE scale per quantity (all voltage-magnitude residuals share one,
        # all angle residuals another), the RMS of the training residuals.
        # v2 scaled each bus by its own spread; buses whose residual barely varies
        # in training (generator buses, SD ~1e-5) then dominated the loss AND the
        # early-stopping criterion, so training could "improve" it while absolute
        # errors grew elsewhere (case118: 20x worse than the starting point on the
        # intact grid). With a common scale, lower validation error means more
        # accurate predictions, so early stopping keeps the linear start unless
        # the GNN genuinely improves on it.
        # Residuals are scaled but NOT centered: an output of zero means exactly
        # the linear path.
        rms = np.sqrt((Y ** 2).mean(axis=(0, 1)))                 # (2,): vm, va
        rms = np.where(rms < 1e-12, 1.0, rms)
        self.ym = np.zeros(Y.shape[1:])
        self.ys = np.broadcast_to(rms, Y.shape[1:]).copy()        # (NB, 2)

    @staticmethod
    def raw(i):
        return np.stack([D["p_load"][i], D["q_load"][i], D["p_gen"][i], D["v_set"][i]], -1)

    def x(self, i):
        X = torch.tensor((self.raw(i) - self.xm) / self.xs, dtype=torch.float32, device=DEVICE)
        return torch.cat([X, STATIC.expand(len(i), -1, -1)], -1)

    def y(self, i):
        Y = np.stack([D["vm"][i], D["va"][i]], -1) - base_stack(i)
        return torch.tensor((Y - self.ym) / self.ys, dtype=torch.float32, device=DEVICE)

    def back(self, out, i):
        Y = out.detach().cpu().numpy() * self.ys + self.ym + base_stack(i)
        return np.concatenate([Y[..., 0], Y[..., 1]], 1)


def mlp(i, o):
    return nn.Sequential(nn.Linear(i, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, o))


class GNN(nn.Module):
    """Edge-gated message passing. Each layer computes, once per forward pass,
    a gate vector per branch direction from that branch's electrical
    parameters (r, x, b, tap, direction): g_e = MLP(e). The message along
    edge e = (j -> i) is g_e * (W h_j); messages over in-service edges are
    summed at i, and the bus state is updated by a residual MLP + LayerNorm.
    Gates depend only on static branch parameters, so the per-sample cost is
    a gather and a sum, not a neural network per edge per sample."""

    def __init__(self, depth, f_node=8, f_edge=5):
        super().__init__()
        self.enc_n = mlp(f_node, HIDDEN)
        self.gate = nn.ModuleList([nn.Sequential(mlp(f_edge, HIDDEN), nn.Tanh())
                                   for _ in range(depth)])
        self.lin = nn.ModuleList([nn.Linear(HIDDEN, HIDDEN, bias=False) for _ in range(depth)])
        self.upd = nn.ModuleList([mlp(2 * HIDDEN, HIDDEN) for _ in range(depth)])
        self.norm = nn.ModuleList([nn.LayerNorm(HIDDEN) for _ in range(depth)])
        self.dec = mlp(HIDDEN, 2)
        # last layer starts at zero: the initial prediction is exactly the linear path
        nn.init.zeros_(self.dec[-1].weight); nn.init.zeros_(self.dec[-1].bias)

    def forward(self, x, mask):
        h = self.enc_n(x)                                    # (B, N, H)
        for gate, lin, upd, norm in zip(self.gate, self.lin, self.upd, self.norm):
            g = gate(EDGE)                                   # (2E, H), static per edge
            m = lin(h)[:, SRC] * g * mask.unsqueeze(-1)      # outaged branches send nothing
            agg = torch.zeros_like(h).index_add_(1, DST, m)
            h = norm(h + upd(torch.cat([h, agg], -1)))
        return self.dec(h)                                   # (B, N, 2)


class OutOfTime(Exception):
    pass


DEADLINE = None   # wall-clock time after which training checkpoints and stops


def forward_chunked(model, X, M):
    with torch.no_grad():
        return torch.cat([model(X[k:k + EVAL_CHUNK], M[k:k + EVAL_CHUNK])
                          for k in range(0, len(X), EVAL_CHUNK)])


def train(depth, tr, va, seed, sc, ckpt=None):
    """Adam + early stopping on va. If ckpt is given, the full training state
    (model, optimizer, RNGs, early-stopping state) is saved after every epoch
    and restored on the next call, so a fit can span several sessions."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = GNN(depth).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    Xtr, Mtr, Ytr = sc.x(tr), edge_mask(tr), sc.y(tr)
    Xva, Mva, Yva = sc.x(va), edge_mask(va), sc.y(va)
    best, best_state, bad, start = np.inf, None, 0, 0
    if ckpt and os.path.exists(ckpt):
        st = torch.load(ckpt, weights_only=False, map_location=DEVICE)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        rng.bit_generator.state = st["np_rng"]; torch.set_rng_state(st["torch_rng"].cpu())
        if DEVICE.type == "cuda" and st.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(st["cuda_rng"].cpu())
        best, best_state, bad, start = st["best"], st["best_state"], st["bad"], st["epoch"]
    if start == 0:                                   # the starting point is a candidate
        model.eval()
        best = float(((forward_chunked(model, Xva, Mva) - Yva) ** 2).mean())
        best_state = {k: p.detach().clone() for k, p in model.state_dict().items()}
    epoch = start - 1
    for epoch in range(start, MAX_EPOCHS):
        if DEADLINE and time.time() > DEADLINE:
            raise OutOfTime
        model.train()
        for b in np.array_split(rng.permutation(len(tr)), max(1, len(tr) // BATCH)):
            b = torch.as_tensor(b, device=DEVICE)
            opt.zero_grad()
            loss = ((model(Xtr[b], Mtr[b]) - Ytr[b]) ** 2).mean()
            loss.backward()
            opt.step()
        model.eval()
        v = float(((forward_chunked(model, Xva, Mva) - Yva) ** 2).mean())
        if v < best - 1e-6:
            best, bad = v, 0
            best_state = {k: p.clone() for k, p in model.state_dict().items()}
        else:
            bad += 1
        if ckpt:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                            np_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state() if DEVICE.type == "cuda" else None,
                            best=best, best_state=best_state, bad=bad, epoch=epoch + 1), ckpt)
        if bad >= PATIENCE:
            break
    model.load_state_dict(best_state)
    if ckpt and os.path.exists(ckpt):
        os.remove(ckpt)
    return model, best, epoch + 1


def predict(model, sc, i):
    if len(i) == 0:
        return np.zeros((0, 2 * NB), np.float32)
    model.eval()
    return sc.back(forward_chunked(model, sc.x(i), edge_mask(i)), i)


def tune_depth(fold):
    """Outage-grouped validation, using only the fold's training data."""
    path = f"{C.RESULTS_DIR}/tuning_gnn_n1_fold{fold}.json"
    if os.path.exists(path):
        return json.load(open(path))["best_depth"]
    tr = idx(f"n1_fold_{fold}/train")
    topo = D["topo_id"][tr]
    outs = np.unique(topo[topo >= 0])
    held = np.random.default_rng(C.SEED + 100 + fold).permutation(outs)[
        :max(1, int(round(len(outs) * VAL_OUTAGE_FRAC)))]
    is_val = np.isin(topo, held)
    fit_i, val_i = tr[~is_val], tr[is_val]
    sc = Scalers(fit_i)
    trials = []
    os.makedirs(f"{C.RESULTS_DIR}/ckpt", exist_ok=True)
    for depth in DEPTHS:
        done_path = f"{C.RESULTS_DIR}/ckpt/tune_fold{fold}_depth{depth}.json"
        if os.path.exists(done_path):
            trials.append(json.load(open(done_path))); continue
        _, v, ep = train(depth, fit_i, val_i, seed=0, sc=sc,
                         ckpt=f"{C.RESULTS_DIR}/ckpt/tune_fold{fold}_depth{depth}.pt")
        trials.append(dict(depth=depth, val_mse=v, epochs=ep))
        json.dump(trials[-1], open(done_path, "w"))
        print(f"    tune gnn fold {fold}: depth {depth} -> held-out-outage MSE {v:.4f} ({ep} epochs)",
              flush=True)
    best = min(trials, key=lambda r: r["val_mse"])["depth"]
    json.dump(dict(best_depth=best, held_out_outages=sorted(int(h) for h in held), trials=trials),
              open(path, "w"), indent=2)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = ap.parse_args()
    global DEADLINE
    if args.budget:
        DEADLINE = time.time() + args.budget
    print(f"GNN v3 trains on: {DEVICE}", flush=True)
    os.makedirs(f"{C.RESULTS_DIR}/runs", exist_ok=True)
    os.makedirs(f"{C.RESULTS_DIR}/preds", exist_ok=True)
    t_start = time.time()
    jobs = [(f, s) for f in FOLDS for s in args.seeds]
    for done, (fold, seed) in enumerate(jobs):
        name = f"gnn_n1__gnn__fold{fold}__seed{seed}"
        if os.path.exists(f"{C.RESULTS_DIR}/runs/{name}.json") and os.path.exists(f"{C.RESULTS_DIR}/preds/{name}.npz"):
            continue
        try:
            depth = tune_depth(fold)
            tr, va = idx(f"n1_fold_{fold}/train"), idx(f"n1_fold_{fold}/val")
            sc = Scalers(tr)
            t0 = time.time()
            model, vbest, epochs = train(depth, tr, va, seed, sc, ckpt=f"{C.RESULTS_DIR}/ckpt/{name}.pt")
        except OutOfTime:
            print(f"Budget reached ({done}/{len(jobs)} runs done; training checkpointed). "
                  "Rerun to continue."); return
        res = dict(regime="gnn_n1", model="gnn", fold=fold, seed=seed,
                   params=dict(depth=depth, hidden=HIDDEN, lr=LR, epochs=epochs),
                   fit_seconds=round(time.time() - t0, 1), results={}, by_topology={},
                   non_converged={})
        preds = {}
        for ts in TEST_SETS:
            allidx = S[f"n1_fold_{fold}/{ts}"]
            i, nc = allidx[CONV[allidx]], allidx[~CONV[allidx]]
            pred = predict(model, sc, i)
            Y = np.concatenate([D["vm"][i], D["va"][i]], 1)
            res["results"][ts] = evaluate(pred, Y, NB)
            res["non_converged"][ts] = int(len(nc))
            if ts != "test_id":
                res["by_topology"][ts] = evaluate_by_topology(pred, Y, D["topo_id"][i], NB)
            preds[f"{ts}/idx"], preds[f"{ts}/pred"] = i, pred.astype(np.float32)
            preds[f"{ts}/nc_idx"], preds[f"{ts}/nc_pred"] = nc, predict(model, sc, nc).astype(np.float32)
        json.dump(res, open(f"{C.RESULTS_DIR}/runs/{name}.json", "w"), indent=1)
        np.savez_compressed(f"{C.RESULTS_DIR}/preds/{name}.npz", **preds)
        u = res["results"]["test_unseen"]
        print(f"  [{done + 1:2d}/{len(jobs)}] fold {fold} seed {seed} depth {depth}: unseen |V| err "
              f"{u['vm_mae_pu']:.4f}, recall {u['recall']:.2f}  ({res['fit_seconds']}s, {epochs} epochs)",
              flush=True)
    print(f"All {len(jobs)} GNN runs done.")


if __name__ == "__main__":
    main()