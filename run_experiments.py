"""Experiment 1 (v3 data): does knowing the topology fix the violation-blindness?

Inputs now include generator voltage setpoints (the OPF chooses them), and
"status" means one indicator per branch: 34 lines and 4 transformers.
Non-converged samples have no target: they are excluded from training and
scoring, and their counts are recorded per test set.

Six models, organized as three regimes x two model classes (linear, MLP):

  regime          trained on                 sees line status?
  blind_intact    intact grid only           no   (the original paper's control)
  blind_n1        intact + seen outages      no
  status_n1       intact + seen outages      yes

blind_n1 vs status_n1 separates "saw outages in training" from "can tell which
outage it is in". Every model is evaluated on the same test sets of each fold:
test_id (intact), test_seen (outages in the training set), test_unseen
(outage lines never seen).

Fair tuning: each MLP regime gets the same small grid. The configuration is
chosen separately for every fold from that fold's training data alone, by
OUTAGE-GROUPED validation for regimes trained on outages (see tune_mlp).
Linear models choose their ridge penalty by outage-grouped CV (see fit).

Resumable: each run's metrics go to results/runs/<name>.json and its test-set
predictions (row indices + predicted [vm | va]) to results/preds/<name>.npz.
A run is skipped if both exist, so the script can be interrupted and rerun.
Phase 2 analyses read the saved predictions instead of retraining.

Run:
    python run_experiments.py               # everything
    python run_experiments.py --budget 600  # stop after ~600 s, rerun to continue
"""

import argparse
import itertools
import json
import os
import time
import warnings

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from mlp_torch import DEVICE, TorchMLP
from sklearn.preprocessing import StandardScaler

import config as C
from metrics import evaluate, evaluate_by_topology

warnings.filterwarnings("ignore")

REGIMES = ["blind_intact", "blind_n1", "status_n1", "hybrid_n1", "hybridv_n1"]
HYBRIDS = ("hybrid_n1", "hybridv_n1")
CONST_TOL = 1e-6   # raw training SD below this = constant input (e.g. condenser P ~ 1e-9 MW noise)
MODELS = ["linear", "mlp"]
FOLDS = range(5)
SEEDS = [0, 1, 2]
TEST_SETS = ["test_id", "test_seen", "test_unseen"]

MLP_GRID = [dict(hidden_layer_sizes=h, learning_rate_init=lr)
            for h, lr in itertools.product([(128, 128), (256, 256)], [1e-3, 3e-4])]
MLP_FIXED = dict(alpha=1e-4)   # L2 penalty; stopping rule and epochs are in mlp_torch.py

RUN_DIR = f"{C.RESULTS_DIR}/runs"
PRED_DIR = f"{C.RESULTS_DIR}/preds"   # test-set predictions per run, for Phase 2 analyses


# ---------------------------------------------------------------- data ----
D = np.load(f"{C.OUT_DIR}/dataset.npz")
S = np.load(f"{C.OUT_DIR}/splits.npz")
NB = D["vm"].shape[1]
CONV = D["converged"]


def idx(name):
    i = S[name]
    return i[CONV[i]]          # non-converged samples have no target; mask them


def train_val_idx(regime, fold):
    if regime == "blind_intact":
        return idx("intact_only/train"), idx("intact_only/val")
    return idx(f"n1_fold_{fold}/train"), idx(f"n1_fold_{fold}/val")


# ------------------------------------------------- physics features ----
# Hybrid models get the pre-outage AC state and the one-iteration fast-decoupled
# (1P1Q) prediction for the outaged grid (see baselines.py), computed once for
# every sample and cached. They learn the residual (truth - 1P1Q) and predict
# 1P1Q + learned residual. They get NO branch-status indicators: everything they
# know about the outage arrives through the physics.
_PHYS = None


def phys():
    global _PHYS
    if _PHYS is None:
        path = f"{C.OUT_DIR}/physics_features.npz"
        if not os.path.exists(path):
            from baselines import predict
            every = np.arange(len(D["topo_id"]))
            print("  computing physics features for all samples (once) ...", flush=True)
            np.savez_compressed(path, base=predict("intact_ac", every),
                                fdlf1=predict("fdlf1", every))
        _PHYS = np.load(path)
    return _PHYS


def anchor(i, regime):
    """What the model's output is added to: 1P1Q for hybrid_n1, nothing otherwise."""
    return phys()["fdlf1"][i] if regime == "hybrid_n1" else 0.0


def assemble(raw, i, regime):
    """Model output -> predicted [vm | va].
    hybrid_n1:  1P1Q + learned residual on [vm | va]
    hybridv_n1: 1P1Q angles unchanged; vm = 1P1Q vm + learned residual on vm only"""
    if regime == "hybridv_n1":
        f = phys()["fdlf1"][i]
        return np.concatenate([f[:, :NB] + raw, f[:, NB:]], axis=1)
    return raw + anchor(i, regime)


class Features:
    """Inputs: per-bus P/Q load, generator P, and generator voltage setpoints,
    standardized. Status models also get one outage indicator per branch
    (1 = out of service), appended unscaled. Hybrid models also get the
    pre-outage AC state and the 1P1Q prediction (both [vm | va]), standardized."""

    def __init__(self, regime, train_idx):
        self.status = regime == "status_n1"
        self.hybrid = regime in HYBRIDS
        raw = self._cont(train_idx)
        # drop inputs that are constant in training: standardizing them only
        # amplifies floating-point noise (condenser P is ~1e-9 MW)
        self.keep = raw.std(0) > CONST_TOL
        self.sx = StandardScaler().fit(raw[:, self.keep])
        Z = self.sx.transform(raw[:, self.keep])
        # learned parts only see inputs within the range seen in training;
        # beyond it they would extrapolate (the physics anchor is never clipped)
        self.lo, self.hi = Z.min(0), Z.max(0)

    def _cont(self, i):
        parts = [D["p_load"][i], D["q_load"][i], D["p_gen"][i], D["v_set"][i]]
        if self.hybrid:
            parts += [phys()["base"][i], phys()["fdlf1"][i]]
        return np.concatenate(parts, axis=1)

    def __call__(self, i):
        X = np.clip(self.sx.transform(self._cont(i)[:, self.keep]), self.lo, self.hi)
        if self.status:
            X = np.concatenate([X, 1 - D["branch_status"][i]], axis=1)
        return X


def targets(i):
    return np.concatenate([D["vm"][i], D["va"][i]], axis=1)


def target_scaler(T):
    """StandardScaler for targets, except columns whose training SD is below
    CONST_TOL are left unscaled. Those columns are constant up to floating-point
    noise (e.g. the slack angle; hybrid residuals at generator buses that never
    hit a reactive limit in training, SD ~1e-16). Dividing by that noise turned
    ordinary residuals on held-out outages into ~1e11 SDs (validation MSE ~1e23
    on case118)."""
    sy = StandardScaler().fit(T)
    sy.scale_ = np.where(T.std(0) < CONST_TOL, 1.0, sy.scale_)
    return sy


def train_targets(i, regime):
    if regime == "hybridv_n1":
        return D["vm"][i] - phys()["fdlf1"][i][:, :NB]
    return targets(i) - anchor(i, regime)


# -------------------------------------------------------------- models ----
RIDGE_ALPHAS = np.logspace(-6, 7, 27)


def _ridge_val_errors(Xtr, Ytr, Xva, Yva, alphas):
    """Validation MSE of ridge regression (with intercept) for every alpha, from ONE
    eigendecomposition of the centered Gram matrix. Mathematically identical to
    fitting sklearn's Ridge once per alpha, but the expensive product X'X is formed
    once instead of once per alpha (case500: 2,733 features x ~20,000 samples)."""
    xm, ym = Xtr.mean(0), Ytr.mean(0)
    Xc = Xtr - xm
    evals, V = np.linalg.eigh(Xc.T @ Xc)
    Bm = V.T @ (Xc.T @ (Ytr - ym))                     # (p, k)
    Zva = (Xva - xm) @ V                               # (n_va, p)
    Yva_c = Yva - ym
    return np.array([(((Zva / (evals + a)) @ Bm - Yva_c) ** 2).mean() for a in alphas])


def grouped_ridge(X, Y, groups, n_groups=4, seed=0):
    """Ridge with the penalty chosen by OUTAGE-GROUPED cross-validation.

    Models meant to generalize to unseen outages must be tuned on held-out
    outages, not held-out samples: sample-level CV picks the penalty that suits
    outages already seen, which for the hybrid led to 0.5 p.u. errors on unseen
    outages. Training outages are split into n_groups; each group is held out in
    turn (intact samples always stay in training)."""
    outs = np.unique(groups[groups >= 0])
    folds = np.array_split(np.random.default_rng(seed).permutation(outs), n_groups)
    err = np.zeros(len(RIDGE_ALPHAS))
    for g in folds:
        va = np.isin(groups, g)
        err += _ridge_val_errors(X[~va], Y[~va], X[va], Y[va], RIDGE_ALPHAS)
    best = RIDGE_ALPHAS[int(np.argmin(err))]
    m = Ridge(alpha=best).fit(X, Y)
    m.alpha_ = best
    return m


def linear_path(X, Y, groups, regime, fold):
    """Fixed linear path for an MLP: the regime's own ridge model for this fold.
    Reuses the penalty its linear run chose (outage-grouped CV; GCV for
    blind_intact), so the expensive search is not repeated per MLP fit; if that
    run is missing, the penalty is chosen here the same way."""
    path = f"{C.RESULTS_DIR}/runs/{regime}__linear__fold{fold}__seed0.json"
    if os.path.exists(path):
        lin = Ridge(alpha=json.load(open(path))["params"]["alpha"]).fit(X, Y)
    elif groups is not None:
        lin = grouped_ridge(X, Y, groups)
    else:
        lin = RidgeCV(alphas=RIDGE_ALPHAS).fit(X, Y)
    return lin.coef_, lin.intercept_


def fit(model, params, X, Y, seed, groups=None, linear=None):
    if model == "linear":
        # Plain least squares blows up when inputs are nearly collinear (the
        # hybrid's pre-outage state and 1P1Q prediction: 3.8 p.u. errors on
        # unseen outages). Regimes trained on outages choose the ridge penalty
        # by outage-grouped CV; the intact-only regime has no outages to group
        # by and uses leave-one-out CV (GCV).
        if groups is not None:
            return grouped_ridge(X, Y, groups)
        return RidgeCV(alphas=RIDGE_ALPHAS).fit(X, Y)
    return TorchMLP(**params, **MLP_FIXED, random_state=seed, linear=linear).fit(X, Y)


VAL_OUTAGE_FRAC = 1 / 3


def tune_mlp(regime, fold):
    """Choose the MLP configuration for one fold, using only that fold's
    training data.

    Regimes trained on outages: OUTAGE-GROUPED validation. A third of the
    fold's training outages is held out; each configuration is trained on the
    rest (intact samples always stay in training) and scored by MSE (on
    standardized targets) on the held-out outages. This mimics the shift the
    model is evaluated on. blind_intact has no outages to hold out and uses its
    sample-level validation set; its choice does not depend on the fold."""
    key = "all" if regime == "blind_intact" else f"fold{fold}"
    path = f"{C.RESULTS_DIR}/tuning_{regime}_{key}.json"
    if os.path.exists(path):
        return json.load(open(path))["best"]
    if regime == "blind_intact":
        fit_idx, val_idx = train_val_idx(regime, 0)
        how = "sample-level validation set"
    else:
        tr, _ = train_val_idx(regime, fold)
        topo = D["topo_id"][tr]
        outs = np.unique(topo[topo >= 0])
        rng = np.random.default_rng(C.SEED + 100 + fold)
        held = rng.permutation(outs)[:max(1, int(round(len(outs) * VAL_OUTAGE_FRAC)))]
        is_val = np.isin(topo, held)
        fit_idx, val_idx = tr[~is_val], tr[is_val]
        how = f"held-out training outages {sorted(int(h) for h in held)}"
    feats = Features(regime, fit_idx)
    sy = target_scaler(train_targets(fit_idx, regime))
    Xtr, Ytr = feats(fit_idx), sy.transform(train_targets(fit_idx, regime))
    Xva, Yva = feats(val_idx), sy.transform(train_targets(val_idx, regime))
    groups = None if regime == "blind_intact" else D["topo_id"][fit_idx]
    lin = linear_path(Xtr, Ytr, groups, regime, 0 if regime == "blind_intact" else fold)
    trials = []
    for p in MLP_GRID:
        m = fit("mlp", p, Xtr, Ytr, seed=0, linear=lin)
        mse = float(((m.predict(Xva) - Yva) ** 2).mean())
        trials.append(dict(params={k: list(v) if isinstance(v, tuple) else v
                                   for k, v in p.items()}, val_mse=mse))
        print(f"    tune {regime} {key}: {p} -> val MSE {mse:.5f}", flush=True)
    best = min(trials, key=lambda t: t["val_mse"])["params"]
    json.dump(dict(best=best, validation=how, trials=trials), open(path, "w"), indent=2)
    return best


# ---------------------------------------------------------------- runs ----
def run_one(regime, model, fold, seed, mlp_params):
    tr, _ = train_val_idx(regime, fold)
    feats = Features(regime, tr)
    sy = target_scaler(train_targets(tr, regime))
    params = {k: tuple(v) if isinstance(v, list) else v for k, v in mlp_params.items()}
    t0 = time.time()
    groups = None if regime == "blind_intact" else D["topo_id"][tr]
    Xtr, Ytr = feats(tr), sy.transform(train_targets(tr, regime))
    lin = linear_path(Xtr, Ytr, groups, regime, fold) if model == "mlp" else None
    m = fit(model, params, Xtr, Ytr, seed, groups, linear=lin)
    out = dict(regime=regime, model=model, fold=fold, seed=seed,
               params=mlp_params if model == "mlp" else {"alpha": float(m.alpha_)},
               fit_seconds=round(time.time() - t0, 1), results={}, by_topology={})
    preds = {}
    for ts in TEST_SETS:
        out.setdefault("non_converged", {})[ts] = int((~CONV[S[f"n1_fold_{fold}/{ts}"]]).sum())
        i = idx(f"n1_fold_{fold}/{ts}")
        pred = assemble(sy.inverse_transform(m.predict(feats(i))), i, regime)
        Y = targets(i)
        out["results"][ts] = evaluate(pred, Y, NB)
        preds[f"{ts}/idx"] = i
        preds[f"{ts}/pred"] = pred.astype(np.float32)
        # Non-converged cases have no true solution, but the model still makes a
        # prediction. Saved so case-level analysis can ask whether the model
        # flags them (a non-converged outage is treated as insecure).
        nc = S[f"n1_fold_{fold}/{ts}"]
        nc = nc[~CONV[nc]]
        preds[f"{ts}/nc_idx"] = nc
        preds[f"{ts}/nc_pred"] = (assemble(sy.inverse_transform(m.predict(feats(nc))), nc, regime).astype(np.float32)
                                  if len(nc) else np.zeros((0, 2 * NB), np.float32))
        if ts != "test_id":
            out["by_topology"][ts] = evaluate_by_topology(pred, Y, D["topo_id"][i], NB)
    return out, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=None, help="seconds before stopping")
    args = ap.parse_args()
    os.makedirs(RUN_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)
    t_start = time.time()
    print(f"MLPs train on: {DEVICE}", flush=True)

    def out_of_time():
        return args.budget is not None and time.time() - t_start > args.budget

    jobs = [(r, m, f, s) for r in REGIMES for m in MODELS for f in FOLDS
            for s in (SEEDS if m == "mlp" else [0])]      # linear is deterministic
    done = 0
    for r, m, f, s in jobs:
        path = f"{RUN_DIR}/{r}__{m}__fold{f}__seed{s}.json"
        ppath = f"{PRED_DIR}/{r}__{m}__fold{f}__seed{s}.npz"
        if os.path.exists(path) and os.path.exists(ppath):
            done += 1; continue
        if out_of_time():
            print(f"Budget reached: {done}/{len(jobs)} runs done; rerun to continue.")
            return
        params = tune_mlp(r, f) if m == "mlp" else {}
        if out_of_time():
            print(f"Budget reached after tuning: {done}/{len(jobs)} runs done; rerun to continue.")
            return
        res, preds = run_one(r, m, f, s, params)
        json.dump(res, open(path, "w"), indent=1)
        np.savez_compressed(ppath, **preds)
        done += 1
        u = res["results"]["test_unseen"]
        print(f"  [{done:2d}/{len(jobs)}] {r:12s} {m:6s} fold {f} seed {s}  "
              f"unseen recall {u['recall']:.2f}  ({res['fit_seconds']}s)", flush=True)
    print(f"All {len(jobs)} runs done. Next: python report.py")


if __name__ == "__main__":
    main()