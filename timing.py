"""Phase 3: speed of every method, per contingency case.

Measured on the same samples (fold 0, unseen outages), on one CPU thread:

  exact AC
    pandapower   pp.runpp with reactive limits enforced (what generate.py does):
                 a realistic tool, including its software overhead
    newton       a lean Newton-Raphson on prebuilt matrices, warm-started from
                 the pre-outage state (pypower's dSbus_dV Jacobian, sparse LU).
                 It does NOT enforce reactive limits (that needs an outer loop),
                 so it is a LOWER BOUND on the cost of an exact solve
  physics        dc, dc_inc, fdlf1 (1P1Q), as in baselines.py
  learned        ridge (60 outputs), PyTorch MLP (128x128 and 256x256), GNN (depth 8,
                 64 hidden), and the hybrid = 1P1Q + ridge correction (+ input
                 preparation). Inference cost does not depend on trained
                 weights, so models of the chosen sizes are used untrained
                 (GNN) or minimally fitted (sklearn) for timing only.

Two regimes: batch 1 (latency: one contingency per call) and batch 1000
(throughput, vectorized). One-off costs are reported separately: matrix
factorization per topology, and the pre-outage AC solve per base case, which
physics baselines and hybrids need and which is shared by that base case's
37 contingencies.

Absolute times are machine-specific; ratios between methods are the result.
Run:  python timing.py   -> results/timing.json
"""

import os
os.environ.setdefault("GRID_DEVICE", "cpu")   # time every method on the CPU, like for like
import copy
import json
import time
import warnings

import numpy as np
import pandapower as pp
import torch
from pandapower.pypower.dSbus_dV import dSbus_dV
from scipy.sparse import hstack, vstack
from scipy.sparse.linalg import spsolve
from sklearn.linear_model import Ridge
import mlp_torch
from mlp_torch import TorchMLP

import baselines as B
import config as C
import gnn as G
from grid_case import load_net

warnings.filterwarnings("ignore")
torch.set_num_threads(1)

# np.load on a compressed .npz returns a LAZY file: every D["key"] re-reads and
# decompresses the whole array. In per-case loops that overhead swamps the
# methods being timed (it made DC look 500x slower), so load everything into
# memory once and point every module at the in-memory arrays.
D = {k: np.asarray(v) for k, v in np.load(f"{C.OUT_DIR}/dataset.npz").items()}
B.D = G.D = D
S, NB = B.S, B.NB
REPS = 3


def best_of(fn, reps=REPS):
    """Minimum wall time over reps (standard practice for microbenchmarks)."""
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best


# ------------------------------------------------------- lean Newton -----
def newton(Ybus, Sbus, V0, tol=1e-8, max_it=20):
    V = V0.copy()
    pvpq, pq = B.PVPQ, B.PQ
    for it in range(max_it + 1):
        mis = V * np.conj(Ybus @ V) - Sbus
        F = np.r_[mis[pvpq].real, mis[pq].imag]
        if np.abs(F).max() < tol:
            return V, it
        dS_dVm, dS_dVa = dSbus_dV(Ybus, V)
        J = vstack([hstack([dS_dVa[pvpq][:, pvpq].real, dS_dVm[pvpq][:, pq].real]),
                    hstack([dS_dVa[pq][:, pvpq].imag, dS_dVm[pq][:, pq].imag])]).tocsc()
        dx = spsolve(J, -F)
        Va, Vm = np.angle(V), np.abs(V)
        Va[pvpq] += dx[:len(pvpq)]; Vm[pq] += dx[len(pvpq):]
        V = Vm * np.exp(1j * Va)
    return V, max_it


def main():
    te = S["n1_fold_0/test_unseen"]; te = te[D["converged"][te]]
    one = te[:300]                                     # latency loops
    res = {}

    base_net = load_net()                    # same repaired network as the data
    assert base_net.load["bus"].is_unique, "one load per bus assumed"

    def build_net(i):
        """pandapower model of sample i: its operating point and its outage."""
        n = copy.deepcopy(base_net)
        n.load["p_mw"] = D["p_load"][i][n.load["bus"].values]
        n.load["q_mvar"] = D["q_load"][i][n.load["bus"].values]
        if "gen_p_el" in D:
            # grids with static generators (case500): per-unit dispatch was stored,
            # and q_load is net of static-generator Q, so add it back for the loads
            n.gen["p_mw"] = D["gen_p_el"][i]
            n.sgen["p_mw"] = D["sgen_p_el"][i]
            n.sgen["q_mvar"] = D["sgen_q_el"][i]
            qs = np.zeros(len(n.bus))
            np.add.at(qs, n.sgen["bus"].values, D["sgen_q_el"][i])
            n.load["q_mvar"] = (D["q_load"][i] + qs)[n.load["bus"].values]
        else:
            n.gen["p_mw"] = D["p_gen"][i][n.gen["bus"].values]
        n.gen["vm_pu"] = D["v_set"][i][n.gen["bus"].values]
        n.ext_grid["vm_pu"] = D["v_set"][i][n.ext_grid["bus"].values]
        el, k = B.GP_BRANCHES[int(D["topo_id"][i])]
        n[el].at[k, "in_service"] = False
        return n

    # ---- validate the lean Newton against pandapower, both WITHOUT reactive
    # limits, so the two solve identical equations. (Comparing against the
    # stored solutions only works where no generator is at a reactive limit;
    # on case118 ~10 generators are at a limit even in normal operation.)
    errs, its = [], []
    for i in one[:20]:
        n = build_net(i)
        try:
            pp.runpp(n, numba=False, enforce_q_lims=False, max_iteration=30)
        except Exception:
            continue
        br = B.INTACT_ROW[int(D["base_id"][i])]
        V0 = D["vm"][br] * np.exp(1j * np.deg2rad(D["va"][br]))
        Sbus = (D["p_gen"][i] - D["p_load"][i] - 1j * D["q_load"][i]) / B.GP.base_mva
        V, k = newton(GP_Y(int(D["topo_id"][i])), Sbus, V0)
        errs.append(np.abs(np.abs(V) - n.res_bus["vm_pu"].values).max()); its.append(k)
    assert errs, "no case converged in pandapower; cannot validate the Newton solver"
    print(f"Newton check vs pandapower (no reactive limits), {len(errs)} cases: "
          f"max |V| difference {max(errs):.1e} p.u., {np.mean(its):.1f} iterations on average")
    assert max(errs) < 1e-6, "lean Newton disagrees with pandapower: timing would be meaningless"

    # ---- exact AC: pandapower ---------------------------------------------
    nets = [build_net(i) for i in one[:30]]
    t_pp = []
    for n in nets:
        t0 = time.perf_counter()
        pp.runpp(n, numba=False, enforce_q_lims=True, max_iteration=30)
        t_pp.append(time.perf_counter() - t0)
    res["pandapower_runpp"] = dict(batch1=float(np.median(t_pp)))

    # ---- exact AC: lean Newton, warm start ---------------------------------
    Ys = {int(t): GP_Y(int(t)) for t in np.unique(D["topo_id"][one])}
    def run_newton():
        for i in one:
            br = B.INTACT_ROW[int(D["base_id"][i])]
            V0 = D["vm"][br] * np.exp(1j * np.deg2rad(D["va"][br]))
            Sbus = (D["p_gen"][i] - D["p_load"][i] - 1j * D["q_load"][i]) / B.GP.base_mva
            newton(Ys[int(D["topo_id"][i])], Sbus, V0)
    res["newton_core"] = dict(batch1=best_of(run_newton, 1) / len(one))

    # ---- physics baselines (matrices prebuilt) ------------------------------
    for t in np.unique(D["topo_id"][te]):
        B.mats(int(t))
    B.mats(-1)
    for m in ["dc", "dc_inc", "fdlf1"]:
        b1 = best_of(lambda: [B.predict(m, one[k:k + 1]) for k in range(len(one))], 1) / len(one)
        bN = best_of(lambda: B.predict(m, te[:1000])) / min(1000, len(te))
        res[m] = dict(batch1=b1, batch1000=bN)

    # ---- learned models ------------------------------------------------------
    rng = np.random.default_rng(0)
    n_in_plain, n_in_hyb = 4 * NB, 8 * NB      # hybrid: 4 inputs + pre-outage [vm|va] + 1P1Q [vm|va] per bus
    X_small = rng.standard_normal((200, n_in_hyb)); Y_small = rng.standard_normal((200, 2 * NB))
    ridge = Ridge(alpha=1.0).fit(X_small[:, :n_in_hyb], Y_small[:, :NB])
    mlp_torch.MAX_EPOCHS = 1                     # weights don't affect inference cost
    mlps = {h: TorchMLP(hidden_layer_sizes=h).fit(X_small[:, :n_in_plain], Y_small)
            for h in [(128, 128), (256, 256)]}
    Xp1, XpN = rng.standard_normal((1, n_in_plain)), rng.standard_normal((1000, n_in_plain))
    Xh1, XhN = rng.standard_normal((1, n_in_hyb)), rng.standard_normal((1000, n_in_hyb))
    res["ridge_correction"] = dict(batch1=best_of(lambda: [ridge.predict(Xh1) for _ in range(300)]) / 300,
                                   batch1000=best_of(lambda: ridge.predict(XhN)) / 1000)
    for h, m in mlps.items():
        res[f"mlp_{h[0]}x{h[1]}"] = dict(batch1=best_of(lambda: [m.predict(Xp1) for _ in range(300)]) / 300,
                                          batch1000=best_of(lambda: m.predict(XpN)) / 1000)
    G.HIDDEN = 64
    gnn = G.GNN(8).eval()
    sc = G.Scalers(te[:1000])
    x1, m1 = sc.x(te[:1]), G.edge_mask(te[:1])
    xN, mN = sc.x(te[:1000]), G.edge_mask(te[:1000])
    with torch.no_grad():
        res["gnn_depth8"] = dict(batch1=best_of(lambda: [gnn(x1, m1) for _ in range(100)]) / 100,
                                 batch1000=best_of(lambda: gnn(xN, mN)) / 1000)
    # hybrid = 1P1Q + input preparation + ridge correction (measured end to end)
    def hybrid_one(i):
        f = B.predict("fdlf1", i)
        rows = np.array([B.INTACT_ROW[int(b)] for b in D["base_id"][i]])
        x = np.concatenate([D["p_load"][i], D["q_load"][i], D["p_gen"][i], D["v_set"][i],
                            D["vm"][rows], D["va"][rows], f], 1)
        return ridge.predict(x[:, :n_in_hyb])
    res["hybrid_1p1q_plus_ridge"] = dict(
        batch1=best_of(lambda: [hybrid_one(one[k:k + 1]) for k in range(len(one))], 1) / len(one),
        batch1000=best_of(lambda: hybrid_one(te[:1000])) / min(1000, len(te)))

    # ---- one-off costs --------------------------------------------------------
    topos = [int(t) for t in np.unique(D["topo_id"][te])]
    res["oneoff_factorize_per_topology"] = best_of(lambda: [B.TopologyMatrices(t) for t in topos], 1) / len(topos)
    Yi = GP_Y(-1)
    def base_solves():
        for i in one[:100]:
            br = B.INTACT_ROW[int(D["base_id"][i])]
            Sbus = (D["p_gen"][br] - D["p_load"][br] - 1j * D["q_load"][br]) / B.GP.base_mva
            V0 = np.where(D["v_set"][br] > 0, D["v_set"][br], 1.0).astype(complex)
            newton(Yi, Sbus, V0)
    res["oneoff_preoutage_solve_per_base_case"] = best_of(base_solves, 1) / 100
    res["preoutage_solve_shared_by_contingencies"] = len(B.D["topo_id"][B.D["base_id"] == 0]) - 1

    json.dump(res, open(f"{C.RESULTS_DIR}/timing.json", "w"), indent=2)
    nr = res["newton_core"]["batch1"]
    print(f"\n{'method':28s}{'batch 1 (us/case)':>20s}{'batch 1000 (us/case)':>22s}{'vs Newton, batch 1':>20s}")
    for k, v in res.items():
        if isinstance(v, dict):
            b1, bN = v["batch1"] * 1e6, v.get("batch1000", np.nan) * 1e6
            print(f"{k:28s}{b1:>20.1f}{bN:>22.1f}{nr / v['batch1']:>19.1f}x")
    print(f"\nOne-off: factorization {res['oneoff_factorize_per_topology'] * 1e3:.2f} ms per topology; "
          f"pre-outage AC solve {res['oneoff_preoutage_solve_per_base_case'] * 1e3:.2f} ms per base case "
          f"(shared by {res['preoutage_solve_shared_by_contingencies']} contingencies)")
    print("Saved results/timing.json")


B.GP_BRANCHES = [("line", i) for i in range(B.GP.nl)] + [("trafo", i) for i in range(B.GP.nt)]
if C.INCLUDE_IMPEDANCE_BRANCHES:
    _imp = load_net().impedance.index
    B.GP_BRANCHES += [("impedance", int(i)) for i in _imp]


def GP_Y(t):
    return B.GP.matrices(t)[0].tocsr()


if __name__ == "__main__":
    main()
