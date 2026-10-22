"""Phase 3, step 1: physics baselines for contingency screening.

An operator screening contingencies knows the pre-outage AC state (from state
estimation). In this dataset that is the intact solution of each base case.
Each baseline maps (inputs, topology, pre-outage state) to predicted [vm | va]
for the outaged grid, so every existing analysis scores it unchanged:

  intact_ac  pre-outage AC state, unchanged. What a perfectly accurate
             topology-BLIND model would output: the ceiling for blind models.
  dc         DC power flow on the outaged grid: angles from B*theta = P,
             vm = 1.0 except generator buses at their setpoints. Voltage-blind
             by construction; included for thermal screening.
  dc_inc     pre-outage AC state + DC estimate of the outage's angle change
             (theta_dc_outaged - theta_dc_intact). Flow changes equal the
             classical LODF/DC-sensitivity screening. Voltage magnitudes are
             unchanged from the pre-outage state.
  fdlf1      one fast-decoupled (XB) P-theta / Q-V iteration on the outaged
             grid, from the pre-outage AC state ("1P1Q" screening). Generator
             buses stay at their voltage setpoints; reactive limits are NOT
             modeled (classical formulation), so it can miss voltage drops
             caused by generators hitting Q limits.

FAIRNESS NOTE: these baselines use the pre-outage AC state, i.e. one AC
solution per base case. The learned models were given only injections and
setpoints. On intact test cases the baselines therefore reproduce the truth
exactly. A learned model given the same pre-outage state is a Phase 3 hybrid.

Writes results/preds/physics__<model>__fold<k>__seed0.npz and the matching
results/runs/*.json, so report.py and every Phase 2 script include them.
Also writes results/baseline_timing.json (seconds per contingency evaluated).

Run:  python baselines.py
"""

import json
import os
import time
import warnings

import numpy as np
from pandapower.pypower.idx_brch import BR_STATUS
from pandapower.pypower.idx_bus import BUS_TYPE
from pandapower.pypower.makeB import makeB
from pandapower.pypower.makeBdc import makeBdc
from scipy.sparse.linalg import splu

import config as C
from metrics import evaluate, evaluate_by_topology
from physics import GridPhysics

warnings.filterwarnings("ignore")

D = np.load(f"{C.OUT_DIR}/dataset.npz")
S = np.load(f"{C.OUT_DIR}/splits.npz")
NB = D["vm"].shape[1]
GP = GridPhysics()
BUS_TYPES = GP.bus[:, BUS_TYPE].real.astype(int)
REF = np.where(BUS_TYPES == 3)[0]
PV = np.where(BUS_TYPES == 2)[0]
PQ = np.where(BUS_TYPES == 1)[0]
PVPQ = np.r_[PV, PQ]
NONREF = np.setdiff1d(np.arange(NB), REF)
MODELS = ["intact_ac", "dc", "dc_inc", "fdlf1"]
TEST_SETS = ["test_id", "test_seen", "test_unseen"]

# pre-outage AC state of every base case = its intact sample
INTACT_ROW = {int(b): i for i, b in enumerate(D["base_id"]) if D["topo_id"][i] == -1}


def branch_for(topo_id):
    br = GP.branch.copy()
    if topo_id >= 0:
        br[topo_id, BR_STATUS] = 0
    return br


class TopologyMatrices:
    """DC and fast-decoupled matrices for one topology, factorized once."""

    def __init__(self, topo_id):
        br = branch_for(topo_id)
        Bbus, _, Pbusinj, _, _ = makeBdc(GP.bus, br)
        self.Pbusinj = np.asarray(Pbusinj).ravel()
        self.dc_lu = splu(Bbus[NONREF][:, NONREF].tocsc())
        Bp, Bpp = makeB(GP.base_mva, GP.bus, br, 2)          # alg 2 = XB
        self.bp_lu = splu(Bp[PVPQ][:, PVPQ].tocsc())
        self.bpp_lu = splu(Bpp[PQ][:, PQ].tocsc())
        # kept SPARSE: a dense Ybus made each mismatch evaluation cost O(N^2) per
        # case (9x slower on case118 than case30); sparse is ~O(N), as in real
        # power-flow codes. The arithmetic is identical.
        self.Ybus = GP.matrices(topo_id)[0].tocsr()

    def dc_angles(self, p_inj_mw):
        """DC angles (rad) for each row of net injections (MW); reference = 0."""
        P = p_inj_mw / GP.base_mva - self.Pbusinj
        th = np.zeros_like(P)
        th[:, NONREF] = self.dc_lu.solve(P[:, NONREF].T).T
        return th

    def fdlf_one_iteration(self, vm0, va0_rad, p_inj_mw, q_inj_mvar):
        Sbus = (p_inj_mw + 1j * q_inj_mvar) / GP.base_mva
        vm, va = vm0.copy(), va0_rad.copy()

        def mismatch():
            V = vm * np.exp(1j * va)
            return (V * np.conj((self.Ybus @ V.T).T) - Sbus) / vm

        P = mismatch()[:, PVPQ].real
        va[:, PVPQ] -= self.bp_lu.solve(P.T).T
        Q = mismatch()[:, PQ].imag
        vm[:, PQ] -= self.bpp_lu.solve(Q.T).T
        return vm, va


_MATS = {}


def mats(t):
    if t not in _MATS:
        _MATS[t] = TopologyMatrices(t)
    return _MATS[t]


def predict(model, idx):
    """Predicted [vm | va(deg)] for dataset rows idx."""
    base_rows = np.array([INTACT_ROW[int(b)] for b in D["base_id"][idx]])
    vm0, va0 = D["vm"][base_rows], D["va"][base_rows]
    if model == "intact_ac":
        return np.concatenate([vm0, va0], axis=1)
    p_inj = D["p_gen"][idx] - D["p_load"][idx]
    q_inj = -D["q_load"][idx]
    out = np.zeros((len(idx), 2 * NB))
    topo = D["topo_id"][idx]
    for t in np.unique(topo):
        m = topo == t
        M = mats(int(t))
        if model == "dc":
            vm = np.ones((m.sum(), NB))
            vset = D["v_set"][idx][m]
            vm = np.where(vset > 0, vset, vm)
            out[m] = np.concatenate([vm, np.rad2deg(M.dc_angles(p_inj[m]))], axis=1)
        elif model == "dc_inc":
            dth = M.dc_angles(p_inj[m]) - mats(-1).dc_angles(p_inj[m])
            out[m] = np.concatenate([vm0[m], va0[m] + np.rad2deg(dth)], axis=1)
        elif model == "fdlf1":
            vm, va = M.fdlf_one_iteration(vm0[m], np.deg2rad(va0[m]), p_inj[m], q_inj[m])
            out[m] = np.concatenate([vm, np.rad2deg(va)], axis=1)
    return out


def main():
    os.makedirs(f"{C.RESULTS_DIR}/preds", exist_ok=True)
    os.makedirs(f"{C.RESULTS_DIR}/runs", exist_ok=True)
    conv = D["converged"]
    timing = {}
    for model in MODELS:
        t_total, n_total = 0.0, 0
        for fold in range(5):
            preds, res = {}, dict(regime="physics", model=model, fold=fold, seed=0, params={},
                                   fit_seconds=0.0, results={}, by_topology={}, non_converged={})
            for ts in TEST_SETS:
                allidx = S[f"n1_fold_{fold}/{ts}"]
                i, nc = allidx[conv[allidx]], allidx[~conv[allidx]]
                t0 = time.perf_counter()
                pred = predict(model, i)
                t_total += time.perf_counter() - t0; n_total += len(i)
                Y = np.concatenate([D["vm"][i], D["va"][i]], axis=1)
                res["results"][ts] = evaluate(pred, Y, NB)
                res["non_converged"][ts] = int(len(nc))
                if ts != "test_id":
                    res["by_topology"][ts] = evaluate_by_topology(pred, Y, D["topo_id"][i], NB)
                preds[f"{ts}/idx"] = i
                preds[f"{ts}/pred"] = pred.astype(np.float32)
                preds[f"{ts}/nc_idx"] = nc
                preds[f"{ts}/nc_pred"] = (predict(model, nc).astype(np.float32) if len(nc)
                                          else np.zeros((0, 2 * NB), np.float32))
            name = f"physics__{model}__fold{fold}__seed0"
            np.savez_compressed(f"{C.RESULTS_DIR}/preds/{name}.npz", **preds)
            json.dump(res, open(f"{C.RESULTS_DIR}/runs/{name}.json", "w"), indent=1)
        timing[model] = t_total / n_total
        u = res["results"]["test_unseen"]
        print(f"  {model:10s} fold 4 unseen: |V| err {u['vm_mae_pu']:.4f} p.u., "
              f"angle err {u['va_mae_deg']:.3f} deg, bus recall {u['recall']:.2f}  "
              f"({timing[model] * 1e6:.0f} us per case)")
    json.dump({k: dict(seconds_per_case=v) for k, v in timing.items()},
              open(f"{C.RESULTS_DIR}/baseline_timing.json", "w"), indent=2)
    print("Saved results/preds/physics__*, results/runs/physics__*, results/baseline_timing.json")


if __name__ == "__main__":
    main()