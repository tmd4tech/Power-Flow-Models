"""Phase 2, step 2: thermal overloads and physical consistency.

The models predict voltages only. Branch loading is computed from the predicted
voltages on the true topology (physics.py, validated to match pandapower), so
overload detection uses exactly the ground truth's definition.

Thermal metrics (on the 38 rated branches):
  * recall by overload DEPTH (true loading 100-102%, 102-110%, 110-125%, >125%);
    the OPF leaves line 0 at exactly its rating, so many post-outage overloads
    are tiny, and depth separates those from serious ones;
  * strict precision, and near-miss precision (an alarm counts as useful if the
    true loading is above 95%);
  * case-level false-alarm rate on intact (secure) test cases.

Physical consistency:
  * mean |dP| (MW) over non-slack buses and mean |dQ| (Mvar) over buses without
    generators, from the AC power-flow equations. The true solution gives ~0.

Run:  python physics.py        (validation, once)
      python flow_analysis.py  -> results/flow_summary.csv
"""

import csv
import glob
import os
from collections import defaultdict

import numpy as np

import config as C
from physics import GridPhysics, injections

D = np.load(f"{C.OUT_DIR}/dataset.npz")
GP = GridPhysics()
NB = D["vm"].shape[1]
BINS = [(100, 102), (102, 110), (110, 125), (125, np.inf)]
NEAR = 95.0
REGIMES = ["blind_intact", "blind_n1", "status_n1", "hybrid_n1", "hybridv_n1", "gnn_n1", "physics"]
MODELS = ["linear", "mlp", "gnn", "intact_ac", "dc", "dc_inc", "fdlf1"]


def limit():
    return C.THERMAL_LIMIT + C.THERMAL_TOL


def predicted_flows(idx, pred):
    """Predicted loading and mismatch for samples idx, grouped by topology."""
    vm, va = pred[:, :NB], pred[:, NB:]
    load = np.zeros((len(idx), GP.n_rated))
    dP = np.full((len(idx), NB), np.nan)
    dQ = np.full((len(idx), NB), np.nan)
    topo = D["topo_id"][idx]
    p_inj, q_inj = injections(D, idx)
    for t in np.unique(topo):
        m = topo == t
        load[m] = GP.loading(vm[m], va[m], int(t))
        dP[m], dQ[m] = GP.mismatch(vm[m], va[m], p_inj[m], q_inj[m], int(t))
    return load, dP, dQ


def thermal_metrics(load_pred, load_true):
    t, p = load_true > limit(), load_pred > limit()
    r = {}
    for lo, hi in BINS:
        m = t & (load_true > lo) & (load_true <= hi)
        r[f"recall_{lo}-{hi:g}"] = (p & m).sum() / m.sum() if m.sum() else np.nan
        r[f"n_{lo}-{hi:g}"] = int(m.sum())
    r["recall_all"] = (p & t).sum() / t.sum() if t.sum() else np.nan
    r["precision_strict"] = (p & t).sum() / p.sum() if p.sum() else np.nan
    r["precision_near_miss"] = (p & (load_true > NEAR)).sum() / p.sum() if p.sum() else np.nan
    r["loading_mae_pct"] = float(np.abs(load_pred - load_true).mean())
    return r


def main():
    acc = defaultdict(list)
    fa_rate = defaultdict(list)
    for f in sorted(glob.glob(f"{C.RESULTS_DIR}/preds/*.npz")):
        reg, mod, _, _ = os.path.basename(f)[:-4].split("__")
        P = np.load(f)
        for ts in ["test_seen", "test_unseen"]:
            idx, pred = P[f"{ts}/idx"], P[f"{ts}/pred"].astype(float)
            load, dP, dQ = predicted_flows(idx, pred)
            r = thermal_metrics(load, D["loading"][idx])
            r["mismatch_P_MW"] = float(np.nanmean(np.abs(dP)))
            r["mismatch_Q_Mvar"] = float(np.nanmean(np.abs(dQ)))
            acc[(reg, mod, ts)].append(r)
        idx, pred = P["test_id/idx"], P["test_id/pred"].astype(float)
        load, dP, dQ = predicted_flows(idx, pred)
        fa_rate[(reg, mod)].append(dict(
            fa=float((load > limit()).any(1).mean()),
            mP=float(np.nanmean(np.abs(dP))), mQ=float(np.nanmean(np.abs(dQ)))))

    rows = []
    for (reg, mod, ts), runs in acc.items():
        row = dict(regime=reg, model=mod, test_set=ts, n_runs=len(runs))
        for k in runs[0]:
            row[k] = float(np.nanmean([r[k] for r in runs]))
        row["intact_case_overload_FA_rate"] = float(np.mean([x["fa"] for x in fa_rate[(reg, mod)]]))
        row["intact_mismatch_P_MW"] = float(np.mean([x["mP"] for x in fa_rate[(reg, mod)]]))
        row["intact_mismatch_Q_Mvar"] = float(np.mean([x["mQ"] for x in fa_rate[(reg, mod)]]))
        rows.append(row)
    rows.sort(key=lambda r: (REGIMES.index(r["regime"]), MODELS.index(r["model"]), r["test_set"]))
    with open(f"{C.RESULTS_DIR}/flow_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    bins = [f"{lo}-{hi:g}" for lo, hi in BINS]
    print("Thermal overload recall by true loading (% of rating), mean over folds x seeds")
    print(f"{'regime':13s}{'model':7s}{'test set':12s}" + "".join(f"{b:>10s}" for b in bins)
          + f"{'all':>6s}{'prec':>6s}{'near-miss':>10s}{'intact FA':>10s}{'load MAE%':>10s}")
    for r in rows:
        print(f"{r['regime']:13s}{r['model']:7s}{r['test_set']:12s}"
              + "".join(f"{r['recall_' + b]:>10.2f}" for b in bins)
              + f"{r['recall_all']:>6.2f}{r['precision_strict']:>6.2f}{r['precision_near_miss']:>10.2f}"
              f"{r['intact_case_overload_FA_rate']:>10.3f}{r['loading_mae_pct']:>10.2f}")
    r0 = rows[0]
    print("  true overloads per bin (one run): " + ", ".join(f"{b}%: {r0['n_' + b]}" for b in bins))

    print("\nPower-flow mismatch of predicted voltages (true solution: ~1e-6)")
    print(f"{'regime':13s}{'model':7s}{'test set':12s}{'|dP| MW':>10s}{'|dQ| Mvar':>11s}"
          f"{'intact |dP|':>13s}{'intact |dQ|':>13s}")
    for r in rows:
        print(f"{r['regime']:13s}{r['model']:7s}{r['test_set']:12s}{r['mismatch_P_MW']:>10.2f}"
              f"{r['mismatch_Q_Mvar']:>11.2f}{r['intact_mismatch_P_MW']:>13.2f}{r['intact_mismatch_Q_Mvar']:>13.2f}")
    mean_load = np.abs(D["p_load"][D["converged"]]).mean()
    print(f"  scale: mean |P load| per bus = {mean_load:.2f} MW")
    print("\nSaved results/flow_summary.csv")


if __name__ == "__main__":
    main()
