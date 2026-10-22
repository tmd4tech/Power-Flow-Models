"""Phase 2, step 1: are missed violations borderline or deep?

A sharp 0.94 p.u. threshold can make small prediction errors look like
missed violations or false alarms. This analysis separates the two:

  * Recall by violation DEPTH (how far beyond the limit the true voltage is).
    If models miss only borderline violations, the headline result is a
    threshold artifact. If they miss deep ones, it is a real failure.
  * Optimism bias: mean signed error (predicted - true voltage) on buses that
    truly violate. Positive = the model thinks the bus is healthier than it is.
  * Where false alarms come from: how close to the limit the true voltage was.
  * Precision with a near-miss tolerance: an alarm counts as useful if the true
    voltage is beyond the limit OR within TOL of it (an operator would want to
    know about a bus at 0.942 p.u. after an outage).
  * False-alarm rate on intact (secure) cases: share of test_id cases where the
    model flags any violation. Replaces the undefined recall/precision there.

Reads results/preds/*.npz (saved by run_experiments.py); no retraining.
Run:  python depth_analysis.py   -> results/depth_summary.csv
"""

import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

import config as C

D = np.load(f"{C.OUT_DIR}/dataset.npz")
META = json.load(open(f"{C.OUT_DIR}/dataset_meta.json"))
VMIN, VMAX = np.array(META["vmin"]), np.array(META["vmax"])
NB = len(VMIN)

BINS = [(0.0, 0.005), (0.005, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, np.inf)]
NEAR_TOL = 0.005      # p.u.; alarms this close to the limit count as useful
REGIMES = ["blind_intact", "blind_n1", "status_n1", "hybrid_n1", "hybridv_n1", "gnn_n1", "physics"]
MODELS = ["linear", "mlp", "gnn", "intact_ac", "dc", "dc_inc", "fdlf1"]


def depth_beyond_limit(vm):
    """Positive = p.u. beyond the nearest violated limit; negative = margin inside."""
    return np.maximum(VMIN - vm, vm - VMAX)


def analyze(vm_pred, vm_true):
    dt = depth_beyond_limit(vm_true)
    dp = depth_beyond_limit(vm_pred)
    t, p = dt > C.V_TOL, dp > C.V_TOL
    r = {}
    for lo, hi in BINS:
        m = t & (dt > lo) & (dt <= hi)
        r[f"recall_{lo:g}-{hi:g}"] = (p & m).sum() / m.sum() if m.sum() else np.nan
        r[f"n_{lo:g}-{hi:g}"] = int(m.sum())
    r["recall_all"] = (p & t).sum() / t.sum() if t.sum() else np.nan
    r["optimism_bias_pu"] = float((vm_pred - vm_true)[t].mean()) if t.sum() else np.nan
    fa = p & ~t
    r["false_alarms"] = int(fa.sum())
    r["fa_share_within_tol"] = (fa & (dt > -NEAR_TOL)).sum() / fa.sum() if fa.sum() else np.nan
    r["precision_strict"] = (p & t).sum() / p.sum() if p.sum() else np.nan
    r["precision_near_miss"] = (p & (dt > -NEAR_TOL)).sum() / p.sum() if p.sum() else np.nan
    return r


def main():
    acc = defaultdict(list)          # (regime, model, test_set) -> list of per-run dicts
    fa_rate = defaultdict(list)      # (regime, model) -> per-run case-level FA rate on test_id
    for f in sorted(glob.glob(f"{C.RESULTS_DIR}/preds/*.npz")):
        reg, mod, _, _ = os.path.basename(f)[:-4].split("__")
        P = np.load(f)
        for ts in ["test_seen", "test_unseen"]:
            idx, pred = P[f"{ts}/idx"], P[f"{ts}/pred"].astype(float)
            acc[(reg, mod, ts)].append(analyze(pred[:, :NB], D["vm"][idx]))
        idx, pred = P["test_id/idx"], P["test_id/pred"].astype(float)
        flagged = (depth_beyond_limit(pred[:, :NB]) > C.V_TOL).any(1)
        fa_rate[(reg, mod)].append(flagged.mean())

    rows = []
    for (reg, mod, ts), runs in acc.items():
        row = dict(regime=reg, model=mod, test_set=ts, n_runs=len(runs))
        for k in runs[0]:
            row[k] = float(np.nanmean([r[k] for r in runs]))
        row["intact_case_false_alarm_rate"] = float(np.mean(fa_rate[(reg, mod)]))
        rows.append(row)
    rows.sort(key=lambda r: (REGIMES.index(r["regime"]), MODELS.index(r["model"]), r["test_set"]))
    with open(f"{C.RESULTS_DIR}/depth_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    bins = [f"{lo:g}-{hi:g}" for lo, hi in BINS]
    print("Recall by violation depth (p.u. below the limit), mean over folds x seeds")
    print(f"{'regime':13s}{'model':7s}{'test set':12s}" + "".join(f"{b:>11s}" for b in bins)
          + f"{'all':>7s}{'bias p.u.':>11s}")
    for r in rows:
        print(f"{r['regime']:13s}{r['model']:7s}{r['test_set']:12s}"
              + "".join(f"{r['recall_' + b]:>11.2f}" for b in bins)
              + f"{r['recall_all']:>7.2f}{r['optimism_bias_pu']:>+11.4f}")
    r0 = rows[0]
    print("  true violations per bin (one run): "
          + ", ".join(f"{b}: {r0['n_' + b]}" for b in bins))

    print("\nFalse alarms")
    print(f"{'regime':13s}{'model':7s}{'test set':12s}{'precision':>10s}{'near-miss prec.':>16s}"
          f"{'FA within 0.005':>16s}{'intact-case FA rate':>20s}")
    for r in rows:
        print(f"{r['regime']:13s}{r['model']:7s}{r['test_set']:12s}{r['precision_strict']:>10.2f}"
              f"{r['precision_near_miss']:>16.2f}{r['fa_share_within_tol']:>16.2f}"
              f"{r['intact_case_false_alarm_rate']:>20.3f}")
    print("\nSaved results/depth_summary.csv")


if __name__ == "__main__":
    main()
