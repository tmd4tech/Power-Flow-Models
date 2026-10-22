"""Phase 2, step 3: case-level security classification and contingency ranking.

Operators don't read bus-by-bus errors; they ask two questions per contingency
screen: "is this outage case secure?" and "which outages are the worst?"

CLASSIFICATION. Each outage case (one base case x one outage) gets four labels:
  V_any       any bus voltage beyond its limit
  V_serious   any bus more than 0.01 p.u. beyond its limit
  T_serious   any branch above 110% of its rating
  serious     V_serious or T_serious
Non-converged cases (no AC solution: likely voltage collapse) are insecure
under every label. The model's predicted label uses the same thresholds on its
predicted voltages and on loading computed from them (physics.py). Reported:
recall (insecure cases caught), false-alarm rate (secure cases flagged), and
recall on non-converged cases alone.

RANKING. Within each test base case, its outages are ranked by true severity:
  voltage severity = lowest bus voltage (lower = worse)
  thermal severity = highest branch loading (higher = worse)
Non-converged outages rank as the worst. Reported per base case and averaged:
Spearman correlation between predicted and true severity, and top-3 recall
(is the truly worst outage among the model's 3 worst?).
Topology-blind models receive identical inputs for every outage of a base case,
so their predictions are identical and they cannot rank. They are scored at
chance (Spearman 0, top-3 = 3/n) rather than dropped.

Run:  python case_analysis.py   -> results/case_summary.csv
"""

import csv
import glob
import os
from collections import defaultdict

import numpy as np

import config as C
from flow_analysis import predicted_flows
from metrics import VMAX, VMIN

D = np.load(f"{C.OUT_DIR}/dataset.npz")
NB = D["vm"].shape[1]
LABELS = ["V_any", "V_serious", "T_serious", "serious"]
V_SERIOUS, T_SERIOUS = 0.01, 110.0
TOPK = 3
REGIMES = ["blind_intact", "blind_n1", "status_n1", "hybrid_n1", "hybridv_n1", "gnn_n1", "physics"]
MODELS = ["linear", "mlp", "gnn", "intact_ac", "dc", "dc_inc", "fdlf1"]


def depth(vm):
    return np.maximum(VMIN - vm, vm - VMAX)


def case_labels(vm, loading):
    dv = depth(vm).max(1)
    vany = dv > C.V_TOL
    vser = dv > V_SERIOUS
    tser = loading.max(1) > T_SERIOUS
    return dict(V_any=vany, V_serious=vser, T_serious=tser, serious=vser | tser)


def rank_corr(a, b):
    """Spearman correlation without scipy's per-call overhead (no ties expected)."""
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


def ranking(base, sev_true, sev_pred):
    """Mean Spearman and top-k recall over base cases. Higher severity = worse."""
    rhos, hits = [], []
    for b in np.unique(base):
        m = base == b
        if m.sum() < TOPK + 1:
            continue
        st, sp = sev_true[m], sev_pred[m]
        n = m.sum()
        if np.ptp(sp) < 1e-9:                      # identical predictions: chance
            rhos.append(0.0); hits.append(TOPK / n); continue
        rhos.append(rank_corr(st, sp))
        worst = np.argmax(st)
        hits.append(float(worst in np.argsort(-sp)[:TOPK]))
    return float(np.mean(rhos)), float(np.mean(hits))


def analyze_run(P, ts):
    idx, pred = P[f"{ts}/idx"], P[f"{ts}/pred"].astype(float)
    nc_idx, nc_pred = P[f"{ts}/nc_idx"], P[f"{ts}/nc_pred"].astype(float)
    all_idx = np.concatenate([idx, nc_idx])
    all_pred = np.concatenate([pred, nc_pred])
    nc = np.r_[np.zeros(len(idx), bool), np.ones(len(nc_idx), bool)]

    load_pred, _, _ = predicted_flows(all_idx, all_pred)
    vm_pred = all_pred[:, :NB]
    vm_true = np.where(nc[:, None], 0.0, np.nan_to_num(D["vm"][all_idx]))   # nc -> worst
    load_true = np.where(nc[:, None], 1e9, np.nan_to_num(D["loading"][all_idx]))

    truth = case_labels(vm_true, load_true)
    predl = case_labels(vm_pred, load_pred)
    r = {}
    for L in LABELS:
        t, p = truth[L], predl[L]
        r[f"{L}_prevalence"] = float(t.mean())
        r[f"{L}_recall"] = float((t & p).sum() / t.sum()) if t.sum() else np.nan
        r[f"{L}_false_alarm"] = float((~t & p).sum() / (~t).sum()) if (~t).sum() else np.nan
    r["nc_cases"] = int(nc.sum())
    r["nc_recall"] = float(predl["serious"][nc].mean()) if nc.any() else np.nan

    base = D["base_id"][all_idx]
    r["rank_V_spearman"], r["rank_V_top3"] = ranking(base, -vm_true.min(1), -vm_pred.min(1))
    r["rank_T_spearman"], r["rank_T_top3"] = ranking(base, load_true.max(1), load_pred.max(1))
    return r


def main():
    acc = defaultdict(list)
    for f in sorted(glob.glob(f"{C.RESULTS_DIR}/preds/*.npz")):
        reg, mod, _, _ = os.path.basename(f)[:-4].split("__")
        P = np.load(f)
        for ts in ["test_seen", "test_unseen"]:
            acc[(reg, mod, ts)].append(analyze_run(P, ts))

    rows = []
    for (reg, mod, ts), runs in acc.items():
        row = dict(regime=reg, model=mod, test_set=ts, n_runs=len(runs))
        for k in runs[0]:
            v = [r[k] for r in runs]
            row[k] = float(np.nanmean(v))
        rows.append(row)
    rows.sort(key=lambda r: (REGIMES.index(r["regime"]), MODELS.index(r["model"]), r["test_set"]))
    with open(f"{C.RESULTS_DIR}/case_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    print("Case-level classification: recall / false-alarm rate (mean over folds x seeds)")
    print(f"{'regime':13s}{'model':7s}{'test set':12s}" + "".join(f"{L:>16s}" for L in LABELS)
          + f"{'non-conv. caught':>18s}")
    for r in rows:
        print(f"{r['regime']:13s}{r['model']:7s}{r['test_set']:12s}"
              + "".join(f"{r[L + '_recall']:>9.2f} /{r[L + '_false_alarm']:>5.2f}" for L in LABELS)
              + f"{r['nc_recall']:>9.2f} ({r['nc_cases']:.1f}/run)")
    print("  prevalence (share of cases insecure), unseen: "
          + ", ".join(f"{L} {rows[1][L + '_prevalence']:.2f}" for L in LABELS))

    print("\nContingency ranking within each base case (mean over base cases, then runs)")
    print(f"{'regime':13s}{'model':7s}{'test set':12s}{'V: Spearman':>12s}{'V: top-3':>10s}"
          f"{'T: Spearman':>13s}{'T: top-3':>10s}")
    for r in rows:
        print(f"{r['regime']:13s}{r['model']:7s}{r['test_set']:12s}{r['rank_V_spearman']:>12.2f}"
              f"{r['rank_V_top3']:>10.2f}{r['rank_T_spearman']:>13.2f}{r['rank_T_top3']:>10.2f}")
    print("\nSaved results/case_summary.csv")


if __name__ == "__main__":
    main()
