"""Turn results/runs/*.json into the tables for the paper.

Writes:
  results/summary.csv      mean and std over folds x seeds, per regime/model/test set
  results/by_line.csv      unseen-outage metrics per line, pooled over seeds
and prints the main table.

Recall and precision are pooled from counts (tp/fn/fp) within each run, then
averaged across runs, so small test sets don't get the same weight as large
ones within a run.

Run:  python report.py
"""

import csv
import glob
import json
from collections import defaultdict

import numpy as np

import config as C
from scipy.stats import spearmanr

META = json.load(open(f"{C.OUT_DIR}/dataset_meta.json"))
NAMES = META["branches"]

RUNS = [json.load(open(f)) for f in sorted(glob.glob(f"{C.RESULTS_DIR}/runs/*.json"))]
TEST_SETS = ["test_id", "test_seen", "test_unseen"]
REGIMES = ["blind_intact", "blind_n1", "status_n1", "hybrid_n1", "hybridv_n1", "gnn_n1", "physics"]
MODELS = ["linear", "mlp", "gnn", "intact_ac", "dc", "dc_inc", "fdlf1"]
METRICS = ["vm_mae_pu", "va_mae_deg", "recall", "precision"]


def main():
    groups = defaultdict(list)
    for r in RUNS:
        groups[(r["regime"], r["model"])].append(r)

    rows = []
    for reg in REGIMES:
        for mod in MODELS:
            runs = groups.get((reg, mod), [])
            if not runs:
                continue
            for ts in TEST_SETS:
                row = dict(regime=reg, model=mod, test_set=ts, n_runs=len(runs))
                for m in METRICS:
                    v = np.array([r["results"][ts][m] for r in runs], dtype=float)
                    row[m + "_mean"], row[m + "_std"] = np.nanmean(v), np.nanstd(v)
                rows.append(row)

    with open(f"{C.RESULTS_DIR}/summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    print(f"{len(RUNS)} runs. Mean ± std over folds x seeds.\n")
    print(f"{'regime':13s}{'model':7s}{'test set':12s}{'|V| err (p.u.)':>18s}"
          f"{'angle err (deg)':>18s}{'viol recall':>14s}{'precision':>12s}")
    for row in rows:
        print(f"{row['regime']:13s}{row['model']:7s}{row['test_set']:12s}"
              f"{row['vm_mae_pu_mean']:>10.4f} ± {row['vm_mae_pu_std']:.4f}"
              f"{row['va_mae_deg_mean']:>11.3f} ± {row['va_mae_deg_std']:.3f}"
              f"{row['recall_mean']:>8.2f} ± {row['recall_std']:.2f}"
              f"{row['precision_mean']:>7.2f} ± {row['precision_std']:.2f}")

    # per-line breakdown on unseen outages (each line is unseen in exactly one fold)
    per_line = defaultdict(lambda: defaultdict(lambda: np.zeros(3)))
    for r in RUNS:
        for line, m in r["by_topology"]["test_unseen"].items():
            per_line[(r["regime"], r["model"])][line] += [m["tp"], m["fn"], m["fp"]]
    print("\nSeverity vs. detection on unseen outages (one point per outage):")
    print("  Spearman correlation between the outage's true violation count and recall.")
    with open(f"{C.RESULTS_DIR}/by_line.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["regime", "model", "branch", "true_violations", "recall", "precision"])
        for (reg, mod), lines in sorted(per_line.items()):
            sev, rec_all = [], []
            for line, (tp, fn, fp) in sorted(lines.items(), key=lambda x: int(x[0])):
                rec = tp / (tp + fn) if tp + fn else float("nan")
                prec = tp / (tp + fp) if tp + fp else float("nan")
                w.writerow([reg, mod, NAMES[int(line)], int(tp + fn), f"{rec:.3f}", f"{prec:.3f}"])
                if tp + fn:
                    sev.append(tp + fn); rec_all.append(rec)
            rho, p = spearmanr(sev, rec_all)
            worst = min(((NAMES[int(l)], c[0] / (c[0] + c[1])) for l, c in lines.items() if c[0] + c[1]),
                        key=lambda x: x[1])
            print(f"  {reg:13s}{mod:7s} rho = {rho:+.2f} (p = {p:.3f}, {len(sev)} outages); "
                  f"worst: {worst[0]} at {worst[1]:.0%} recall")
    print("\nSaved results/summary.csv and results/by_line.csv")


if __name__ == "__main__":
    main()
