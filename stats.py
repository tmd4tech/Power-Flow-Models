"""Phase 2, step 4: statistics.

1. VARIANCE DECOMPOSITION. The "mean +/- std over folds x seeds" in report.py
   mixes two different things. Here they are separated:
     fold SD = spread of fold means (which outages are held out)
     seed SD = average spread across seeds within a fold (initialization; MLP only)
2. CONFIDENCE INTERVALS by a TWO-WAY bootstrap: test base cases AND outages are
   resampled with replacement. With only 37 outages, generalization to unseen
   outages depends on which outages exist; resampling only base cases would
   understate the uncertainty. Intact test cases resample base cases only.
3. PAIRED TESTS. Every model is evaluated on the same bootstrap draws, so
   differences between models (and between seen and unseen for one model) are
   computed on identical resampled test sets. Two-sided bootstrap p-values.

Metrics (all from saved predictions; non-converged cases count as seriously
insecure in the case-level metrics and are excluded from error metrics):
  vm_mae, va_mae          bus-level error (p.u., degrees)
  bus_recall              voltage violations caught, bus level
  serious_recall          seriously insecure cases caught (see case_analysis.py)
  serious_false_alarm     secure-under-"serious" cases flagged
  rank_V                  Spearman of predicted vs true voltage severity across
                          outages within a base case (chance = 0); bootstrapped
                          over base cases only, since it is defined per base case

Run:  python stats.py   -> results/stats_ci.csv, stats_paired.csv, stats_variance.csv
"""

import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

import config as C
from case_analysis import case_labels, rank_corr
from flow_analysis import predicted_flows
from metrics import violations

B = 2000                      # bootstrap replicates
D = np.load(f"{C.OUT_DIR}/dataset.npz")
S = np.load(f"{C.OUT_DIR}/splits.npz")
META = json.load(open(f"{C.OUT_DIR}/dataset_meta.json"))
NB = D["vm"].shape[1]
TEST_BASES = S["bases/test"]
BPOS = {b: k for k, b in enumerate(TEST_BASES)}
OUTAGES = META["outage_topologies"]
OPOS = {t: k for k, t in enumerate(OUTAGES)}
OPOS[-1] = 0                  # intact cases use a single outage slot
TEST_SETS = ["test_id", "test_seen", "test_unseen"]
REGIMES = ["blind_intact", "blind_n1", "status_n1", "hybrid_n1", "hybridv_n1", "gnn_n1", "physics"]
MODELS = ["linear", "mlp", "gnn", "intact_ac", "dc", "dc_inc", "fdlf1"]

# per-cell sufficient statistics (summed over runs), in this order:
STATS = ["vm_abs", "va_abs", "n_bus", "bus_tp", "bus_pos",
         "case_tp", "case_pos", "case_fp", "case_neg"]
SI = {s: k for k, s in enumerate(STATS)}
METRICS = ["vm_mae", "va_mae", "bus_recall", "serious_recall", "serious_false_alarm"]


def metrics_from(st):
    """Ratio metrics from summed statistics (last axis = STATS)."""
    def ratio(a, b):
        with np.errstate(invalid="ignore", divide="ignore"):
            return st[..., SI[a]] / st[..., SI[b]]
    return dict(vm_mae=ratio("vm_abs", "n_bus") , va_mae=ratio("va_abs", "n_bus"),
                bus_recall=ratio("bus_tp", "bus_pos"),
                serious_recall=ratio("case_tp", "case_pos"),
                serious_false_alarm=ratio("case_fp", "case_neg"))


def per_sample_stats(P, ts):
    """Statistics for every test sample of one run (converged + non-converged)."""
    idx, pred = P[f"{ts}/idx"], P[f"{ts}/pred"].astype(float)
    if ts != "test_id":
        nc_idx, nc_pred = P[f"{ts}/nc_idx"], P[f"{ts}/nc_pred"].astype(float)
    else:
        nc_idx, nc_pred = np.zeros(0, int), np.zeros((0, 2 * NB))
    all_idx = np.concatenate([idx, nc_idx]).astype(int)
    all_pred = np.concatenate([pred, nc_pred])
    nc = np.r_[np.zeros(len(idx), bool), np.ones(len(nc_idx), bool)]

    vm_t = np.nan_to_num(D["vm"][all_idx]); va_t = np.nan_to_num(D["va"][all_idx])
    vm_p, va_p = all_pred[:, :NB], all_pred[:, NB:]
    st = np.zeros((len(all_idx), len(STATS)))
    ok = ~nc
    st[ok, SI["vm_abs"]] = np.abs(vm_p - vm_t)[ok].sum(1)
    st[ok, SI["va_abs"]] = np.abs(va_p - va_t)[ok].sum(1)
    st[ok, SI["n_bus"]] = NB
    tv, pv = violations(vm_t), violations(vm_p)
    st[ok, SI["bus_tp"]] = (tv & pv)[ok].sum(1)
    st[ok, SI["bus_pos"]] = tv[ok].sum(1)

    load_p, _, _ = predicted_flows(all_idx, all_pred)
    vm_case = np.where(nc[:, None], 0.0, vm_t)
    load_case = np.where(nc[:, None], 1e9, np.nan_to_num(D["loading"][all_idx]))
    truth = case_labels(vm_case, load_case)["serious"]
    predl = case_labels(vm_p, load_p)["serious"]
    st[:, SI["case_tp"]] = truth & predl
    st[:, SI["case_pos"]] = truth
    st[:, SI["case_fp"]] = ~truth & predl
    st[:, SI["case_neg"]] = ~truth

    # per-base-case ranking (voltage severity; non-converged = worst)
    rank = {}
    if ts != "test_id":
        base = D["base_id"][all_idx]
        sev_t, sev_p = -vm_case.min(1), -vm_p.min(1)
        for b in np.unique(base):
            m = base == b
            if m.sum() < 4:
                continue
            rank[b] = 0.0 if np.ptp(sev_p[m]) < 1e-9 else rank_corr(sev_t[m], sev_p[m])
    return all_idx, st, rank


def main():
    rng = np.random.default_rng(C.SEED + 7)
    n_b, n_o = len(TEST_BASES), len(OUTAGES)
    cells = defaultdict(lambda: np.zeros((n_b, n_o, len(STATS))))
    rank_by_base = defaultdict(lambda: defaultdict(list))
    per_run = defaultdict(list)          # (reg, mod, ts) -> [(fold, seed, metrics)]

    files = sorted(glob.glob(f"{C.RESULTS_DIR}/preds/*.npz"))
    for k, f in enumerate(files):
        reg, mod, fold, seed = os.path.basename(f)[:-4].split("__")
        fold, seed = int(fold[4:]), int(seed[4:])
        P = np.load(f)
        for ts in TEST_SETS:
            idx, st, rank = per_sample_stats(P, ts)
            bp = np.array([BPOS[b] for b in D["base_id"][idx]])
            op = np.array([OPOS[t] for t in D["topo_id"][idx]])
            np.add.at(cells[(reg, mod, ts)], (bp, op), st)
            m = {k2: float(v) for k2, v in metrics_from(st.sum(0)).items()}
            for b, r in rank.items():
                rank_by_base[(reg, mod, ts)][b].append(r)
            m["rank_V"] = float(np.mean(list(rank.values()))) if rank else np.nan
            per_run[(reg, mod, ts)].append((fold, seed, m))
        print(f"  processed {k + 1}/{len(files)} runs", end="\r", flush=True)
    print()

    keys = [(r, m, ts) for r in REGIMES for m in MODELS for ts in TEST_SETS if (r, m, ts) in cells]

    # ---- bootstrap: shared weights across all models -------------------------
    wb = rng.multinomial(n_b, np.full(n_b, 1 / n_b), size=B).astype(float)
    wo = rng.multinomial(n_o, np.full(n_o, 1 / n_o), size=B).astype(float)
    boot, point = {}, {}
    for key in keys:
        c = cells[key]
        intact = key[2] == "test_id"
        w_o = np.ones((B, n_o)) if intact else wo     # intact: one slot, bases only
        boot_st = np.einsum("rb,ro,bos->rs", wb, w_o, c)
        boot[key] = metrics_from(boot_st)
        point[key] = {k2: float(v) for k2, v in metrics_from(c.sum((0, 1))).items()}
        if not intact:
            rb = rank_by_base[key]
            vals = np.array([np.mean(rb[b]) if b in rb else np.nan for b in TEST_BASES])
            ok = ~np.isnan(vals)
            point[key]["rank_V"] = float(vals[ok].mean())
            boot[key]["rank_V"] = (wb[:, ok] @ vals[ok]) / wb[:, ok].sum(1)

    def ci(x):
        x = x[~np.isnan(x)]
        return (np.percentile(x, 2.5), np.percentile(x, 97.5)) if len(x) else (np.nan, np.nan)

    rows = []
    for key in keys:
        for met in METRICS + ["rank_V"]:
            if met not in point[key]:
                continue
            lo, hi = ci(boot[key][met])
            rows.append(dict(regime=key[0], model=key[1], test_set=key[2], metric=met,
                             estimate=point[key][met], ci_low=lo, ci_high=hi))
    with open(f"{C.RESULTS_DIR}/stats_ci.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # ---- variance decomposition -------------------------------------------
    vrows = []
    for key in keys:
        runs = per_run[key]
        folds = sorted({f for f, _, _ in runs})
        for met in METRICS + ["rank_V"]:
            by_fold = [[m[met] for f, _, m in runs if f == fo] for fo in folds]
            fold_means = [np.nanmean(v) for v in by_fold]
            fold_sd = float(np.nanstd(fold_means))
            seed_sd = (float(np.nanmean([np.nanstd(v) for v in by_fold]))
                       if max(len(v) for v in by_fold) > 1 else np.nan)
            vrows.append(dict(regime=key[0], model=key[1], test_set=key[2], metric=met,
                              mean=float(np.nanmean(fold_means)), fold_sd=fold_sd, seed_sd=seed_sd))
    with open(f"{C.RESULTS_DIR}/stats_variance.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(vrows[0])); w.writeheader(); w.writerows(vrows)

    # ---- paired comparisons ------------------------------------------------
    pairs = [(("status_n1", "mlp"), ("status_n1", "linear")),
             (("status_n1", "mlp"), ("blind_n1", "mlp")),
             (("status_n1", "linear"), ("blind_n1", "linear")),
             (("blind_n1", "mlp"), ("blind_intact", "mlp")),
             (("physics", "fdlf1"), ("status_n1", "mlp")),
             (("physics", "dc_inc"), ("status_n1", "mlp")),
             (("physics", "fdlf1"), ("physics", "dc_inc")),
             (("blind_n1", "mlp"), ("physics", "intact_ac")),
             (("hybrid_n1", "mlp"), ("physics", "fdlf1")),
             (("hybrid_n1", "linear"), ("physics", "fdlf1")),
             (("hybrid_n1", "mlp"), ("hybrid_n1", "linear")),
             (("hybrid_n1", "mlp"), ("status_n1", "mlp")),
             (("hybridv_n1", "mlp"), ("physics", "fdlf1")),
             (("hybridv_n1", "linear"), ("physics", "fdlf1")),
             (("hybridv_n1", "mlp"), ("hybrid_n1", "mlp")),
             (("hybridv_n1", "linear"), ("hybridv_n1", "mlp")),
             (("hybridv_n1", "linear"), ("hybrid_n1", "mlp")),
             (("gnn_n1", "gnn"), ("status_n1", "mlp")),
             (("gnn_n1", "gnn"), ("physics", "fdlf1")),
             (("hybridv_n1", "linear"), ("gnn_n1", "gnn"))]
    comps = [(a + (ts,), b + (ts,), f"{a[0]}/{a[1]} - {b[0]}/{b[1]}")
             for a, b in pairs for ts in ["test_seen", "test_unseen"]]
    comps += [((r, m, "test_seen"), (r, m, "test_unseen"), f"{r}/{m}: seen - unseen")
              for r in REGIMES for m in MODELS]
    comps = [c for c in comps if c[0] in boot and c[1] in boot]
    prow = []
    for ka, kb, label in comps:
        for met in ["serious_recall", "serious_false_alarm", "bus_recall", "vm_mae", "rank_V"]:
            d = boot[ka][met] - boot[kb][met]
            d = d[~np.isnan(d)]
            est = point[ka][met] - point[kb][met]
            if len(d) == 0 or np.allclose(d, 0):
                p = 1.0; lo = hi = 0.0
            else:
                p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
                # p = 0 only means "below bootstrap resolution"; report as p < 1/B
                lo, hi = np.percentile(d, [2.5, 97.5])
            prow.append(dict(comparison=label, test_set=ka[2] if ka[2] == kb[2] else "seen vs unseen",
                             metric=met, difference=est, ci_low=lo, ci_high=hi, p_value=p))
    with open(f"{C.RESULTS_DIR}/stats_paired.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(prow[0])); w.writeheader(); w.writerows(prow)

    # ---- print the headline tables -----------------------------------------
    print(f"\n95% CIs (two-way bootstrap over {n_b} test base cases x {n_o} outages, B={B})")
    print(f"{'regime':13s}{'model':7s}{'test set':12s}{'serious recall':>24s}{'bus recall':>22s}"
          f"{'rank_V':>22s}")
    look = {(r["regime"], r["model"], r["test_set"], r["metric"]): r for r in rows}
    for key in keys:
        if key[2] == "test_id":
            continue
        cells_txt = ""
        for met in ["serious_recall", "bus_recall", "rank_V"]:
            r = look[key + (met,)]
            cells_txt += f"{r['estimate']:>8.2f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}]"
        print(f"{key[0]:13s}{key[1]:7s}{key[2]:12s}{cells_txt}")

    print("\nVariance decomposition, serious recall on unseen outages (fold SD vs seed SD)")
    for v in vrows:
        if v["metric"] == "serious_recall" and v["test_set"] == "test_unseen":
            print(f"  {v['regime']:13s}{v['model']:7s} mean {v['mean']:.3f}  fold SD {v['fold_sd']:.3f}  "
                  f"seed SD {v['seed_sd']:.3f}" if not np.isnan(v["seed_sd"])
                  else f"  {v['regime']:13s}{v['model']:7s} mean {v['mean']:.3f}  fold SD {v['fold_sd']:.3f}  "
                       f"seed SD n/a (deterministic)")

    print("\nPaired comparisons (difference, 95% CI, two-sided bootstrap p)")
    for r in prow:
        if r["metric"] in ("serious_recall", "serious_false_alarm", "rank_V"):
            print(f"  {r['comparison']:42s}{r['test_set']:15s}{r['metric']:15s}"
                  f"{r['difference']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]  "
                  + (f"p<{1 / B:.4f}" if r["p_value"] < 1 / B else f"p={r['p_value']:.4f}"))
    print("\nSaved results/stats_ci.csv, results/stats_variance.csv, results/stats_paired.csv")


if __name__ == "__main__":
    main()
