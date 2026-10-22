"""Define train/val/test splits as index arrays into data/dataset.npz.

Two levels of separation:
  1. BASE CASES are split 70/10/20 into train/val/test. A test base case (its
     load pattern and dispatch) is never seen in training under ANY topology,
     so the paired design cannot leak load information into the test set.
  2. OUTAGES are split into K folds. In fold k, the outages in that fold are
     never seen in training, for any base case.

Regimes:
  intact_only   train/val: intact samples of train/val base cases
                test_id:   intact samples of test base cases
                test_ood:  every outage sample of test base cases
  n1_fold_k     train/val: intact + outages NOT in fold k, train/val bases
                test_id:   intact, test bases
                test_seen: outages not in fold k, test bases
                test_unseen: outages in fold k, test bases

Non-converged samples stay in the lists; they are masked when training and
scored separately at evaluation time.

Folds are stratified by outage severity (see below), so no fold is made of
only the hardest or only the mildest outages.

Run:  python splits.py   -> data/splits.npz
"""

import json

import numpy as np

import config as C

K = 5
FRACS = (0.7, 0.1)   # train, val; remainder is test


def split_bases(ids, rng):
    ids = rng.permutation(ids)
    a, b = int(len(ids) * FRACS[0]), int(len(ids) * sum(FRACS))
    return ids[:a], ids[a:b], ids[b:]


def main():
    d = np.load(f"{C.OUT_DIR}/dataset.npz")
    meta = json.load(open(f"{C.OUT_DIR}/dataset_meta.json"))
    base, topo = d["base_id"], d["topo_id"]
    rng = np.random.default_rng(C.SEED + 1)

    paired_ids = np.arange(meta["n_base_paired"])
    intact_ids = np.arange(meta["n_base_paired"], meta["n_base_paired"] + meta["n_base_intact_only"])
    # split paired and intact-only bases separately so every split has both kinds
    sp, si = split_bases(paired_ids, rng), split_bases(intact_ids, rng)
    bases = {name: np.concatenate([sp[k], si[k]]) for k, name in enumerate(["train", "val", "test"])}
    in_split = {name: np.isin(base, ids) for name, ids in bases.items()}
    intact = topo == -1

    out = {}
    out["intact_only/train"] = np.where(in_split["train"] & intact)[0]
    out["intact_only/val"] = np.where(in_split["val"] & intact)[0]
    out["intact_only/test_id"] = np.where(in_split["test"] & intact)[0]
    out["intact_only/test_ood"] = np.where(in_split["test"] & ~intact)[0]

    # Stratify folds by severity so each fold gets a similar mix of mild and
    # severe outages. Severity = share of converged samples with any voltage
    # violation (non-convergence counts as severe). Outages are ranked, cut into
    # groups of K, and each group is dealt out across the folds in random order.
    outages = np.array(meta["outage_topologies"])
    vm, conv = d["vm"], d["converged"]
    vmin, vmax = np.array(meta["vmin"]), np.array(meta["vmax"])
    bad = ~conv | ((vm < vmin - C.V_TOL) | (vm > vmax + C.V_TOL)).any(1)
    severity = np.array([bad[topo == t].mean() for t in outages])
    ranked = outages[np.argsort(-severity)]
    folds = [[] for _ in range(K)]
    for g in range(0, len(ranked), K):
        for f, t in zip(rng.permutation(K), ranked[g:g + K]):
            folds[f].append(t)
    folds = [np.array(f) for f in folds]
    for k, held in enumerate(folds):
        is_held = np.isin(topo, held)
        p = f"n1_fold_{k}/"
        out[p + "train"] = np.where(in_split["train"] & ~is_held)[0]
        out[p + "val"] = np.where(in_split["val"] & ~is_held)[0]
        out[p + "test_id"] = np.where(in_split["test"] & intact)[0]
        out[p + "test_seen"] = np.where(in_split["test"] & ~intact & ~is_held)[0]
        out[p + "test_unseen"] = np.where(in_split["test"] & is_held)[0]
        out[p + "held_out"] = np.sort(held)

    for name, ids in bases.items():
        out[f"bases/{name}"] = np.sort(ids)
    np.savez(f"{C.OUT_DIR}/splits.npz", **out)

    names = meta["branches"]
    print(f"Saved {C.OUT_DIR}/splits.npz. Base cases: "
          + ", ".join(f"{k} {len(v)}" for k, v in bases.items()))
    for k in out:
        if k.endswith("held_out"):
            print(f"  {k:24s} {[names[i] for i in out[k]]}")
        elif not k.startswith("bases/"):
            n_nc = int((~d['converged'][out[k]]).sum())
            print(f"  {k:24s} {len(out[k]):6d} samples ({n_nc} non-converged)")


if __name__ == "__main__":
    main()
