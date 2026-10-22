"""One evaluation function for every model and split.

Everything is computed in physical units. Voltage magnitude (p.u.) and angle
(degrees) are never averaged together. Violations are computed from the
model's OWN predictions and compared with the true AC solution, so we measure
whether the model can tell an operator that limits are broken.

Counts (tp, fn, fp) are stored alongside rates so results can be pooled
correctly across folds and outage lines.
"""

import json

import numpy as np

import config as C

# Per-bus voltage limits, read from the case file via the dataset metadata.
_META = json.load(open(f"{C.OUT_DIR}/dataset_meta.json"))
VMIN = np.array(_META["vmin"])
VMAX = np.array(_META["vmax"])


def violations(vm):
    return (vm < VMIN - C.V_TOL) | (vm > VMAX + C.V_TOL)


def evaluate(pred, Y, nb):
    """pred, Y: (n, 2*nb) arrays, [vm (p.u.) | va (deg)]. Returns a flat dict."""
    vm_p, va_p = pred[:, :nb], pred[:, nb:]
    vm_t, va_t = Y[:, :nb], Y[:, nb:]
    t, p = violations(vm_t), violations(vm_p)
    tp, fn, fp = int((t & p).sum()), int((t & ~p).sum()), int((~t & p).sum())
    rec = tp / (tp + fn) if tp + fn else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    return dict(
        n=len(Y),
        vm_mae_pu=float(np.abs(vm_p - vm_t).mean()),
        va_mae_deg=float(np.abs(va_p - va_t).mean()),
        vm_max_err_pu=float(np.abs(vm_p - vm_t).max()),
        tp=tp, fn=fn, fp=fp,
        recall=rec, precision=prec,
    )


def evaluate_by_topology(pred, Y, topo, nb):
    """Same metrics, broken down by outage line. Keys are topo ids as strings."""
    return {str(int(t)): evaluate(pred[topo == t], Y[topo == t], nb)
            for t in np.unique(topo)}
