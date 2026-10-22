"""Generate the structural-shift dataset (v3: physically valid operating points).

Changes from v2, each answering a specific review point:
  * Test system: PGLib-OPF IEEE 30-bus (true IEEE topology, documented thermal
    ratings, voltage limits from the case file). v2 used MATPOWER's modified
    case30 and called it "IEEE 30-bus".
  * Every base case is dispatched by AC-OPF, so it respects all generator,
    voltage, and thermal limits BEFORE any outage. (In v2, 35% of intact
    operating points already violated voltage limits.)
  * Post-contingency power flow ENFORCES generator reactive limits. (In v2 they
    were ignored; a generator exceeded its limit in 45% of line-28 cases.)
  * Power factor varies per load; generator costs are perturbed per sample so
    dispatch varies beyond the load pattern.
  * Paired design: each base case is solved under the intact grid and every
    non-islanding N-1 branch outage (lines AND transformers), so topology is
    the only difference between paired samples.
  * Non-converged post-contingency solves are kept (converged=False). Base
    cases where the OPF fails are redrawn, and every failure is counted in the
    metadata, because an OPF failure means the sampled load is not operable.

Run:
    python generate.py --pilot   # ~2 min sanity run
    python generate.py           # full dataset -> data/dataset.npz (resumable)
"""

import argparse
import copy
import json
import os
import time
import warnings

import networkx as nx
import numpy as np
import pandapower as pp
import pandapower.topology as top
import config as C
from grid_case import load_net

warnings.filterwarnings("ignore")
CHUNK = C._G["chunk"]   # base cases per part file (progress granularity only; data unchanged)


# ------------------------------------------------------------ network ----
def load_case():
    return load_net()          # grid_case.py: identical network for every script


def branch_list(net):
    """All branches as (element, index): lines, transformers, then (if enabled)
    impedance elements, matching pandapower's internal branch order."""
    br = [("line", int(i)) for i in net.line.index] + \
         [("trafo", int(i)) for i in net.trafo.index]
    if C.INCLUDE_IMPEDANCE_BRANCHES:
        br += [("impedance", int(i)) for i in net.impedance.index]
    return br


def outage_topologies(net):
    """Non-islanding single-branch outages, as indices into branch_list."""
    br = branch_list(net)
    keep, islanding = [], []
    for b, (el, i) in enumerate(br):
        n = copy.deepcopy(net)
        n[el].at[i, "in_service"] = False
        if nx.number_connected_components(top.create_nxgraph(n)) > 1:
            islanding.append(b)
        else:
            keep.append(b)
    return keep, islanding


# --------------------------------------------------------- base cases ----
def draw_base_case(base, rng):
    """Sample loads and costs, then AC-OPF. Returns (net, n_failed_attempts)."""
    failures = 0
    while True:
        net = copy.deepcopy(base)
        level = rng.uniform(*C.LOAD_LEVEL)
        fp = level * rng.uniform(1 - C.LOAD_NOISE, 1 + C.LOAD_NOISE, len(net.load))
        fq = fp * rng.uniform(1 - C.PF_NOISE, 1 + C.PF_NOISE, len(net.load))
        net.load["p_mw"] = net.load["p_mw"].values * fp
        net.load["q_mvar"] = net.load["q_mvar"].values * fq
        for col in ["cp1_eur_per_mw", "cp2_eur_per_mw2"]:
            net.poly_cost[col] = net.poly_cost[col].values * rng.uniform(
                1 - C.COST_NOISE, 1 + C.COST_NOISE, len(net.poly_cost))
        try:
            pp.runopp(net, numba=False)
            if net.OPF_converged:
                return net, failures, level
        except Exception:
            pass
        failures += 1


def freeze_dispatch(opf_net):
    """Turn the OPF solution into power flow setpoints."""
    net = copy.deepcopy(opf_net)
    vm = opf_net.res_bus["vm_pu"].values
    bus_pos = {b: k for k, b in enumerate(opf_net.bus.index)}
    net.gen["p_mw"] = opf_net.res_gen["p_mw"].values
    net.gen["vm_pu"] = [vm[bus_pos[b]] for b in net.gen["bus"]]
    net.ext_grid["vm_pu"] = [vm[bus_pos[b]] for b in net.ext_grid["bus"]]
    if len(net.sgen):          # dispatchable static generators: freeze P and Q
        net.sgen["p_mw"] = opf_net.res_sgen["p_mw"].values
        net.sgen["q_mvar"] = opf_net.res_sgen["q_mvar"].values
    return net


# -------------------------------------------------------- power flow ----
def solve_topology(pf_net, branches, outage):
    """Solve one topology. outage = branch index, or None for intact."""
    net = copy.deepcopy(pf_net)
    if outage is not None:
        el, i = branches[outage]
        net[el].at[i, "in_service"] = False
    nb = len(net.bus)
    nbr = len(net.line) + len(net.trafo)   # rated branches
    if C._G.get("rate_impedance"):
        nbr += len(net.impedance)
    try:
        pp.runpp(net, numba=False, enforce_q_lims=True, max_iteration=C.MAX_ITERATION)
        ok = bool(net.converged)
    except Exception:
        ok = False
    if not ok:
        return dict(vm=np.full(nb, np.nan), va=np.full(nb, np.nan),
                    loading=np.full(nbr, np.nan), converged=False,
                    n_q_limited=-1, slack_p=np.nan, slack_q=np.nan)
    parts = [net.res_line["loading_percent"].values, net.res_trafo["loading_percent"].values]
    if C._G.get("rate_impedance"):
        # impedance elements: MVA loading against MATPOWER RATE_A (grid_case.py)
        ri = net.res_impedance
        s_f = np.hypot(ri["p_from_mw"].values, ri["q_from_mvar"].values)
        s_t = np.hypot(ri["p_to_mw"].values, ri["q_to_mvar"].values)
        parts.append(np.maximum(s_f, s_t) / net.impedance["rate_mva"].values * 100.0)
    loading = np.concatenate(parts)
    q = net.res_gen["q_mvar"].values
    at_lim = (q >= net.gen["max_q_mvar"].values - 1e-3) | (q <= net.gen["min_q_mvar"].values + 1e-3)
    return dict(vm=net.res_bus["vm_pu"].values.copy(), va=net.res_bus["va_degree"].values.copy(),
                loading=np.nan_to_num(loading), converged=True,
                n_q_limited=int(at_lim[net.gen["in_service"].values].sum()),
                slack_p=float(net.res_ext_grid["p_mw"].sum()),
                slack_q=float(net.res_ext_grid["q_mvar"].sum()))


def injections(net):
    """Inputs per bus: P/Q load, generator P, and voltage setpoints (0 = none)."""
    nb = len(net.bus)
    pos = {b: k for k, b in enumerate(net.bus.index)}
    p_load, q_load, p_gen, v_set = (np.zeros(nb) for _ in range(4))
    for b, p, q in zip(net.load["bus"], net.load["p_mw"], net.load["q_mvar"]):
        p_load[pos[b]] += p; q_load[pos[b]] += q
    for b, p, v in zip(net.gen["bus"], net.gen["p_mw"], net.gen["vm_pu"]):
        p_gen[pos[b]] += p; v_set[pos[b]] = v
    for b, v in zip(net.ext_grid["bus"], net.ext_grid["vm_pu"]):
        v_set[pos[b]] = v
    # static generators (case500): their P counts as generation at the bus and
    # their Q as negative demand, so p_gen - p_load and -q_load stay the net
    # specified injections everywhere downstream
    for b, p, q in zip(net.sgen["bus"], net.sgen["p_mw"], net.sgen["q_mvar"]):
        p_gen[pos[b]] += p; q_load[pos[b]] -= q
    return p_load, q_load, p_gen, v_set


# --------------------------------------------------------------- main ----
def generate_chunk(base, branches, topologies, base_ids, seeds, paired):
    rows, opf_failures, consistency = [], 0, []
    for bid in base_ids:
        rng = np.random.default_rng(seeds[bid])
        opf_net, fails, level = draw_base_case(base, rng)
        opf_failures += fails
        pf_net = freeze_dispatch(opf_net)
        p_load, q_load, p_gen, v_set = injections(pf_net)
        for t in (topologies if paired else [None]):
            r = solve_topology(pf_net, branches, t)
            if t is None and r["converged"]:   # PF must reproduce the OPF voltages
                consistency.append(float(np.abs(r["vm"] - opf_net.res_bus["vm_pu"].values).max()))
            status = np.ones(len(branches), dtype=np.int8)
            if t is not None:
                status[t] = 0
            row = dict(r, base_id=bid, topo_id=-1 if t is None else t, level=level,
                       p_load=p_load, q_load=q_load, p_gen=p_gen, v_set=v_set,
                       branch_status=status)
            if len(pf_net.sgen):   # per-unit dispatch, so timing.py can rebuild the case
                row.update(gen_p_el=pf_net.gen["p_mw"].values.copy(),
                           sgen_p_el=pf_net.sgen["p_mw"].values.copy(),
                           sgen_q_el=pf_net.sgen["q_mvar"].values.copy())
            rows.append(row)
    out = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    out["opf_failures"] = np.array([opf_failures])
    out["pf_opf_max_diff"] = np.array([max(consistency) if consistency else np.nan])
    return out


def summarize(d, vmin, vmax, branches):
    conv = d["converged"]
    vm = d["vm"]
    vv = ((vm < vmin - C.V_TOL) | (vm > vmax + C.V_TOL)) & conv[:, None]
    th = (d["loading"] > C.THERMAL_LIMIT + C.THERMAL_TOL) & conv[:, None]
    print(f"{'topology':>12} {'n':>5} {'conv%':>6} {'V-viol bus%':>11} {'cases w/ V-viol%':>16} "
          f"{'cases w/ overload%':>18} {'Q-limited gens':>14}")
    for t in np.unique(d["topo_id"]):
        m = d["topo_id"] == t
        c = conv[m]
        name = "intact" if t == -1 else f"{branches[t][0]} {branches[t][1]}"
        ql = d["n_q_limited"][m][c].mean() if c.any() else float("nan")
        print(f"{name:>12} {m.sum():>5} {c.mean()*100:>6.1f} "
              f"{vv[m][c].mean()*100 if c.any() else float('nan'):>11.2f} "
              f"{vv[m][c].any(1).mean()*100 if c.any() else float('nan'):>16.1f} "
              f"{th[m][c].any(1).mean()*100 if c.any() else float('nan'):>18.1f} {ql:>14.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--budget", type=float, default=None, help="stop after ~N seconds; rerun resumes")
    args = ap.parse_args()

    base = load_case()
    branches = branch_list(base)
    topologies, islanding = outage_topologies(base)
    topologies = [None] + topologies
    vmin, vmax = base.bus["min_vm_pu"].values, base.bus["max_vm_pu"].values

    n_paired = 20 if args.pilot else C.N_BASE_PAIRED
    n_intact = 0 if args.pilot else C.N_BASE_INTACT_ONLY
    seeds = np.random.SeedSequence(C.SEED).spawn(n_paired + n_intact)
    out_dir = os.path.join(C.OUT_DIR, "pilot" if args.pilot else "parts")
    os.makedirs(out_dir, exist_ok=True)

    print(f"{C.CASE_NAME}: {len(base.bus)} buses, {len(branches)} branches "
          f"({len(base.line)} lines, {len(base.trafo)} transformers); "
          f"{len(topologies)-1} non-islanding outages; islanding branches skipped: "
          f"{[f'{branches[b][0]} {branches[b][1]}' for b in islanding]}")

    jobs = [(list(range(s, min(s + CHUNK, n_paired))), True) for s in range(0, n_paired, CHUNK)]
    jobs += [(list(range(s, min(s + CHUNK, n_paired + n_intact))), False)
             for s in range(n_paired, n_paired + n_intact, CHUNK)]
    t0, files = time.time(), []
    for ids, paired in jobs:
        f = os.path.join(out_dir, f"{'paired' if paired else 'intact'}_{ids[0]:05d}.npz")
        files.append(f)
        if os.path.exists(f):
            continue
        if args.budget and time.time() - t0 > args.budget:
            print(f"Budget reached ({sum(os.path.exists(x) for x in files)}/{len(jobs)} chunks); rerun to continue.")
            return
        np.savez_compressed(f, **generate_chunk(base, branches, topologies, ids, seeds, paired))
        print(f"  {'paired' if paired else 'intact'} bases {ids[0]}-{ids[-1]} done "
              f"({time.time()-t0:.0f}s)", flush=True)

    parts = [dict(np.load(f)) for f in files]
    per_sample = [k for k in parts[0] if k not in ("opf_failures", "pf_opf_max_diff")]
    d = {k: np.concatenate([p[k] for p in parts]) for k in per_sample}
    name = "pilot" if args.pilot else "dataset"
    np.savez_compressed(os.path.join(C.OUT_DIR, f"{name}.npz"), **d)
    n_bases = n_paired + n_intact
    fails = int(sum(p["opf_failures"][0] for p in parts))
    meta = dict(case=C.CASE_NAME, seed=C.SEED, n_bus=len(base.bus),
                branches=[f"{e} {i}" for e, i in branches],
                outage_topologies=[t for t in topologies if t is not None],
                islanding_branches=islanding,
                vmin=vmin.tolist(), vmax=vmax.tolist(),
                load_level=C.LOAD_LEVEL, load_noise=C.LOAD_NOISE, pf_noise=C.PF_NOISE,
                cost_noise=C.COST_NOISE, n_base_paired=n_paired, n_base_intact_only=n_intact,
                opf_failures=fails, opf_failure_rate=fails / (fails + n_bases),
                pf_vs_opf_max_vm_diff=float(np.nanmax([p["pf_opf_max_diff"][0] for p in parts])),
                n_samples=int(len(d["topo_id"])), pandapower=pp.__version__)
    json.dump(meta, open(os.path.join(C.OUT_DIR, f"{name}_meta.json"), "w"), indent=2)
    print(f"\nSaved {C.OUT_DIR}/{name}.npz: {meta['n_samples']} samples from {n_bases} base cases")
    print(f"OPF failures (redrawn): {fails} ({meta['opf_failure_rate']*100:.1f}% of draws)")
    print(f"Intact PF vs OPF max |V| difference: {meta['pf_vs_opf_max_vm_diff']:.2e} p.u. "
          "(should be ~0: confirms the dispatch was frozen correctly)\n")
    summarize(d, vmin, vmax, branches)


if __name__ == "__main__":
    main()
