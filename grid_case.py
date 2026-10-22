"""Load the selected grid, with any case-specific repairs, identically everywhere.

Every script that builds the network (generate.py, physics.py, timing.py) calls
load_net(), so data generation, physics checks and timing all see the same grid.
For case30 and case118 this is exactly pandapower's MATPOWER conversion.

case500 (pglib_opf_case500_goc) needs two repairs, both documented in config:
  * SLACK: the case's reference generator converts as out of service, so
    pandapower has no slack bus. Rather than switching on a unit the case
    designers left off, the reference role moves to the largest IN-SERVICE
    generator (index GRIDS[...]['slack_gen']): the ext_grid element is moved to
    its bus and takes its voltage setpoint, P/Q limits and cost, and the
    generator itself is removed (it now IS the slack).
  * IMPEDANCE RATINGS: pandapower converts generator step-up transformers
    without taps into 'impedance' elements and drops their thermal rating.
    The MATPOWER RATE_A of each such branch is read from the case file and
    stored in net.impedance['rate_mva'], so overload analysis covers them.
"""

import numpy as np
from pandapower.converter.matpower import from_mpc

import config as C


def load_net():
    net = from_mpc(C.CASE_FILE, f_hz=60)
    g = C._G.get("slack_gen")
    if g is not None:
        _make_gen_slack(net, g)
    if C._G.get("rate_impedance"):
        net.impedance["rate_mva"] = impedance_ratings(net)
    return net


def _make_gen_slack(net, g):
    assert len(net.ext_grid) == 1, "expected one ext_grid"
    assert bool(net.gen.at[g, "in_service"]), "slack generator must be in service"
    eg = net.ext_grid.index[0]
    r = net.gen.loc[g]
    for col in ["bus", "vm_pu", "min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar"]:
        net.ext_grid.at[eg, col] = r[col]
    net.ext_grid.at[eg, "in_service"] = True
    pc = net.poly_cost
    pc.drop(pc.index[pc.et == "ext_grid"], inplace=True)
    i = pc.index[(pc.et == "gen") & (pc.element == g)]
    assert len(i) == 1, "slack generator must have exactly one cost row"
    pc.loc[i, "et"] = "ext_grid"
    pc.loc[i, "element"] = eg
    net.gen.drop(g, inplace=True)


def impedance_ratings(net):
    """MATPOWER RATE_A (MVA) of every impedance element, in net.impedance order."""
    from matpowercaseframes import CaseFrames
    br = CaseFrames(C.CASE_FILE).branch
    lk = net._from_ppc_lookups["branch"]
    imp = lk[lk.element_type == "impedance"]
    rate = dict(zip(imp.element.astype(int).values, br.iloc[imp.index.values]["RATE_A"].values))
    out = np.array([rate[int(e)] for e in net.impedance.index], dtype=float)
    assert (out > 0).all(), "an impedance branch has no rating"
    return out
