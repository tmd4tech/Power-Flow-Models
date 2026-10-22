"""Physics from voltages: branch loading and AC power mismatch.

Given voltages (true or predicted) and a topology, compute:
  * branch loading in % of rating, defined exactly as pandapower defines it
    (lines: max end current / max_i_ka; transformers: max of HV and LV
    current over rated current), so predicted and true loading are comparable;
  * the AC power-flow mismatch: specified injections minus V * conj(Ybus V).
    True voltages give ~0. Predicted voltages that don't obey the physics give
    a nonzero residual, which measures physical consistency directly.

Each topology's Ybus is built from the intact network by setting one branch's
status to 0 (Ybus construction multiplies every branch by its status), so no
power flow has to be re-solved. The operator knows the topology, so flows are
always computed on the TRUE topology, even for topology-blind models.

Mismatch is checked where the injection is actually specified:
  * active power at every bus except the slack;
  * reactive power at buses with no generator and no slack. Generator buses are
    excluded because a generator's Q is free (and after reaching a limit it is
    pinned at that limit, which the inputs don't record).

Run:  python physics.py   -> validates against the stored true solutions
"""

import json
import warnings

import numpy as np
import pandapower as pp
from pandapower.pypower.idx_brch import BR_STATUS, F_BUS, T_BUS
from pandapower.pypower.makeYbus import makeYbus

import config as C
from grid_case import load_net

warnings.filterwarnings("ignore")


class GridPhysics:
    def __init__(self):
        net = load_net()
        pp.runpp(net, numba=False)            # only to build the internal case
        ppc, lk = net._ppc, net._pd2ppc_lookups
        assert np.array_equal(lk["bus"][net.bus.index.values], np.arange(len(net.bus))), \
            "bus ordering assumption violated"
        self.base_mva = float(ppc["baseMVA"])
        self.bus = ppc["bus"].copy()
        self.branch = ppc["branch"].copy()
        self.nl, self.nt = len(net.line), len(net.trafo)
        assert lk["branch"]["line"] == (0, self.nl)
        assert lk["branch"]["trafo"] == (self.nl, self.nl + self.nt)
        self.ni = len(net.impedance) if C._G.get("rate_impedance") else 0
        if self.ni:
            assert lk["branch"]["impedance"] == (self.nl + self.nt, self.nl + self.nt + self.ni)
            self.imp_rate = net.impedance["rate_mva"].values
        self.n_rated = self.nl + self.nt + self.ni   # branches with ratings, in dataset order
        self.f = self.branch[:, F_BUS].real.astype(int)
        self.t = self.branch[:, T_BUS].real.astype(int)
        vn = net.bus["vn_kv"].values
        self.line_base_f = self.base_mva / (np.sqrt(3) * vn[self.f[:self.nl]])
        self.line_base_t = self.base_mva / (np.sqrt(3) * vn[self.t[:self.nl]])
        self.line_imax = net.line["max_i_ka"].values
        tr = net.trafo
        self.tr_base_hv = self.base_mva / (np.sqrt(3) * vn[self.f[self.nl:self.nl + self.nt]])
        self.tr_base_lv = self.base_mva / (np.sqrt(3) * vn[self.t[self.nl:self.nl + self.nt]])
        self.tr_irated_hv = tr["sn_mva"].values / (np.sqrt(3) * tr["vn_hv_kv"].values)
        self.tr_irated_lv = tr["sn_mva"].values / (np.sqrt(3) * tr["vn_lv_kv"].values)
        slack = set(net.ext_grid["bus"])
        gens = set(net.gen["bus"]) | slack
        nb = len(net.bus)
        self.p_check = np.array([b not in slack for b in range(nb)])
        self.q_check = np.array([b not in gens for b in range(nb)])
        self._cache = {}

    def matrices(self, topo_id):
        if topo_id not in self._cache:
            br = self.branch.copy()
            if topo_id >= 0:
                br[topo_id, BR_STATUS] = 0
            self._cache[topo_id] = makeYbus(self.base_mva, self.bus, br)
        return self._cache[topo_id]

    @staticmethod
    def complex_v(vm, va_deg):
        return vm * np.exp(1j * np.deg2rad(va_deg))

    def loading(self, vm, va_deg, topo_id):
        """vm, va: (n, nb). Returns (n, n_rated) loading in % for one topology."""
        _, Yf, Yt = self.matrices(topo_id)
        V = self.complex_v(vm, va_deg)
        Ifc, Itc = (Yf @ V.T).T, (Yt @ V.T).T                  # complex branch currents (p.u.)
        If, It = np.abs(Ifc), np.abs(Itc)
        nl, nt = self.nl, self.nt
        line = np.maximum(If[:, :nl] * self.line_base_f, It[:, :nl] * self.line_base_t) / self.line_imax
        trafo = np.maximum(If[:, nl:nl + nt] * self.tr_base_hv / self.tr_irated_hv,
                           It[:, nl:nl + nt] * self.tr_base_lv / self.tr_irated_lv)
        parts = [line, trafo]
        if self.ni:      # impedance elements: MVA against RATE_A, as in generate.py
            k = slice(nl + nt, nl + nt + self.ni)
            s_f = np.abs(V[:, self.f[k]] * np.conj(Ifc[:, k])) * self.base_mva
            s_t = np.abs(V[:, self.t[k]] * np.conj(Itc[:, k])) * self.base_mva
            parts.append(np.maximum(s_f, s_t) / self.imp_rate)
        out = np.concatenate(parts, axis=1) * 100.0
        if 0 <= topo_id < self.n_rated:
            out[:, topo_id] = 0.0                 # outaged branch carries nothing
        return out

    def mismatch(self, vm, va_deg, p_inj, q_inj, topo_id):
        """Returns (dP [MW], dQ [Mvar]) on the checked buses, shape (n, nb) with NaN elsewhere."""
        Y, _, _ = self.matrices(topo_id)
        V = self.complex_v(vm, va_deg)
        S = V * np.conj(V @ Y.T.toarray()) * self.base_mva
        dP = np.where(self.p_check, p_inj - S.real, np.nan)
        dQ = np.where(self.q_check, q_inj - S.imag, np.nan)
        return dP, dQ


def injections(D, idx):
    """Specified net injections (MW, Mvar) per bus."""
    return D["p_gen"][idx] - D["p_load"][idx], -D["q_load"][idx]


def validate():
    """Recompute loading and mismatch from the TRUE voltages: must match the stored
    loading and give ~zero mismatch for every topology."""
    D = np.load(f"{C.OUT_DIR}/dataset.npz")
    gp = GridPhysics()
    worst_load, worst_p, worst_q = 0.0, 0.0, 0.0
    for t in np.unique(D["topo_id"]):
        idx = np.where((D["topo_id"] == t) & D["converged"])[0][:50]
        vm, va = D["vm"][idx], D["va"][idx]
        L = gp.loading(vm, va, int(t))
        worst_load = max(worst_load, float(np.abs(L - D["loading"][idx]).max()))
        dP, dQ = gp.mismatch(vm, va, *injections(D, idx), int(t))
        worst_p = max(worst_p, float(np.nanmax(np.abs(dP))))
        worst_q = max(worst_q, float(np.nanmax(np.abs(dQ))))
    print(f"Validation on true solutions, all {len(np.unique(D['topo_id']))} topologies:")
    print(f"  max |recomputed - stored loading| = {worst_load:.2e} percentage points")
    print(f"  max |P mismatch| = {worst_p:.2e} MW,  max |Q mismatch| = {worst_q:.2e} Mvar")
    ok = worst_load < 1e-6 and worst_p < 1e-3 and worst_q < 1e-3
    print("  PASS" if ok else "  FAIL: do not trust flow-based metrics until this passes")
    return ok


if __name__ == "__main__":
    validate()
