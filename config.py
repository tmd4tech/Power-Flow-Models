"""Central configuration (v3: physically valid operating points).

A run is fully described by this file plus requirements.txt."""

# PGLib-OPF v23.07 IEEE 30-bus case (true IEEE 30-bus topology, documented
# thermal ratings, voltage limits 0.94-1.06 p.u. taken from the case file).
import os

# Which grid: set the environment variable GRID (default case30), e.g. on
# Windows PowerShell:  $env:GRID="case118"; python generate.py
GRID = os.environ.get("GRID", "case30")
GRIDS = {
    "case30": dict(case="pglib_opf_case30_ieee", load_level=(0.60, 0.95),
                   n_paired=400, n_intact=1000, impedance=False, chunk=20),
    "case118": dict(case="pglib_opf_case118_ieee", load_level=(0.60, 1.10),
                    n_paired=200, n_intact=1000, impedance=True, chunk=5),
    # case500: PGLib GOC 500-bus grid. Repairs in grid_case.py: slack moved to the
    # largest in-service generator (index 16); impedance branches rated from the
    # case file. OPF converges at 85-110% of nominal load (~3 s per solve).
    "case500": dict(case="pglib_opf_case500_goc", load_level=(0.85, 1.10),
                    n_paired=80, n_intact=1000, impedance=True, chunk=2,
                    slack_gen=16, rate_impedance=True),
}
_G = GRIDS[GRID]
CASE_NAME = _G["case"]
CASE_FILE = f"cases/{CASE_NAME}.m"
SEED = 2026

# --- Base cases: load sample -> AC-OPF dispatch (secure before any outage) ---
LOAD_LEVEL = _G["load_level"]   # x PGLib nominal load. case30: pandapower's
                            # OPF stops converging near 1.0 (README, known issues);
                            # case118 converges up to at least 1.1
LOAD_NOISE = 0.20           # independent per-load factor U[1-x, 1+x] on P
PF_NOISE = 0.15             # extra independent factor on Q, so Q is not
                            # proportional to P (power factor varies)
COST_NOISE = 0.30           # per-sample multiplicative noise on generator cost
                            # coefficients, so dispatch varies beyond load

N_BASE_PAIRED = _G["n_paired"]      # base cases solved under intact + every N-1 outage
N_BASE_INTACT_ONLY = _G["n_intact"] # extra base cases solved intact only (gives the
                                    # intact-only control a fair amount of training data)

# --- Post-contingency power flow -----------------------------------------
# Generator P and voltage setpoints frozen at the OPF dispatch; the slack bus
# picks up the change in losses. Reactive limits ENFORCED (PV -> PQ switching).
MAX_ITERATION = 30

# pandapower's MATPOWER converter turns 3 of PGLib's 41 branches into
# "impedance" elements. v3 data enumerated only lines and transformers (38),
# so ONE valid contingency (impedance between buses 8 and 9) is missing; the
# other two impedances island a synchronous condenser. Keep False to reproduce
# the v3 dataset exactly; set True when regenerating (Phase 4).
INCLUDE_IMPEDANCE_BRANCHES = _G["impedance"]   # per grid, see GRIDS

# case30 keeps the original folders; other grids get their own
OUT_DIR = "data" if GRID == "case30" else f"data_{GRID}"
RESULTS_DIR = "results" if GRID == "case30" else f"results_{GRID}"

# --- Limit checks ----------------------------------------------------------
# The OPF drives binding constraints exactly to their limits, so floating-point
# noise can put a binding line at 100.0000001%. These tolerances keep such
# cases from being counted as violations. They are numerical, not operational
# margins; Phase 2 adds recall at operational margins.
V_TOL = 1e-6        # p.u.
THERMAL_LIMIT = 100.0
THERMAL_TOL = 1e-3  # percentage points
