# Learned power flow under N-1 outages (v3: physically valid data)

```bash
pip install -r requirements.txt
python generate.py --pilot     # ~1.5 min sanity run
python generate.py             # ~35 min on one core -> data/dataset.npz (resumable)
python splits.py               # data/splits.npz
python run_experiments.py      # ~10 min; resumable
python report.py               # results/summary.csv, results/by_line.csv
python depth_analysis.py       # Phase 2: recall by violation depth, false alarms
python physics.py              # validates flow/mismatch code against true solutions
python flow_analysis.py        # Phase 2: thermal overloads, power-flow mismatch
python case_analysis.py        # Phase 2: case-level security labels, contingency ranking
python baselines.py            # Phase 3: physics baselines (intact AC, DC, DC increment, 1P1Q)
python gnn.py                  # Phase 3: graph neural network (PyTorch, CPU; slow, resumable)
python stats.py                # Phase 2: fold vs seed variance, bootstrap CIs, paired tests
python timing.py               # Phase 3: per-contingency speed of every method
```

`timing.py` loads the dataset into memory first: `np.load` on a compressed
`.npz` is lazy, and every `D["key"]` access re-reads and decompresses the whole
array, which made per-case loops look ~500x slower than they are. (The other
scripts still access the file lazily; this costs speed, not correctness.)

GNN v3 (final) learns a correction on top of the intact-grid ridge model
(linear path), up to 400 epochs, with one residual scale per quantity. v1 (no
linear path) and v2 (per-bus residual scaling, which let training degrade
case118 accuracy) are superseded; never mix versions.
`gnn_colab.ipynb` runs both grids on a Colab GPU.

`gnn.py` checkpoints every epoch; `python gnn.py --seeds 1 2` runs only those
seeds, and several copies with different `--seeds` can run in parallel.

GNN results in this package cover seed 0 only (5 folds), run on one CPU core.
`python gnn.py` adds seeds 1 and 2 (existing seed-0 runs are skipped). Depth
was tuned per fold by outage-grouped validation; depth 8 won in every fold.
`fit_seconds` is not a timing measurement for runs that were interrupted.

Run `baselines.py` before the Phase 2 scripts to include the physics baselines
in every table.

### Hybrid models (Phase 3)

`run_experiments.py` includes two hybrid regimes. Both receive the inputs, the
pre-outage AC state, and the 1P1Q prediction for the outaged grid (cached in
`data/physics_features.npz`), and NO branch-status indicators:

- `hybrid_n1`: 1P1Q + learned correction to [vm | va]
- `hybridv_n1`: 1P1Q angles kept; learned correction to vm only. Added after
  `hybrid_n1` MLPs degraded thermal results; report both.

Model-fitting details that matter (all regimes):
- inputs constant in training (condenser P ~ 1e-9 MW noise) are dropped;
- TARGETS constant in training (SD < 1e-6: the slack angle; hybrid residuals at
  generator buses that never reach a reactive limit, SD ~1e-16) are left
  unscaled. Before this fix, standardizing them produced validation MSEs of
  ~1e23 for case118 hybrids. case30 blind/status models had one such column
  (slack angle) that never exploded; their shipped runs predate the fix and
  should be rerun in the final sweep for consistency.
- learned inputs are clipped to their training range;
- linear models are ridge; regimes trained on outages choose the penalty by
  OUTAGE-GROUPED cross-validation (held-out outages, not held-out samples).
  Sample-level CV picked weak penalties and gave 0.5 p.u. errors on unseen
  outages for the hybrid linear model.
- MLP configurations are chosen PER FOLD from that fold's training data only,
  by outage-grouped validation (a third of the training outages held out);
  `results/tuning_<regime>_fold<k>.json` records each choice and the held-out
  outages. `blind_intact` has no outages to hold out and keeps sample-level
  validation. Earlier fold-0 sample-level tuning files are kept in
  `results/old_tuning/` for reference. Physics baselines use the pre-outage AC state (one AC solution
per base case); the learned models do not. See the docstring in `baselines.py`.

`run_experiments.py` saves each run's test-set predictions to `results/preds/`,
so Phase 2 analyses never retrain.

All settings live in `config.py`.

## Choosing the grid

The grid is chosen with the environment variable `GRID` (default `case30`).
case30 uses `data/` and `results/`; case118 uses `data_case118/` and
`results_case118/`. In Windows PowerShell:

```
$env:GRID="case118"     # every script in this window now uses case118
$env:GRID="case30"      # switch back (or open a new window)
```

Per-grid settings (load range, number of base cases, impedance branches) are in
`GRIDS` in `config.py`.

### case500 (PGLib `pglib_opf_case500_goc`)

`grid_case.py` loads every grid; for case500 it applies two documented repairs:
1. **Slack:** the case's reference generator converts as out of service, leaving
   no slack bus. The reference role moves to the largest in-service generator
   (index 16, bus 311, 1,165 MW), which takes over the ext_grid element with its
   own setpoint, limits and cost. No out-of-service unit is switched on.
2. **Impedance ratings:** pandapower converts 89 generator step-up transformers
   to `impedance` elements without ratings; their MATPOWER RATE_A is read from
   the case file, and their loading is MVA against that rating.

case500 also has 110 dispatchable static generators: their OPF dispatch is
frozen for outage solves, their P counts as generation and their Q as negative
demand in the model inputs, and per-unit dispatch (`gen_p_el`, `sgen_p_el`,
`sgen_q_el`) is stored so `timing.py` can rebuild each case. 146 of 733
branches island the grid; 587 outages are studied. 80 paired base cases.

## Data (v3)

- **Test system:** PGLib-OPF v23.07 `pglib_opf_case30_ieee` (true IEEE 30-bus
  topology, documented thermal ratings, voltage limits 0.94-1.06 p.u. from
  the case file). File in `cases/`, CC BY 4.0.
- **Base cases:** loads sampled at 60-95% of nominal with per-load P noise
  (+/-20%) and independent power-factor noise (+/-15%); generator costs
  perturbed +/-30%; dispatched by AC-OPF, so every base case respects all
  limits before any outage. OPF failures are redrawn and counted in
  `data/dataset_meta.json` (1.5% of draws).
- **Contingencies:** each of 400 base cases is solved intact and under 37
  single-branch outages (33 lines + 4 transformers; line 27 islands the grid).
  This is 37 of the 38 non-islanding N-1 contingencies: see known issues. Generator P and voltage setpoints frozen at the OPF
  dispatch, slack picks up losses, **generator reactive limits enforced**.
  1,000 further base cases are solved intact only.
- **Non-convergence kept:** 80 post-contingency solves (line 0 outage) do not
  converge; they are stored with `converged=False`.
- **Splits:** base cases split 70/10/20, so test load patterns are never seen
  in training under any topology; outages split into 5 folds stratified by
  severity.
- **Tolerances:** binding OPF limits sit at e.g. 100.0000001% loading, so
  violations use numerical tolerances (`V_TOL`, `THERMAL_TOL` in config).

## MLPs and GPU

MLPs are PyTorch (`mlp_torch.py`), replacing sklearn's MLPRegressor, whose
early stopping halted after a few epochs on case118 (236 outputs): intact-only
MLPs had 6 degree angle errors on the INTACT grid. The tuning grid, L2 penalty,
and 10% stopping split are unchanged; training continues while validation MSE
improves, with patience 30 and the best weights restored.

Each MLP has a fixed LINEAR PATH: output = ridge(x) + mlp(x), where ridge is the
regime's own linear model for that fold (penalty from its linear run, chosen by
outage-grouped CV). The MLP's last layer starts at zero, so training starts at
the linear model and cannot end worse than it on validation. Linear runs must
therefore exist before MLP runs; `run_experiments.py` runs each regime's
linear models first. On case118, plain MLPs were 6-10x worse than linear
in-distribution; with the linear path that baseline weakness is removed.

`run_experiments.py` and `gnn.py` use an NVIDIA GPU automatically when PyTorch
sees one, and print which device they use. Set `$env:GRID_DEVICE="cpu"` to force
the CPU. `timing.py` always times on the CPU. GPU and CPU results can differ in
the last decimals, so run all learned models of one comparison on one device.

## Known issues

- pandapower's MATPOWER converter turns 3 of PGLib's 41 branches into
  "impedance" elements, which v3 did not enumerate. One of them (buses 8-9) is a
  valid contingency missing from the dataset; the other two island a
  synchronous condenser. Fixed behind `INCLUDE_IMPEDANCE_BRANCHES` in
  `config.py` (off, so v3 regenerates exactly; turn on for the next dataset).

- pandapower's AC-OPF does not converge at PGLib nominal load (the slack
  generator's 0-10 Mvar reactive range binds), although PGLib documents the
  case as feasible. Loads are therefore sampled at 60-95% of nominal. Cross-check
  a sample of base cases with MATPOWER or PowerModels before publication.
- `test_id` contains no violations (base cases are secure), so recall there is
  undefined. `depth_analysis.py` reports a case-level false-alarm rate instead.
- MLP is sklearn's; Phase 3 moves to PyTorch with a larger tuning budget.

## Dataset fields (`data/dataset.npz`)

`base_id`, `topo_id` (-1 intact, else branch index; names in metadata),
`p_load`, `q_load`, `p_gen`, `v_set` (per bus), `branch_status` (38),
`vm` (p.u.), `va` (deg), `loading` (% of rating, 38 branches), `converged`,
`n_q_limited` (generators at a reactive limit), `slack_p`, `slack_q`, `level`.
