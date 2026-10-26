# Checkpoint analysis harness (v3)

```bash
# 0. tests (numpy only, no torch): the tools find planted structure and do not find absent structure
python tests/check_analysis.py

# 1. train with dense checkpoints around the transition (about 7-8 minutes per run on your machine)
python run_p0.py --mode train --model m1 --weight-decay 0.3 --train-seed 0 --epochs 600 --checkpoint-every 10 --save-checkpoints --skip-existing

# 2. collect: reproduces logged accuracy (GATE), captures QUERY-position activations at every layer
python -m analysis.collect --run runs/task0_split0_train0_m1_fixed_wd0.3
#    -> runs/<name>/analysis/trajectory.npz, reproduction_check.json
#    read the line "REPRODUCTION GATE: PASS/FAIL" first. If FAIL, stop and send me the json.

# 3. report: tables + figures + printed summary
python -m analysis.report --traj runs/task0_split0_train0_m1_fixed_wd0.3/analysis/trajectory.npz
```
Outputs: `curves.csv` (every group at every checkpoint), `transitions.csv`, `dr_relations.csv` (D_R of R/G/D/U pairs per layer and epoch,
with percentile among all 120 operator pairs and the task-level agreement), `circulant.csv`, `fig_sets.png`, `fig_families.png`,
`fig_dr_relations.png`, `fig_circulant.png`.

Notes
* Epoch 0 is rebuilt as `seed_all(train_seed); TinyTransformer(...)`, the same order as `train_one`. The gate checks it is near chance.
* `--contexts train` (default) restricts D_R to trained (pair, state) cells so held-out status cannot confound it.
* `--every 2` thins checkpoints for speed. Each checkpoint costs one forward pass over all 8192 cells plus 3 sites x 240 pair distances.
* D_R is scale-free: swap distance divided by the typical distance between the same operator's representations in different contexts.
* The circulant fraction is the translation-invariant share of off-diagonal similarity among state embeddings on Z_32 (structureless
  null ~ 0.03 at any dimension). It is a positive-control probe, not a claim that the model uses Fourier circuits.
* Status: numpy core fully tested with planted structure. `analysis/collect.py` (the torch layer: checkpoint loading, hooks) has been
  statically checked only; run it on one run first and send me any traceback.
