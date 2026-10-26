# v1 -> v2 changes
| # | v1 issue | v2 change |
|---|----------|-----------|
| 2 | OOD set used for monitoring and selection | train/val/test signature split; selection only via `best_by_val`; `val` logged every eval |
| 3 | fixed vs changing confounded (full table vs resample with replacement; coverage 62%) | one sampler, matched per-pair budgets; `fixed` == `changing(q=0.5)`; shift epoch is `--shift-epoch` |
| 4 | D as FIRST operand uncontrolled (~0.5) | both operand positions controlled; realised exposure logged per epoch |
| 5 | no in-distribution state holdout | per-signature held-out states (`id` set), shared by functionally equal pairs |
| 6 | weak coverage rule (operator in any slot) | per-slot minimum (`--min-slot-coverage`, asserted); split depends on `--split-seed` only (documented) |
| 7 | pair-weighted OOD average | `breakdown()` per pair / family / unique function in every metrics.json |
| 8 | no schedule, wd on LayerNorm/bias, sparse checkpoints, no provenance | warmup+cosine, wd groups, dense checkpoints, config hash + data fingerprint + versions |
| 9 | tautological oracle, random-fallback "lookups" | baselines.py: coverage-only lookups, shortcut probes, rectangle solver (full and information-matched), independent generator self-check |
| - | accuracy only | cross-entropy, calibration (ECE, confident-error rate), mean probability on target, D-region accuracy |
