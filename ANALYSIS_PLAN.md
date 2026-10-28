# Analysis plan (fill in and freeze BEFORE running `analysis.report` on any run)

## G0. P0 gate (status)
Criterion proposed in chat: at wd 0.3, >= 8 of 10 seeds reach >= 90% held-out-function test accuracy at epoch 600.
Result (task 0, split 0, train seeds 0-9): 9/10 (one seed at 84.8%; one at 96.9%; mean 98.1%). Exact 95% interval for
a 9/10 success rate: 55%-99.7%. Scope: ONE task seed and ONE split seed. Confirm this is the criterion you adopt, and add
split seeds 1-2 before claiming anything beyond this world.

## G1. Measurement gates (a run is analysed only if all pass)
- `reproduction_check.json`: recomputed accuracies match logged ones (|diff| < 1e-3) at every logged checkpoint, and epoch-0 train accuracy is near chance.
- `python tests/check_analysis.py` passes (tools find planted structure, and do not find it when absent).

## Design
- PRIMARY contrast: within one run, before vs after its transition (same seed, same wd, same data). Only training progress differs.
- SECONDARY contrast: generalizing (wd 0.3) vs memorizing (wd 0.1) runs. Confounded by weight decay itself (it changes weight norms),
  so never interpret it alone.
- Report every seed. The weakest seed is included. No post-hoc seed selection.

## Definitions (fixed now)
- transition epoch = first epoch at which the curve stays >= 0.9 for 3 consecutive checkpoints (checkpoints are 10 epochs apart).
- onset = interpolated epoch at which the curve first crosses 0.5.
- D_R = operator-swap distance at the QUERY position (see analysis/core.py), contexts = trained cells only, per layer;
  reported as value AND percentile among all 120 operator pairs, relative to epoch 0.

## Hypotheses and what would falsify them
| id | prediction | falsified if |
|----|-----------|--------------|
| H1 behaviour | operator families transition at different epochs | all family curves have transition epochs within one checkpoint spacing |
| H2 redundancy | after the transition D_R(R1,R2) is in the closest 10% of operator pairs and lower than at epoch 0; D_R(U1,U2) is not | R1/R2 percentile >= 0.5 after the transition (the model keeps functionally identical operators distinct) |
| H3 partial overlap | D_R(G1,G2) and D_R(D1,D2) lie between R and U | G or D at or beyond U in most seeds |
| H4 positive control | state-embedding circulant fraction exceeds the permutation-null 95th percentile after the transition in at least some seeds | never exceeds the null (then the embeddings carry no Z_32 structure: report it, and do not use embeddings as the structure probe) |
| H5 timing | ID, val and test curves cross 0.5 within one checkpoint spacing of each other | they separate by more than 3 spacings |

H6: families whose second operator is D, G or U cross 50% earlier than others (C→D at epoch 97, A→G at 140, D→D at 131, versus about 215–265 for most).
H7: the timing of the R1/R2 merge relative to the test transition. dr_relations.csv has this per epoch, but I haven’t seen it.

A null or opposite result on any row is a result. Record it in the paper as such.

## Not yet (only after G0, G1 and H1-H5 are examined)
Component ablations, I_ij interaction, functional signatures Phi, degeneracy G. Needs its own plan: which components, which
outcome (log-prob of the correct token), which ablation (mean and resample, not only zero), which controls.
