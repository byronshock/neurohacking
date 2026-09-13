# Sweep teacher_hops_nodecay (September 12, 2026)

improved_sustain, 10x7 hex grid, arrays engine, seed 1, 1,000,000 epochs per arm, 20 ms epochs. Fixed: alpha 2, theta (ms) 11, lr 0.02, weight decay 0, seed 1. Swept: refractory hops. Everything else at the defaults in constants.py.
Driver: `sweep-driver.py`; figures: `teacher_hops_nodecay-expected.png`, `teacher_hops_nodecay-score.png`; every arm's checkpoint, per-epoch trace (CSV) and log are under `runs/teacher_hops_nodecay/` (not in git).

| refractory hops | score to date | score last tenth | score max | expected final | expected last tenth | expected peak | dopamine last tenth | releases | total released | spikes | neurons spiked | weights at +1 | at -1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.01 | 0.500 | 0.500 | 1.000 | 1.231e+08 | 1.131e+08 | 1.231e+08 | -9e-05 | 261,360,027 | 4.383e+06 | 261,360,097 | 70 | 0 | 0 |
| 1.5 | 0.499 | 0.500 | 1.000 | 2.253e+08 | 2.149e+08 | 2.253e+08 | 9e-05 | 102,701,993 | 2.052e+06 | 102,702,063 | 70 | 12 | 544 |
| 1.99 | 0.500 | 0.500 | 1.000 | 2.856e+08 | 2.71e+08 | 2.856e+08 | -9e-05 | 259,340,221 | 2.356e+06 | 259,340,291 | 70 | 0 | 0 |
| 2.01 | 0.500 | 0.500 | 1.000 | 9.476e+07 | 8.878e+07 | 9.476e+07 | -9e-05 | 265,036,948 | 2.774e+06 | 265,037,018 | 70 | 0 | 0 |
| 2.5 | 0.500 | 0.500 | 1.000 | 3.516e+08 | 3.331e+08 | 3.516e+08 | -0.000115 | 199,766,146 | 2.719e+06 | 199,766,216 | 70 | 0 | 0 |
| 2.99 | 0.500 | 0.500 | 1.000 | 2.298e+08 | 2.188e+08 | 2.298e+08 | -9e-05 | 266,749,528 | 1.7e+06 | 266,749,598 | 70 | 0 | 0 |
| 3.01 | 0.500 | 0.500 | 1.000 | 1.354e+08 | 1.231e+08 | 1.354e+08 | 0.000515 | 260,526,960 | 2.108e+06 | 260,527,030 | 70 | 0 | 42 |
| 3.5 | 0.500 | 0.500 | 1.000 | 3.004e+08 | 2.856e+08 | 3.004e+08 | -8.5e-05 | 223,637,752 | 2.295e+06 | 223,637,822 | 70 | 1 | 0 |
| 3.99 | 0.500 | 0.500 | 1.000 | 1.903e+08 | 1.809e+08 | 1.903e+08 | -9e-05 | 270,267,487 | 1.323e+06 | 270,267,557 | 70 | 0 | 0 |
| 4.01 | 0.501 | 0.500 | 1.000 | 2.961e+08 | 2.727e+08 | 2.961e+08 | 9e-05 | 217,418,143 | 1.822e+06 | 217,418,213 | 70 | 0 | 324 |
