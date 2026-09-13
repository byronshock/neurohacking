# Sweep teacher_hops (September 12, 2026)

improved_sustain, 10x7 hex grid, arrays engine, seed 1, 1,000,000 epochs per arm, 20 ms epochs. Fixed: alpha 2, theta (ms) 11, lr 0.02, weight decay 0.002, seed 1. Swept: refractory hops. Everything else at the defaults in constants.py.
Driver: `sweep-driver.py`; figures: `teacher_hops-expected.png`, `teacher_hops-score.png`; every arm's checkpoint, per-epoch trace (CSV) and log are under `runs/teacher_hops/` (not in git).

| refractory hops | score to date | score last tenth | score max | expected final | expected last tenth | expected peak | dopamine last tenth | releases | total released | spikes | neurons spiked | weights at +1 | at -1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.1 | 0.500 | 0.500 | 1.000 | 1.354e+08 | 1.287e+08 | 1.354e+08 | 4e-05 | 11,236,260 | 4.641e+04 | 11,236,330 | 70 | 0 | 0 |
| 1.5 | 0.500 | 0.500 | 1.000 | 1.345e+08 | 1.278e+08 | 1.345e+08 | 5e-05 | 11,164,482 | 4.665e+04 | 11,164,552 | 70 | 0 | 0 |
| 1.9 | 0.500 | 0.500 | 1.000 | 1.336e+08 | 1.27e+08 | 1.336e+08 | 4e-05 | 11,196,498 | 4.741e+04 | 11,196,568 | 70 | 0 | 0 |
| 2.1 | 0.500 | 0.500 | 1.000 | 1.326e+08 | 1.26e+08 | 1.326e+08 | 5e-05 | 11,190,999 | 4.769e+04 | 11,191,069 | 70 | 0 | 0 |
| 2.5 | 0.500 | 0.500 | 1.000 | 1.323e+08 | 1.257e+08 | 1.323e+08 | 6.5e-05 | 11,144,198 | 4.781e+04 | 11,144,268 | 70 | 0 | 0 |
| 2.9 | 0.500 | 0.500 | 1.000 | 1.318e+08 | 1.253e+08 | 1.318e+08 | 7.5e-05 | 11,164,144 | 4.777e+04 | 11,164,214 | 70 | 0 | 0 |
| 3.1 | 0.500 | 0.500 | 1.000 | 1.315e+08 | 1.25e+08 | 1.315e+08 | 8e-05 | 11,159,409 | 4.803e+04 | 11,159,479 | 70 | 0 | 0 |
| 3.5 | 0.500 | 0.500 | 1.000 | 1.314e+08 | 1.249e+08 | 1.314e+08 | 8.5e-05 | 11,137,570 | 4.812e+04 | 11,137,640 | 70 | 0 | 0 |
| 3.9 | 0.500 | 0.500 | 1.000 | 1.312e+08 | 1.247e+08 | 1.312e+08 | 8.5e-05 | 11,145,551 | 4.79e+04 | 11,145,621 | 70 | 0 | 0 |
