# Sweep lr-sweep-low (September 12, 2026)

sustain_inputs, 8x10 hex grid, arrays engine, seed 1 (paired), 1,000,000 epochs per arm, 20 ms epochs. Fixed: alpha 5, theta (ms) 25, refractory hops 2.5. Swept: lr. Everything else at the defaults in constants.py.
Driver: `sweep-driver.py`; figures: `lr-sweep-low-expected.png`, `lr-sweep-low-score.png`; every arm's checkpoint, per-epoch trace (CSV) and log are under `runs/lr-sweep-low/` (not in git).

| lr | score to date | score last tenth | score max | expected final | expected last tenth | expected peak | dopamine last tenth | releases | total released | spikes | neurons spiked | weights at +1 | at -1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0001 | 0.506 | 0.505 | 1.000 | 0.00387 | 0.00365 | 0.005169 | 0.002951 | 159,524,143 | 3984 | 159,524,223 | 80 | 0 | 0 |
| 0.0002 | 0.506 | 0.505 | 1.000 | 0.004303 | 0.003944 | 0.005509 | 0.003596 | 160,117,806 | 3796 | 160,117,885 | 79 | 0 | 0 |
| 0.0005 | 0.507 | 0.506 | 1.000 | 0.004281 | 0.002837 | 0.005575 | 0.002744 | 159,280,983 | 4063 | 159,281,063 | 80 | 0 | 0 |
| 0.001 | 0.507 | 0.506 | 1.000 | 0.004236 | 0.004059 | 0.005518 | 0.003581 | 159,937,682 | 4070 | 159,937,761 | 79 | 0 | 0 |
| 0.002 | 0.506 | 0.506 | 1.000 | 0.00463 | 0.004738 | 0.005236 | 0.004061 | 160,092,732 | 3920 | 160,092,812 | 80 | 0 | 0 |
| 0.005 | 0.506 | 0.506 | 1.000 | 0.004877 | 0.004238 | 0.005632 | 0.003764 | 161,270,380 | 4214 | 161,270,460 | 80 | 0 | 0 |
