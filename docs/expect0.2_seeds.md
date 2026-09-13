# Sweep expect0.2_seeds (September 12, 2026)

sustain_inputs, 4x10 hex grid, arrays engine, seeds 1, 2, 3, 4, 5, 6, 7, 1,000,000 epochs per arm, 20 ms epochs. Fixed: alpha 2, theta (ms) 11, refractory hops 2.9, lr 0.02, punish gain 15, weight decay 0.002, expectation start 0.2. Swept: seed. Everything else at the defaults in constants.py.
Driver: `sweep-driver.py`; figures: `expect0.2_seeds-expected.png`, `expect0.2_seeds-score.png`; every arm's checkpoint, per-epoch trace (CSV) and log are under `runs/expect0.2_seeds/` (not in git).

| seed | score to date | score last tenth | score max | expected final | expected last tenth | expected peak | dopamine last tenth | releases | total released | spikes | neurons spiked | weights at +1 | at -1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.488 | 0.490 | 1.000 | 0.06006 | 0.05985 | 0.6453 | 0.02254 | 4,386,619 | 6.357e+04 | 4,386,659 | 40 | 0 | 0 |
| 2 | 0.480 | 0.491 | 1.000 | 0.05967 | 0.05979 | 1.374 | 0.02244 | 22,019,548 | 2.902e+05 | 22,019,588 | 40 | 0 | 0 |
| 3 | 0.489 | 0.490 | 1.000 | 0.05992 | 0.06001 | 0.5747 | 0.02256 | 3,752,824 | 5.794e+04 | 3,752,864 | 40 | 0 | 0 |
| 4 | 0.475 | 0.490 | 1.000 | 0.05998 | 0.05994 | 1.391 | 0.02255 | 39,522,434 | 6.184e+05 | 39,522,474 | 40 | 0 | 0 |
| 5 | 0.489 | 0.490 | 1.000 | 0.05992 | 0.05985 | 0.2044 | 0.02251 | 2,446,117 | 3.899e+04 | 2,446,157 | 40 | 0 | 0 |
| 6 | 0.490 | 0.489 | 1.000 | 0.06002 | 0.05984 | 0.2055 | 0.02256 | 2,456,920 | 3.9e+04 | 2,456,960 | 40 | 0 | 0 |
| 7 | 0.490 | 0.490 | 1.000 | 0.05971 | 0.06 | 0.2368 | 0.02258 | 2,550,545 | 4.024e+04 | 2,550,585 | 40 | 0 | 0 |
