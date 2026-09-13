# Sweep expect20_seeds (September 12, 2026)

sustain_inputs, 4x10 hex grid, arrays engine, seeds 1, 2, 3, 4, 5, 6, 7, 1,000,000 epochs per arm, 20 ms epochs. Fixed: alpha 2, theta (ms) 11, refractory hops 2.9, lr 0.02, punish gain 15, weight decay 0.002, expectation start 20. Swept: seed. Everything else at the defaults in constants.py.
Driver: `sweep-driver.py`; figures: `expect20_seeds-expected.png`, `expect20_seeds-score.png`; every arm's checkpoint, per-epoch trace (CSV) and log are under `runs/expect20_seeds/` (not in git).

| seed | score to date | score last tenth | score max | expected final | expected last tenth | expected peak | dopamine last tenth | releases | total released | spikes | neurons spiked | weights at +1 | at -1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.490 | 0.490 | 1.000 | 0.06006 | 0.05985 | 20 | 0.02254 | 4,271,465 | 5.003e+04 | 4,271,505 | 40 | 0 | 0 |
| 2 | 0.462 | 0.491 | 1.000 | 0.05967 | 0.05979 | 20 | 0.02244 | 2,513,507 | 4.355e+04 | 2,513,547 | 40 | 0 | 0 |
| 3 | 0.490 | 0.490 | 1.000 | 0.05992 | 0.06001 | 20 | 0.02256 | 4,257,445 | 4.982e+04 | 4,257,485 | 40 | 0 | 0 |
| 4 | 0.473 | 0.490 | 1.000 | 0.05998 | 0.05994 | 20 | 0.02255 | 3,395,928 | 4.702e+04 | 3,395,968 | 40 | 0 | 0 |
| 5 | 0.490 | 0.490 | 1.000 | 0.05992 | 0.05985 | 20 | 0.02251 | 4,252,048 | 4.984e+04 | 4,252,088 | 40 | 0 | 0 |
| 6 | 0.490 | 0.489 | 1.000 | 0.06002 | 0.05984 | 20 | 0.02256 | 4,322,590 | 5.02e+04 | 4,322,630 | 40 | 0 | 0 |
| 7 | 0.461 | 0.490 | 1.000 | 0.05971 | 0.06 | 20 | 0.02258 | 3,028,801 | 4.67e+04 | 3,028,841 | 40 | 0 | 0 |
