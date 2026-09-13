# Sweep lr-sweep (September 12, 2026)

sustain_inputs, 8x10 hex grid, arrays engine, seed 1 (paired), 1,000,000 epochs per arm, 20 ms epochs. Fixed: alpha 5, theta (ms) 25, refractory hops 2.5. Swept: lr. Everything else at the defaults in constants.py.
Driver: `sweep-driver.py`; figures: `lr-sweep-expected.png`, `lr-sweep-score.png`; every arm's checkpoint, per-epoch trace (CSV) and log are under `runs/lr-sweep/` (not in git).

| lr | score to date | score last tenth | score max | expected final | expected last tenth | expected peak | dopamine last tenth | releases | total released | spikes | neurons spiked | weights at +1 | at -1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 0.507 | 0.507 | 1.000 | 0.005245 | 0.004744 | 0.005487 | 0.004206 | 163,232,468 | 4048 | 163,232,548 | 80 | 4 | 0 |
| 0.02 | 0.507 | 0.506 | 1.000 | 0.004713 | 0.004181 | 0.005901 | 0.003578 | 168,032,427 | 4365 | 168,032,507 | 80 | 8 | 0 |
| 0.03 | 0.506 | 0.504 | 1.000 | 0.00315 | 0.003465 | 0.005509 | 0.002762 | 171,135,090 | 3726 | 171,135,170 | 80 | 13 | 0 |
| 0.05 | 0.505 | 0.503 | 1.000 | 0.0022 | 0.002682 | 0.005506 | 0.002067 | 179,456,231 | 3545 | 179,456,311 | 80 | 17 | 0 |
| 0.1 | 0.504 | 0.502 | 1.000 | 0.001404 | 0.00102 | 0.004987 | 0.0008679 | 191,649,031 | 2445 | 191,649,111 | 80 | 19 | 0 |
