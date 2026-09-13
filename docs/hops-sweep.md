# Sweep hops-sweep (September 12, 2026)

sustain_inputs, 8x10 hex grid, arrays engine, seed 1 (paired), 1,000,000 epochs per arm, 20 ms epochs. Fixed: alpha 5, theta (ms) 5.5. Swept: refractory hops. Everything else at the defaults in constants.py.
Driver: `sweep-driver.py`; figures: `hops-sweep-expected.png`, `hops-sweep-score.png`; every arm's checkpoint, per-epoch trace (CSV) and log are under `runs/hops-sweep/` (not in git).

| refractory hops | score to date | score last tenth | score max | expected final | expected last tenth | expected peak | dopamine last tenth | releases | total released | spikes | neurons spiked | weights at +1 | at -1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.500 | 0.500 | 1.000 | 0.01089 | 0.01084 | 0.01729 | 0.00958 | 215,818,166 | 1.122e+04 | 215,818,238 | 72 | 6 | 0 |
| 2.5 | 0.501 | 0.500 | 1.000 | 0.02506 | 0.02505 | 0.03091 | 0.02266 | 188,357,576 | 2.515e+04 | 188,357,654 | 78 | 16 | 0 |
| 3 | 0.500 | 0.500 | 0.875 | 0.002489 | 0.002486 | 0.01437 | 0.002284 | 237,827,985 | 4069 | 237,828,064 | 79 | 23 | 0 |
| 3.5 | 0.500 | 0.500 | 1.000 | 0.009483 | 0.009714 | 0.01617 | 0.008791 | 222,212,458 | 9910 | 222,212,537 | 79 | 25 | 0 |
| 4 | 0.500 | 0.500 | 1.000 | 0.001358 | 0.001831 | 0.008959 | 0.001648 | 252,912,443 | 2369 | 252,912,522 | 79 | 21 | 0 |
| 4.5 | 0.500 | 0.500 | 1.000 | 0.005032 | 0.004873 | 0.01487 | 0.004704 | 242,009,130 | 6254 | 242,009,209 | 79 | 27 | 0 |
| 5 | 0.500 | 0.500 | 0.875 | 0.001072 | 0.001984 | 0.008432 | 0.001265 | 261,730,707 | 2444 | 261,730,787 | 80 | 22 | 0 |
