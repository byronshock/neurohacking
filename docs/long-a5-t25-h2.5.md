# Sweep long-a5-t25-h2.5 (September 12, 2026)

sustain_inputs, 8x10 hex grid, arrays engine, seed 1 (paired), 10,000,000 epochs per arm, 20 ms epochs. Fixed: alpha 5, theta (ms) 25, refractory hops 2.5. Swept: . Everything else at the defaults in constants.py.
Driver: `sweep-driver.py`; figures: `long-a5-t25-h2.5-expected.png`, `long-a5-t25-h2.5-score.png`; every arm's checkpoint, per-epoch trace (CSV) and log are under `runs/long-a5-t25-h2.5/` (not in git).

|  | score to date | score last tenth | score max | expected final | expected last tenth | expected peak | dopamine last tenth | releases | total released | spikes | neurons spiked | weights at +1 | at -1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  | 0.502 | 0.500 | 1.000 | 2.096e-05 | 2.096e-05 | 0.005509 | 1.897e-05 | 2,045,868,578 | 1.044e+04 | 2,045,868,658 | 80 | 21 | 0 |
