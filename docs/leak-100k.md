# Sweep leak-100k (September 12, 2026)

sustain_inputs, 4x10 hex grid, arrays engine, seed 1, 100,000 epochs per arm, 20 ms epochs. Fixed: alpha 2, theta (ms) 11, refractory hops 2.9, lr 0.02, punish gain 15, weight decay 0.002, seed 1. Swept: . Everything else at the defaults in constants.py.
Driver: `sweep-driver.py`; figures: `leak-100k-expected.png`, `leak-100k-score.png`; every arm's checkpoint, per-epoch trace (CSV) and log are under `runs/leak-100k/` (not in git).

|  | score to date | score last tenth | score max | expected final | expected last tenth | expected peak | dopamine last tenth | releases | total released | spikes | neurons spiked | weights at +1 | at -1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  | 0.498 | 0.501 | 1.000 | 0.09715 | 0.1028 | 0.4567 | 0.02487 | 2,934,278 | 2.223e+04 | 2,934,318 | 40 | 0 | 0 |
