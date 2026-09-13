# Sweep bored-100k (September 12, 2026)

sustain_inputs, 4x10 hex grid, arrays engine, seed 1, 100,000 epochs per arm, 20 ms epochs. Fixed: alpha 2, theta (ms) 11, refractory hops 2.9, lr 0.02, punish gain 15, weight decay 0.002, seed 1. Swept: . Everything else at the defaults in constants.py.
Driver: `sweep-driver.py`; figures: `bored-100k-expected.png`, `bored-100k-score.png`; every arm's checkpoint, per-epoch trace (CSV) and log are under `runs/bored-100k/` (not in git).

|  | score to date | score last tenth | score max | expected final | expected last tenth | expected peak | dopamine last tenth | releases | total released | spikes | neurons spiked | weights at +1 | at -1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  | 0.472 | 0.476 | 1.000 | 1.407 | 1.367 | 1.407 | 1.506 | 12,770,957 | 1.247e+05 | 12,770,997 | 40 | 0 | 0 |
