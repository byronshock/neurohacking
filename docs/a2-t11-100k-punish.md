# Sweep a2-t11-100k-punish (September 12, 2026)

sustain_inputs, 4x10 hex grid, arrays engine, seed 1 (paired), 100,000 epochs per arm, 20 ms epochs. Fixed: alpha 2, theta (ms) 11, refractory hops 2.5, lr 0.02. Swept: . Everything else at the defaults in constants.py.
Driver: `sweep-driver.py`; figures: `a2-t11-100k-punish-expected.png`, `a2-t11-100k-punish-score.png`; every arm's checkpoint, per-epoch trace (CSV) and log are under `runs/a2-t11-100k-punish/` (not in git).

|  | score to date | score last tenth | score max | expected final | expected last tenth | expected peak | dopamine last tenth | releases | total released | spikes | neurons spiked | weights at +1 | at -1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|  | 0.500 | 0.499 | 1.000 | 1.422 | 1.412 | 1.422 | 1.335 | 10,294,688 | 1.399e+05 | 10,294,724 | 36 | 103 | 0 |
