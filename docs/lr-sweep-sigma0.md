# Sweep lr-sweep-sigma0 (September 12, 2026)

sustain_inputs, 8x10 hex grid, arrays engine, seed 1 (paired), 1,000,000 epochs per arm, 20 ms epochs. Fixed: alpha 5, theta (ms) 25, refractory hops 2.5, sigma 0. Swept: lr. Everything else at the defaults in constants.py.
Driver: `sweep-driver.py`; figures: `lr-sweep-sigma0-expected.png`, `lr-sweep-sigma0-score.png`; every arm's checkpoint, per-epoch trace (CSV) and log are under `runs/lr-sweep-sigma0/` (not in git).

| lr | score to date | score last tenth | score max | expected final | expected last tenth | expected peak | dopamine last tenth | releases | total released | spikes | neurons spiked | weights at +1 | at -1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 0.500 | 0.500 | 0.750 | 1.669e-05 | 1.669e-05 | 1.669e-05 | 1.51e-05 | 169,999,727 | 16 | 169,999,793 | 66 | 0 | 0 |
| 0.02 | 0.500 | 0.500 | 0.750 | 1.669e-05 | 1.669e-05 | 1.669e-05 | 1.51e-05 | 169,999,727 | 16 | 169,999,793 | 66 | 0 | 0 |
| 0.03 | 0.500 | 0.500 | 0.750 | 1.669e-05 | 1.669e-05 | 1.669e-05 | 1.51e-05 | 169,999,727 | 16 | 169,999,793 | 66 | 0 | 0 |
| 0.05 | 0.500 | 0.500 | 0.750 | 1.669e-05 | 1.669e-05 | 1.669e-05 | 1.51e-05 | 169,999,727 | 16 | 169,999,793 | 66 | 0 | 0 |
| 0.1 | 0.500 | 0.500 | 0.750 | 1.669e-05 | 1.669e-05 | 1.669e-05 | 1.51e-05 | 169,999,727 | 16 | 169,999,793 | 66 | 0 | 0 |
