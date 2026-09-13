# Sweep bored_seeds (September 12, 2026)

sustain_inputs, 4x10 hex grid, arrays engine, seeds 1, 2, 3, 4, 5, 6, 7, 1,000,000 epochs per arm, 20 ms epochs. Fixed: alpha 2, theta (ms) 11, refractory hops 2.9, lr 0.02, punish gain 15, weight decay 0.002. Swept: seed. Everything else at the defaults in constants.py.
Driver: `sweep-driver.py`; figures: `bored_seeds-expected.png`, `bored_seeds-score.png`; every arm's checkpoint, per-epoch trace (CSV) and log are under `runs/bored_seeds/` (not in git).

| seed | score to date | score last tenth | score max | expected final | expected last tenth | expected peak | dopamine last tenth | releases | total released | spikes | neurons spiked | weights at +1 | at -1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.481 | 0.499 | 1.000 | 0.07047 | 0.07022 | 1.567 | 0.02618 | 20,489,926 | 2.209e+05 | 20,489,966 | 40 | 0 | 0 |
| 2 | 0.493 | 0.500 | 1.000 | 0.06991 | 0.07011 | 1.551 | 0.02604 | 19,950,584 | 2.094e+05 | 19,950,624 | 40 | 0 | 0 |
| 3 | 0.494 | 0.499 | 1.000 | 0.07005 | 0.07023 | 1.555 | 0.02611 | 20,279,541 | 2.138e+05 | 20,279,581 | 40 | 0 | 0 |
| 4 | 0.493 | 0.499 | 1.000 | 0.07035 | 0.07029 | 1.543 | 0.02615 | 20,081,996 | 2.116e+05 | 20,082,036 | 40 | 0 | 0 |
| 5 | 0.494 | 0.500 | 1.000 | 0.07013 | 0.07013 | 1.52 | 0.02607 | 19,914,871 | 2.071e+05 | 19,914,911 | 40 | 0 | 0 |
| 6 | 0.495 | 0.499 | 1.000 | 0.07035 | 0.07023 | 1.544 | 0.02617 | 20,024,781 | 2.075e+05 | 20,024,821 | 40 | 0 | 0 |
| 7 | 0.495 | 0.499 | 1.000 | 0.07002 | 0.07031 | 1.554 | 0.02616 | 20,686,736 | 2.25e+05 | 20,686,776 | 40 | 0 | 0 |
