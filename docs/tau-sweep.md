# Leak time constant sweep (September 10, 2026)

Default 8x10 mesh, array engine, row critic, inputs 10 ms apart, refractory
5 ms, 15 paired seeds per tau, 1M epochs each. A log-spaced grid with the
established baseline tau = 5 in the middle, then a one-dimensional
Nelder-Mead on log(tau) between the best grid point's neighbours, then a
confirmation of the champion against the baseline at 3M epochs. Driver:
`tau-sweep-driver.py`; chart: `tau-sweep.png`.

| tau (ms) | survives a 10 ms gap | median last tenth | median to date | seeds above 0.95 |
|---|---|---|---|---|
| 0.35 | 0.0000% | 0.844 | 0.807 | 4 |
| 0.5 | 0.0000% | 0.844 | 0.807 | 4 |
| 0.71 | 0.0001% | 0.962 | 0.868 | 9 |
| 1 | 0.005% | 0.874 | 0.799 | 5 |
| 1.41 | 0.08% | 0.898 | 0.817 | 5 |
| 2 | 0.7% | 0.836 | 0.803 | 4 |
| 3.5 | 5.7% | 0.844 | 0.797 | 1 |
| 5 (baseline) | 13.5% | 0.778 | 0.781 | 1 |
| 7 | 24% | 0.675 | 0.699 | 1 |
| 10 | 37% | 0.678 | 0.650 | 0 |
| 15 | 51% | 0.598 | 0.612 | 0 |
| 25 | 67% | 0.592 | 0.578 | 0 |
| 50 | 82% | 0.556 | 0.542 | 0 |
| ∞ (no leak) | 100% | 0.548 | 0.554 | 0 |

Confirmation at 3M epochs, same seeds: tau 0.71 median last tenth 0.802,
to date 0.857; tau 5 median last tenth 0.747, to date 0.780. Tau 0.71
beat 5 on 12 of 15 seeds by accuracy to date and 7 of 15 by last tenth.

## Reading

- **Memory between inputs hurts, monotonically.** Once an appreciable
  fraction of a potential survives into the next cascade, learning degrades
  in step with that fraction, down to chance with no leak at all. The inputs
  are independent, so what survives is noise the network has to fight.
- **The peak at 0.71 is seed noise.** Below about 2 ms under 1% survives,
  so every arm there is the discharge regime within a hair; 0.35 and 0.5
  came out identical to four decimals. Their spread is what 15 seeds do when
  a negligible difference flips one threshold-edge decision and the
  trajectories part like new seeds.
- **Decision:** the default tau is 2 ms. The leak stays physically
  meaningful (potentials fade over a few milliseconds) while under 1% crosses
  the default 10 ms gap, so a default run learns as well as discharge did.
  The regime where the leak matters is inputs closer than 10 ms.
