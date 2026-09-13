"""Core logic: build a grid, present input patterns on its bottom row, and propagate.

This module knows nothing about command-line arguments. Keeping the logic
separate from the CLI makes it easy to test and to reuse from other code.
"""

from __future__ import annotations

import random
from typing import Sequence

from .constants import ACROSS, MINIMUM_POTENTIAL, OMEGA, ROWS, THRESHOLD, WEIGHT_RANGE
from .grid import GridOfNeurons
from .inputs import format_bits


def run_epoch(
    grid: GridOfNeurons,
    bits: Sequence[bool] | None = None,
    verbose: bool = True,
    noise: float = 0.0,
    rng: random.Random | None = None,
    discharge: bool = False,
    time: float | None = None,
) -> list:
    """One epoch: present an input (random unless `bits` is given) at `time`, and run the schedule to the next input's time.

    Weights, shortcuts and thresholds are untouched by this function (the
    dopamine rule, if the network has one, learns as it runs; the teacher
    rule accumulates eligibility for the Teacher to pay at the read). Every
    neuron's fired-this-epoch state is cleared; potentials are kept (there
    is no leak), or zeroed with `discharge=True`. `time` defaults to the
    network's interval after the last input. With `noise` > 0 a Gaussian
    draw of that standard deviation (the exploration) is added to every
    neuron's potential; each neuron remembers it as `noise`. Prints the
    input unless `verbose` is False. Returns the epoch's waves.
    """
    grid.reset(discharge)
    if bits is None:
        grid.new_random_input(time)
    else:
        grid.set_input_bits(bits, time)
    if noise > 0:
        grid.perturb(noise, rng or random)
    if verbose:
        stages = f"input {format_bits(grid.input_bits)}"
        if grid.code:
            stages = f"data {format_bits(grid.input_data)} -> {grid.code.name} {format_bits(grid.input_bits)}"
        print(f"epoch {grid.epoch + 1} at {grid.input_time:g} ms: {stages} -> coded {format_bits(grid.input_coded)} -> bottom row {format_bits(grid.input_pattern)}")
    return grid.fire_input()


def main(
    across: int = ACROSS,
    rows: int = ROWS,
    weight: float | None = None,
    threshold: float = THRESHOLD,
    seed: int | None = None,
    omega: float = OMEGA,
    input_bits: Sequence[bool] | None = None,
    permute: bool = True,
    weight_range: tuple[float, float] = WEIGHT_RANGE,
    minimum_potential: float = MINIMUM_POTENTIAL,
) -> GridOfNeurons:
    """Build a across x rows grid, run one epoch on its bottom row, and return it.

    `weight` is given to every connection, or None (the default) for random
    weights uniform between -1 and 1. `threshold` is given to every neuron and
    `omega` is the proportion of small-world shortcuts. `input_bits` are the raw
    input bits (across / 2 of them); if None they are drawn at random. They
    are complement-coded and, with `permute`, scrambled by a permutation fixed
    for the run. `seed` makes the shortcuts, the weights, the permutation and
    the random inputs all reproducible. Returning the grid lets callers (and
    tests) inspect which neurons fired.
    """
    grid = GridOfNeurons(
        across=across,
        rows=rows,
        weight=weight,
        threshold=threshold,
        seed=seed,
        omega=omega,
        permute=permute,
        weight_range=weight_range,
        minimum_potential=minimum_potential,
    )
    run_epoch(grid, input_bits)
    return grid
