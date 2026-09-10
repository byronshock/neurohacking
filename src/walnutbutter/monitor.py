"""Core logic: build a grid, present input patterns on its bottom row, and propagate.

This module knows nothing about command-line arguments. Keeping the logic
separate from the CLI makes it easy to test and to reuse from other code.
"""

from __future__ import annotations

import random
from typing import Sequence

from .grid import GridOfNeurons
from .inputs import format_bits


def run_epoch(
    grid: GridOfNeurons,
    bits: Sequence[bool] | None = None,
    verbose: bool = True,
    noise: float = 0.0,
    rng: random.Random | None = None,
    discharge: bool = True,
) -> list:
    """Reset every neuron, present an input (random unless `bits` is given), and propagate.

    Weights, shortcuts and thresholds are untouched. Every neuron's fired
    state and potential are cleared. With `discharge=False` neurons that did
    not fire keep the sub-threshold potential they accumulated (carry-over).
    With `noise` > 0 a Gaussian draw of that standard deviation (the
    exploration used by learning) is added to every neuron's potential; each
    neuron remembers it as `noise`. Prints the input unless `verbose` is
    False. Returns the waves.
    """
    grid.reset(discharge)
    if bits is None:
        grid.new_random_input()
    else:
        grid.set_input_bits(bits)
    if noise > 0:
        grid.perturb(noise, rng or random)
    if verbose:
        stages = f"input {format_bits(grid.input_bits)}"
        if grid.code:
            stages = f"data {format_bits(grid.input_data)} -> {grid.code.name} {format_bits(grid.input_bits)}"
        print(f"epoch {grid.epoch + 1}: {stages} -> coded {format_bits(grid.input_coded)} -> bottom row {format_bits(grid.input_pattern)}")
    return grid.fire_input()


def main(
    columns: int = 8,
    rows: int = 10,
    weight: float | None = None,
    threshold: float = 0.25,
    seed: int | None = None,
    omega: float = 0.2,
    input_bits: Sequence[bool] | None = None,
    permute: bool = True,
    weight_range: tuple[float, float] = (-1.0, 1.0),
    minimum_potential: float = -1.0,
) -> GridOfNeurons:
    """Build a columns x rows grid, run one epoch on its bottom row, and return it.

    `weight` is given to every connection, or None (the default) for random
    weights uniform between -1 and 1. `threshold` is given to every neuron and
    `omega` is the proportion of small-world shortcuts. `input_bits` are the raw
    input bits (columns / 2 of them); if None they are drawn at random. They
    are complement-coded and, with `permute`, scrambled by a permutation fixed
    for the run. `seed` makes the shortcuts, the weights, the permutation and
    the random inputs all reproducible. Returning the grid lets callers (and
    tests) inspect which neurons fired.
    """
    grid = GridOfNeurons(
        columns=columns,
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
