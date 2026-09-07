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
) -> list:
    """Reset every neuron, present an input (random unless `bits` is given), and propagate.

    Weights, shortcuts and thresholds are untouched; only the neurons' fired
    state and potential are cleared. With `noise` > 0 every neuron starts the
    epoch with a Gaussian random potential of that standard deviation (the
    exploration used by learning); each neuron remembers it as `noise`.
    Prints the input unless `verbose` is False. Returns the waves.
    """
    grid.reset()
    if bits is None:
        grid.new_random_input()
    else:
        grid.set_input_bits(bits)
    if noise > 0:
        rng = rng or random
        for neuron in grid.neurons.values():
            neuron.noise = rng.gauss(0.0, noise)
            neuron.potential = neuron.noise
    if verbose:
        print(
            f"epoch {grid.epoch + 1}: input {format_bits(grid.input_bits)} -> coded "
            f"{format_bits(grid.input_coded)} -> bottom row {format_bits(grid.input_pattern)}"
        )
    return grid.fire_input()


def main(
    columns: int = 8,
    rows: int = 8,
    weight: float | None = None,
    threshold: float = 0.25,
    seed: int | None = None,
    omega: float = 0.05,
    input_bits: Sequence[bool] | None = None,
    permute: bool = True,
    weight_range: tuple[float, float] = (-1.0, 1.0),
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
    )
    run_epoch(grid, input_bits)
    return grid
