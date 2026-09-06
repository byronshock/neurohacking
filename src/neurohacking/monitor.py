"""Core logic: build a grid, present input patterns on its bottom row, and propagate.

This module knows nothing about command-line arguments. Keeping the logic
separate from the CLI makes it easy to test and to reuse from other code.
"""

from __future__ import annotations

from typing import Sequence

from .grid import GridOfNeurons
from .inputs import format_bits


def run_epoch(grid: GridOfNeurons, bits: Sequence[bool] | None = None, verbose: bool = True) -> list:
    """Reset every neuron, present an input (random unless `bits` is given), and propagate.

    Weights, shortcuts and thresholds are untouched; only the neurons' fired
    state and potential are cleared. Prints the input unless `verbose` is
    False. Returns the waves.
    """
    grid.reset()
    if bits is None:
        grid.new_random_input()
    else:
        grid.set_input_bits(bits)
    if verbose:
        print(f"epoch {grid.epoch + 1}: input {format_bits(grid.input_bits)} -> bottom row {format_bits(grid.input_pattern)}")
    return grid.fire_input()


def main(
    columns: int = 24,
    rows: int = 20,
    weight: float | None = None,
    threshold: float = 0.25,
    seed: int | None = None,
    omega: float = 0.05,
    input_bits: Sequence[bool] | None = None,
) -> GridOfNeurons:
    """Build a columns x rows grid, run one epoch on its bottom row, and return it.

    `weight` is given to every connection, or None (the default) for random
    weights uniform between -1 and 1. `threshold` is given to every neuron and
    `omega` is the proportion of small-world shortcuts. `input_bits` are the raw
    input bits (columns / 2 of them); if None they are drawn at random. `seed`
    makes the shortcuts, the weights and the random inputs all reproducible.
    Returning the grid lets callers (and tests) inspect which neurons fired.
    """
    grid = GridOfNeurons(
        columns=columns, rows=rows, weight=weight, threshold=threshold, seed=seed, omega=omega
    )
    run_epoch(grid, input_bits)
    return grid
