"""Core logic: build a grid, present an input pattern on its bottom row, and propagate.

This module knows nothing about command-line arguments. Keeping the logic
separate from the CLI makes it easy to test and to reuse from other code.
"""

from __future__ import annotations

from typing import Sequence

from .grid import GridOfNeurons
from .inputs import complement_code, format_bits, random_bits


def prepare_input(grid: GridOfNeurons, bits: Sequence[bool] | None = None, seed: int | None = None) -> list[bool]:
    """Set the grid's input pattern from raw bits (random if None), complement-coded.

    The raw bits number half the columns; complement coding doubles them so
    the pattern covers the whole bottom row. Returns the raw bits used.
    """
    if grid.columns % 2:
        raise ValueError(f"complement coding needs an even number of columns, got {grid.columns}")
    if bits is None:
        bits = random_bits(grid.columns // 2, seed)
    bits = [bool(b) for b in bits]
    if len(bits) != grid.columns // 2:
        raise ValueError(f"expected {grid.columns // 2} input bits for {grid.columns} columns, got {len(bits)}")
    grid.set_input(complement_code(bits))
    print(f"input {format_bits(bits)} -> bottom row {format_bits(grid.input_pattern)}")
    return bits


def main(
    columns: int = 24,
    rows: int = 20,
    weight: float | None = None,
    threshold: float = 0.25,
    seed: int | None = None,
    omega: float = 0.05,
    input_bits: Sequence[bool] | None = None,
) -> GridOfNeurons:
    """Build a columns x rows grid, fire its bottom row with an input pattern, and return it.

    `weight` is given to every connection, or None (the default) for random
    weights uniform between -1 and 1. `threshold` is given to every neuron and
    `omega` is the proportion of small-world shortcuts. `input_bits` are the raw
    input bits (columns / 2 of them); if None they are drawn at random. `seed`
    makes the shortcuts, the weights and the random input all reproducible.
    Returning the grid lets callers (and tests) inspect which neurons fired.
    """
    grid = GridOfNeurons(
        columns=columns, rows=rows, weight=weight, threshold=threshold, seed=seed, omega=omega
    )
    prepare_input(grid, input_bits, seed)
    grid.fire_input()
    return grid
