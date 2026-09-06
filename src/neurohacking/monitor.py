"""Core logic: build a grid of neurons and start signal propagation.

This module knows nothing about command-line arguments. Keeping the logic
separate from the CLI makes it easy to test and to reuse from other code.
"""

from __future__ import annotations

from .grid import GridOfNeurons


def main(
    columns: int = 24,
    rows: int = 20,
    weight: float | None = None,
    threshold: float = 0.25,
    seed: int | None = None,
) -> GridOfNeurons:
    """Build a columns x rows grid, fire the origin neuron, and return the grid.

    `weight` is given to every connection, or None (the default) for random
    weights uniform between -1 and 1, reproducible with `seed`. `threshold`
    is given to every neuron. Returning the grid lets callers (and tests)
    inspect which neurons fired.
    """
    grid = GridOfNeurons(columns=columns, rows=rows, weight=weight, threshold=threshold, seed=seed)
    grid.activate_origin()
    return grid
