"""Core logic: build a grid of neurons and start signal propagation.

This module knows nothing about command-line arguments. Keeping the logic
separate from the CLI makes it easy to test and to reuse from other code.
"""

from __future__ import annotations

from .grid import GridOfNeurons


def main(columns: int = 24, rows: int = 20, weight: float = 1.0, threshold: float = 1.0) -> GridOfNeurons:
    """Build a columns x rows grid, fire the origin neuron, and return the grid.

    `weight` is given to every connection and `threshold` to every neuron.
    Returning the grid lets callers (and tests) inspect which neurons fired.
    """
    grid = GridOfNeurons(columns=columns, rows=rows, weight=weight, threshold=threshold)
    grid.activate_origin()
    return grid
