"""Core logic: build a grid of neurons and start signal propagation.

This module knows nothing about command-line arguments. Keeping the logic
separate from the CLI makes it easy to test and to reuse from other code.
"""

from __future__ import annotations

from .grid import GridOfNeurons


def main(grid_size: int = 10) -> GridOfNeurons:
    """Build a grid, fire the origin neuron, and return the grid.

    Returning the grid lets callers (and tests) inspect which neurons fired.
    """
    grid = GridOfNeurons(size=grid_size)
    grid.activate_origin()
    return grid
