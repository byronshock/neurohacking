"""Core logic: build a grid of neurons and start signal propagation.

This module knows nothing about command-line arguments. Keeping the logic
separate from the CLI makes it easy to test and to reuse from other code.
"""

from __future__ import annotations

import sys
from typing import TextIO

from .grid import GridOfNeurons


class ErrorRateMonitor:
    """Owns a hexagonal grid of neurons and fires a signal from its origin."""

    def __init__(self, output: TextIO = sys.stdout, grid_size: int = 10):
        self._grid = GridOfNeurons(size=grid_size)
        self._output = output

    @property
    def grid(self) -> GridOfNeurons:
        """The grid this monitor drives."""
        return self._grid

    def run(self):
        """Fire the origin neuron once; the signal spreads across the grid."""
        self._grid.activate_origin()
