"""Core logic: hold an error rate and display it repeatedly.

This module knows nothing about command-line arguments. Keeping the logic
separate from the CLI makes it easy to test and to reuse from other code.
"""

from __future__ import annotations

import sys
import time
from typing import TextIO
from .grid import GridOfNeurons


class ErrorRateMonitor:
    """Holds an error rate and prints it on a fixed interval.

    The rate is stored as a fraction (0.0 to 1.0) and displayed as a
    percentage with two decimals, e.g. 0.0523 -> "5.23%".
    """

    def __init__(self, output: TextIO = sys.stdout):

        self._grid = GridOfNeurons(size=10)

        self._output = output

    # --- state ---------------------------------------------------------

    @property

    def run(self):
        self._grid.activate_origin()
