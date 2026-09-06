"""Core logic: hold an error rate and display it repeatedly.

This module knows nothing about command-line arguments. Keeping the logic
separate from the CLI makes it easy to test and to reuse from other code.
"""

from __future__ import annotations

import sys
import time
from typing import TextIO


class ErrorRateMonitor:
    """Holds an error rate and prints it on a fixed interval.

    The rate is stored as a fraction (0.0 to 1.0) and displayed as a
    percentage with two decimals, e.g. 0.0523 -> "5.23%".
    """

    def __init__(
        self,
        error_rate: float,
        interval: float = 1.0,
        output: TextIO = sys.stdout,
    ) -> None:
        if interval <= 0:
            raise ValueError(f"interval must be positive, got {interval}")
        self._error_rate = 0.0
        self._interval = interval
        self._output = output
        self.set_error_rate(error_rate)

    # --- state ---------------------------------------------------------

    @property
    def error_rate(self) -> float:
        """The current error rate as a fraction between 0 and 1."""
        return self._error_rate

    def set_error_rate(self, value: float) -> None:
        """Change the displayed error rate. Takes effect on the next tick."""
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"error rate must be between 0 and 1, got {value}")
        self._error_rate = value

    @property
    def interval(self) -> float:
        """Seconds between display updates."""
        return self._interval

    # --- display -------------------------------------------------------

    def format_error_rate(self) -> str:
        """Render the current rate as a percentage with two decimals."""
        return f"{self._error_rate * 100:.2f}%"

    def display_once(self) -> None:
        """Write one line showing the current rate."""
        print(self.format_error_rate(), file=self._output, flush=True)

    def run(self, max_ticks: int | None = None) -> None:
        """Display the rate every `interval` seconds.

        Runs forever unless `max_ticks` is given (useful for tests). Stopping
        with Ctrl+C raises KeyboardInterrupt, which the caller handles.
        """
        ticks = 0
        while max_ticks is None or ticks < max_ticks:
            self.display_once()
            ticks += 1
            if max_ticks is None or ticks < max_ticks:
                time.sleep(self._interval)
