"""Command-line entry point. Parses arguments, then hands off to the monitor."""

from __future__ import annotations

import argparse
import sys

from .monitor import ErrorRateMonitor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="error-rate-monitor",
        description="Continuously display an error rate until Ctrl+C.",
    )
    parser.add_argument(
        "error_rate",
        type=float,
        help="error rate as a fraction between 0 and 1 (e.g. 0.05 for 5%%)",
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=float,
        default=1.0,
        help="seconds between updates (default: 1.0)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns a process exit code (0 = success)."""
    args = build_parser().parse_args(argv)

    try:
        monitor = ErrorRateMonitor(args.error_rate, interval=args.interval)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        monitor.run()
    except KeyboardInterrupt:
        # Ctrl+C is the normal way to stop, so exit cleanly rather than with a traceback.
        print("\nStopped.", file=sys.stderr)
    return 0
