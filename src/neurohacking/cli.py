"""Command-line entry point. Parses arguments, then hands off to the monitor."""

from __future__ import annotations

import argparse
import sys

from .monitor import main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neurohacking",
        description="Fire a signal through a hexagonal grid of neurons.",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=10,
        help="grid radius in neurons (default: 10)",
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=float,
        default=1.0,
        help="update interval in seconds (accepted but not used yet)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="open a window showing the grid after the signal has spread",
    )
    parser.add_argument(
        "--save",
        metavar="PATH",
        help="write a picture of the grid to PATH (e.g. grid.png)",
    )
    return parser


def cli_main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns a process exit code (0 = success)."""
    args = build_parser().parse_args(argv)

    try:
        grid = main(grid_size=args.size)
        if args.show or args.save:
            try:
                from . import visualizer
            except ImportError:
                print(
                    "error: the visualizer needs pygame; install it with: pip install -e '.[viz]'",
                    file=sys.stderr,
                )
                return 2
            if args.save:
                visualizer.save(grid, args.save)
                print(f"Saved {args.save}", file=sys.stderr)
            if args.show:
                visualizer.show(grid)
    except KeyboardInterrupt:
        # Ctrl+C is the normal way to stop, so exit cleanly rather than with a traceback.
        print("\nStopped.", file=sys.stderr)
    return 0
