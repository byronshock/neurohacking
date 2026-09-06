"""Command-line entry point. Parses arguments, then hands off to the monitor."""

from __future__ import annotations

import argparse
import random
import sys

from .grid import GridOfNeurons
from .monitor import main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neurohacking",
        description="Fire a signal through a hexagonal grid of neurons.",
    )
    parser.add_argument(
        "-c",
        "--columns",
        type=int,
        default=24,
        help="number of hexagons across (default: 24)",
    )
    parser.add_argument(
        "-r",
        "--rows",
        type=int,
        default=20,
        help="number of hexagon rows (default: 20)",
    )
    parser.add_argument(
        "--window",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(800, 600),
        help="window or image size in pixels (default: 800 600)",
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=float,
        default=1.0,
        help="update interval in seconds (accepted but not used yet)",
    )
    parser.add_argument(
        "-w",
        "--weight",
        type=float,
        default=None,
        help="fixed weight for every connection (default: random, uniform between -1 and 1)",
    )
    parser.add_argument(
        "-o",
        "--omega",
        type=float,
        default=0.0,
        help="proportion of connections that are small-world shortcuts, 0 to <1 (default: 0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed for the random weights, so a run can be repeated (default: chosen and printed)",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.25,
        help="input a neuron needs before it fires (default: 0.25)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="open a window showing the mesh; press Space to fire the origin, R to reset, Esc to quit",
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
        if args.show or args.save:
            try:
                from . import visualizer
            except ImportError:
                print(
                    "error: the visualizer needs pygame; install it with: pip install -e '.[viz]'",
                    file=sys.stderr,
                )
                return 2
        width, height = args.window
        seed = args.seed if args.seed is not None else random.randrange(2**31)
        settings = dict(
            columns=args.columns,
            rows=args.rows,
            weight=args.weight,
            threshold=args.threshold,
            seed=seed,
            omega=args.omega,
        )
        if args.weight is None or args.omega > 0:
            print(f"seed {seed}", file=sys.stderr)

        try:
            if args.show:
                # Show the mesh as soon as it exists; firing happens from the keyboard.
                grid = GridOfNeurons(**settings)
                visualizer.show(grid, width, height)
            else:
                grid = main(**settings)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        if args.omega > 0:
            print(
                f"omega {args.omega:g}: {len(grid.small_world_connections())} small-world "
                f"connections among {len(grid.connections)}",
                file=sys.stderr,
            )

        print(
            f"{len(grid.fired_neurons())} of {len(grid.neurons)} neurons fired in {len(grid.waves)} waves",
            file=sys.stderr,
        )
        if args.save:
            visualizer.save(grid, args.save, width, height)
            print(f"Saved {args.save}", file=sys.stderr)
    except KeyboardInterrupt:
        # Ctrl+C is the normal way to stop, so exit cleanly rather than with a traceback.
        print("\nStopped.", file=sys.stderr)
    return 0
