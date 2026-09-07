"""Command-line entry point. Parses arguments, then hands off to the monitor."""

from __future__ import annotations

import argparse
import random
import sys

from .cartesian import CartesianNodes
from .inputs import parse_bits
from .learning import ELIGIBILITIES, TARGETS, Teacher
from .monitor import main, run_epoch
from .neuron import Neuron
from .persistence import checkpoint, restore, resume_teacher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="walnutbutter",
        description=(
            "A hexagonal mesh of neurons that learns to reproduce its input on its output row. "
            "By default it opens a window, free-runs, learns, and reports accuracy until you close it."
        ),
    )
    parser.add_argument(
        "-c",
        "--columns",
        type=int,
        default=8,
        help="number of hexagons across (default: 8)",
    )
    parser.add_argument(
        "-r",
        "--rows",
        type=int,
        default=10,
        help="number of hexagon rows (default: 10)",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        metavar="N",
        nargs="?",
        const=64,
        default=None,
        help="instead of the hex grid: N neurons at random Cartesian positions in [-1, 1] x [-1, 1] "
        "(N defaults to 64; shown, not yet wired)",
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
        default=0.05,
        help="proportion of connections that are small-world shortcuts, 0 to <1 (default: 0.05)",
    )
    parser.add_argument(
        "-i",
        "--input",
        metavar="BITS",
        default=None,
        help="raw input bits, one per half column, e.g. 1011 for 8 columns (default: random)",
    )
    parser.add_argument(
        "--no-permute",
        action="store_true",
        help="lay the complement-coded bits on the bottom row in order instead of scrambling them",
    )
    parser.add_argument(
        "--positive-weights",
        "--positive_weights",
        action="store_true",
        help="keep every weight between epsilon and 1: no inhibitory connections, before or after learning",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.001,
        help="the smallest weight allowed under --positive-weights (default: 0.001)",
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
        "--step",
        action="store_true",
        help="window mode where nothing happens until you press Space for the next epoch (instead of free-running)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="no window: run --epochs epochs and exit",
    )
    parser.add_argument(
        "--report",
        type=float,
        default=30.0,
        metavar="SECONDS",
        help="while free-running, print a progress line this often (default: 30)",
    )
    parser.add_argument(
        "--no-learn",
        action="store_true",
        help="do not teach the network (learning is on by default)",
    )
    parser.add_argument(
        "--target",
        choices=sorted(TARGETS),
        default="reversed",
        help="what the top row should show, derived from the input row (default: reversed)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.03,
        help="learning rate for --learn (default: 0.03)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.1,
        help="exploration noise: std dev of each neuron's starting potential under --learn (default: 0.1)",
    )
    parser.add_argument(
        "--eligibility",
        choices=ELIGIBILITIES,
        default="perturb",
        help="what the global reward acts on: the neuron's exploration noise (perturb) or plain Hebbian (default: perturb)",
    )
    parser.add_argument(
        "--homeostasis",
        type=float,
        default=1e-6,
        metavar="RATE",
        help="per-epoch rate at which each neuron's threshold moves toward its target firing rate (default: 1e-6; 0 = off)",
    )
    parser.add_argument(
        "--target-rate",
        type=float,
        default=0.4,
        help="firing rate homeostasis aims for, 0 to 1 (default: 0.4)",
    )
    parser.add_argument(
        "--threshold-range",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=(-5.0, 5.0),
        help="limits homeostasis may move a threshold to (default: -5 5)",
    )
    parser.add_argument(
        "--minimum-potential",
        type=float,
        default=-1.0,
        help="floor on a neuron's potential: inhibition and carried-over charge can go no lower (default: -1)",
    )
    parser.add_argument(
        "--carry-over",
        action="store_true",
        help="unfired neurons keep their potential from one epoch to the next (default: every potential is cleared)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="with --headless: how many epochs to run, printing accuracy along the way (default: 1)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="do not print a line for every neuron that fires (free-running is always quiet)",
    )
    parser.add_argument(
        "--save-weights",
        metavar="FILE",
        help="write a checkpoint of the learned weights here at every progress report and on exit",
    )
    parser.add_argument(
        "--load-weights",
        metavar="FILE",
        help="start from a checkpoint: rebuilds its mesh (size, omega, seed, permutation) and loads its weights",
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
    args.show = not args.headless
    args.fast = args.show and not args.step
    args.learn = not args.no_learn
    was_verbose = Neuron.verbose
    Neuron.verbose = not (args.quiet or args.fast)
    try:
        return _run(args)
    finally:
        Neuron.verbose = was_verbose


def _run(args: argparse.Namespace) -> int:
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
        if args.nodes is not None:
            return _run_nodes(args, width, height)
        loaded = None
        if args.load_weights:
            try:
                loaded = restore(args.load_weights)
            except (OSError, ValueError, KeyError) as exc:
                print(f"error: cannot load {args.load_weights}: {exc}", file=sys.stderr)
                return 2
            grid_from_file, data = loaded
            args.seed = data["seed"]
            args.columns, args.rows, args.omega = data["columns"], data["rows"], data["omega"]
            args.threshold = data["threshold"]
            args.minimum_potential = data.get("minimum_potential", -1.0)
            args.weight = None if data["random_weights"] else data["weight"]
            low, high = data.get("weight_range", (-1.0, 1.0))
            args.positive_weights, args.epsilon = low > 0, low
            print(
                f"loaded {args.load_weights}: {args.columns}x{args.rows}, seed {args.seed}, "
                f"{data['epoch']:,} epochs so far",
                file=sys.stderr,
            )
        seed = args.seed if args.seed is not None else random.randrange(2**31)
        settings = dict(
            columns=args.columns,
            rows=args.rows,
            weight=args.weight,
            threshold=args.threshold,
            seed=seed,
            omega=args.omega,
            permute=not args.no_permute,
            weight_range=(args.epsilon, 1.0) if args.positive_weights else (-1.0, 1.0),
            minimum_potential=args.minimum_potential,
        )
        if args.minimum_potential >= args.threshold:
            print(f"error: minimum potential ({args.minimum_potential}) must be below the threshold ({args.threshold})", file=sys.stderr)
            return 2
        if args.positive_weights and not 0 < args.epsilon < 1:
            print(f"error: epsilon must be between 0 and 1, got {args.epsilon}", file=sys.stderr)
            return 2
        if args.positive_weights:
            print(f"positive weights: every weight kept between {args.epsilon:g} and 1", file=sys.stderr)
        if args.weight is None or args.omega > 0 or args.input is None:
            print(f"seed {seed}", file=sys.stderr)

        try:
            input_bits = parse_bits(args.input) if args.input is not None else None
            if loaded:
                grid, data = loaded
                run_epoch(grid, input_bits)  # first epoch on the restored weights
            else:
                grid = main(**settings, input_bits=input_bits)
            if not args.no_permute or loaded:
                print(f"input permutation: bottom-row column i shows coded bit {grid.permutation}", file=sys.stderr)
            teacher = None
            if args.learn:
                teacher = Teacher(
                    grid,
                    target=args.target,
                    lr=args.lr,
                    sigma=args.sigma,
                    eligibility=args.eligibility,
                    seed=seed,
                    homeostasis=args.homeostasis,
                    target_rate=args.target_rate,
                    threshold_range=tuple(args.threshold_range),
                    carry_over=args.carry_over,
                )
                if loaded:
                    resume_teacher(teacher, data)
                teacher.step()  # the first epoch ran without exploration; still score and learn from it

            def save_checkpoint():
                if args.save_weights:
                    checkpoint(grid, args.save_weights, teacher)
            if args.show:
                # Free-running: the system runs and learns on its own and the window monitors it.
                # --step: nothing happens until Space is pressed.
                visualizer.show(
                    grid, width, height, fast=args.fast, teacher=teacher, report_seconds=args.report,
                    on_report=save_checkpoint,
                )
            else:
                report_every = max(1, args.epochs // 10)
                for epoch in range(2, args.epochs + 1):
                    if teacher:
                        teacher.epoch()  # --quiet drops the per-neuron lines, not the per-epoch line
                        if epoch % report_every == 0 or epoch == args.epochs:
                            print(f"epoch {epoch}: {teacher.status()}", file=sys.stderr)
                            save_checkpoint()
                    else:
                        run_epoch(grid, discharge=not args.carry_over)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        if args.omega > 0:
            print(
                f"omega {args.omega:g}: {len(grid.small_world_connections())} small-world "
                f"connections among {len(grid.connections)}",
                file=sys.stderr,
            )
        if teacher:
            print(f"after {teacher.epochs} epochs: {teacher.status()}", file=sys.stderr)
        if args.save_weights:
            save_checkpoint()
            print(f"saved weights to {args.save_weights}", file=sys.stderr)

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


def _run_nodes(args: argparse.Namespace, width: int, height: int) -> int:
    """--nodes [N]: place a Cartesian population and show it. No wiring, input or learning yet."""
    if args.nodes < 1:
        print(f"error: --nodes needs at least 1 neuron, got {args.nodes}", file=sys.stderr)
        return 2
    seed = args.seed if args.seed is not None else random.randrange(2**31)
    nodes = CartesianNodes(count=args.nodes, seed=seed, threshold=args.threshold)
    (x_min, x_max), (y_min, y_max) = nodes.bounds
    print(f"seed {seed}", file=sys.stderr)
    print(
        f"{len(nodes)} nodes placed at random in [{x_min:g}, {x_max:g}] x [{y_min:g}, {y_max:g}]; "
        "connections, input and learning are not defined for nodes yet",
        file=sys.stderr,
    )
    if args.save or args.show:
        try:
            from . import visualizer
        except ImportError:
            print("error: the visualizer needs pygame; install it with: pip install -e '.[viz]'", file=sys.stderr)
            return 2
        if args.save:
            visualizer.save_nodes(nodes, args.save, width, height)
            print(f"Saved {args.save}", file=sys.stderr)
        if args.show:
            visualizer.show_nodes(nodes, width, height)
    else:
        for neuron in nodes:
            x, y = neuron.position
            print(f"{neuron.name}: ({x:+.3f}, {y:+.3f})")
    return 0
