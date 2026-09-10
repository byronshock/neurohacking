"""Command-line entry point. Parses arguments, then hands off to the monitor."""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
import os
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

from .butter import CELL_AREA
from .cartesian import CartesianNodes
from .columns import HexColumns
from .grid import GridOfNeurons
from .inputs import CODES, DEFAULT_CODE, parse_bits
from .learning import CRITICS, ELIGIBILITIES, LATE, TARGETS, Teacher
from .monitor import main, run_epoch
from .neuron import Neuron
from .persistence import across_of, checkpoint, restore, resume_teacher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="walnutbutter",
        description=(
            "A hexagonal mesh of neurons that learns to reproduce its input on its output row. "
            "By default it opens a window, free-runs, learns, and reports accuracy until you close it."
        ),
    )
    parser.add_argument(
        "-a",
        "--across",
        type=int,
        default=None,
        help="number of hexagons across (default: 8, or twice the code length with --ecc)",
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
        const=0,
        default=None,
        help="Cartesian neurons instead of the hex grid: a --across x --rows hexagonal lattice at unit spacing, "
        "wired by distance (--receptive-field-sigma) and learning like the grid; or with N, that many neurons at "
        "random in a --across x --rows unit region, just shown",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=None,
        metavar="N",
        help="hexagonal columns in R3: the --across x --rows field of cells extruded into N layers, one neuron "
        "per cell per layer, half a unit apart in the plane and an eighth between layers. Butter within one unit "
        "horizontally always connects, at any "
        "height; the rest is --omega shortcuts. The bottom layer is the input and the top layer the output "
        "(with one layer: bottom row in, top row out, and the stack is the hex grid exactly)",
    )
    parser.add_argument(
        "--engine",
        choices=("objects", "arrays"),
        default=None,
        help="how the network is run: objects (default; every neuron and connection is an object and signals "
        "queue wave by wave) or arrays (the same network as numpy vectors and a scipy sparse matrix, much "
        "faster, needs numpy and scipy). A loaded checkpoint keeps its engine unless this is given.",
    )
    parser.add_argument(
        "--reach",
        type=float,
        default=2.0,
        metavar="UNITS",
        help="with --nodes: a neuron connects to every neuron within this many unit distances "
        "(default: 2, the two hex rings at unit density)",
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
        default=0.2,
        help="proportion of connections that are small-world shortcuts, 0 to <1 (default: 0.2)",
    )
    parser.add_argument(
        "-i",
        "--input",
        metavar="BITS",
        default=None,
        help="raw input bits, one per half place, e.g. 1011 for 8 across (default: random)",
    )
    parser.add_argument(
        "--ecc",
        nargs="?",
        const=DEFAULT_CODE,
        default=None,
        choices=sorted(CODES),
        metavar="CODE",
        help="encode the 4 data bits with an error-correcting code before complement coding: hamming74 "
        "(the default with bare --ecc; corrects single errors; 14 across) or parity64 (detects only; "
        "12 across). Sets --across to fit unless given.",
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
        "--seeds",
        type=int,
        metavar="N",
        default=None,
        help="run N seeds in parallel (consecutive from --seed, or from a random base), headless, for --epochs "
        "each; print a table sorted best first and checkpoint every run",
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
        default=1.0,
        metavar="SECONDS",
        help="while free-running, print a progress line (and record it in the checkpoint history) this often (default: 1)",
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
        "--late",
        choices=LATE,
        default="count",
        help="what a signal arriving after its target has already fired earns: count (default: the same update "
        "as one that landed, a local Hebbian term under the global reward), ignore (nothing: the "
        "node-perturbation estimator proper) or depress (the opposite update, the shape of spike-timing-dependent "
        "plasticity)",
    )
    parser.add_argument(
        "--critic",
        choices=sorted(CRITICS),
        default="row",
        help="how the reward is judged: row (fraction of output neurons matching the target), decoded "
        "(read the row as a word, error-correct it, fraction of data bits right), or decoded-exact "
        "(all data bits right or nothing). Default: row",
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
        default=0.5,
        help="firing rate homeostasis aims for, 0 to 1 (default: 0.5)",
    )
    parser.add_argument(
        "--unstick",
        type=float,
        default=1e-3,
        metavar="RATE",
        help="per-epoch rate at which a stuck output neuron's threshold moves toward --unstick-target; only "
        "output neurons firing >99%% or <1%% of the time are touched, only while stuck (default: 0.001; 0 = off)",
    )
    parser.add_argument(
        "--unstick-target",
        type=float,
        default=0.5,
        help="firing rate the output un-sticking aims for (default: 0.5)",
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
        "-v",
        "--verbose",
        action="store_true",
        help="print a line for every epoch's input and for every neuron that fires (slow; off by default)",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="accepted for compatibility: runs are quiet unless --verbose",
    )
    parser.add_argument(
        "--save-weights",
        metavar="FILE",
        help="checkpoint file, written at every progress report and on exit "
        "(default: runs/<date>-<time>-seed<seed>.json)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="do not write a checkpoint",
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
    if args.across is None:
        args.across = 2 * CODES[args.ecc].code_bits if args.ecc else 8
    args.show = not args.headless and args.seeds is None  # a seed batch is headless by definition
    args.fast = args.show and not args.step
    args.learn = not args.no_learn
    was_verbose = Neuron.verbose
    Neuron.verbose = bool(args.verbose) and not args.fast and not args.quiet
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
        if args.nodes is not None and args.nodes < 0:
            print(f"error: --nodes cannot be negative, got {args.nodes}", file=sys.stderr)
            return 2
        if args.nodes is not None and args.nodes > 0:
            return _run_nodes(args, width, height)  # a random scatter: shown, not learnable (no rows)
        if args.seeds is not None:
            return _run_seeds(args)  # grid, or the lattice with bare --nodes
        loaded = None
        if args.load_weights:
            try:
                loaded = restore(args.load_weights)
            except (OSError, ValueError, KeyError) as exc:
                print(f"error: cannot load {args.load_weights}: {exc}", file=sys.stderr)
                return 2
            grid_from_file, data = loaded
            args.seed = data["seed"]
            args.across, args.rows, args.omega = across_of(data), data["rows"], data["omega"]
            if data.get("container") == "lattice":
                args.nodes = 0
            args.ecc = _ecc_from_checkpoint(data)
            args.threshold = data["threshold"]
            args.minimum_potential = data.get("minimum_potential", -1.0)
            args.weight = None if data["random_weights"] else data["weight"]
            low, high = data.get("weight_range", (-1.0, 1.0))
            args.positive_weights, args.epsilon = low > 0, low
            print(
                f"loaded {args.load_weights}: {args.across}x{args.rows}, seed {args.seed}, "
                f"{data['epoch']:,} epochs so far",
                file=sys.stderr,
            )
        seed = args.seed if args.seed is not None else random.randrange(2**31)
        if args.save_weights is None and not args.no_save:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            args.save_weights = str(Path("runs") / f"{stamp}-seed{seed}.json")
        if args.save_weights:
            Path(args.save_weights).parent.mkdir(parents=True, exist_ok=True)
            print(f"checkpointing to {args.save_weights}", file=sys.stderr)
        settings = dict(
            across=args.across,
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
            elif args.layers is not None:
                if args.layers < 1:
                    print(f"error: --layers needs at least 1, got {args.layers}", file=sys.stderr)
                    return 2
                grid = HexColumns(layers=args.layers, **settings)
                print(f"{grid!r}: input {len(grid.input_row())} neurons, output {len(grid.output_row())}", file=sys.stderr)
            elif args.nodes is not None:
                if args.reach < 0:
                    print(f"error: --reach must not be negative, got {args.reach}", file=sys.stderr)
                    return 2
                grid = CartesianNodes(
                    across=args.across, rows=args.rows, seed=seed, threshold=args.threshold,
                    minimum_potential=args.minimum_potential, permute=not args.no_permute,
                    weight_range=settings["weight_range"],
                )
                grid.connect_within(reach=args.reach, weight=args.weight)
            else:
                grid = GridOfNeurons(**settings)
            if args.ecc:
                try:
                    grid.use_ecc(args.ecc)
                except ValueError as exc:
                    print(f"error: {exc}", file=sys.stderr)
                    return 2
                code = CODES[args.ecc]
                print(
                    f"input: {code.data_bits} data bits -> {code.name} ({code.code_bits}, {code.data_bits}) code, "
                    f"{'corrects' if code.corrects_single_errors else 'detects'} single errors -> complement code "
                    f"-> {2 * code.code_bits} across",
                    file=sys.stderr,
                )
            if isinstance(grid, CartesianNodes):
                print(
                    f"{grid!r}, wired: {len(grid.connections)} one-way connections, every pair within "
                    f"{grid.reach:g} units ({grid.mean_out_degree():.1f} per neuron)",
                    file=sys.stderr,
                )
            if not args.no_permute or loaded:
                print(f"input permutation: place i along the bottom row shows coded bit {grid.permutation}", file=sys.stderr)
            engine = args.engine or (data.get("engine", "objects") if loaded else "objects")
            if engine == "arrays":
                try:
                    from .arrays import ArrayNetwork
                except ImportError:
                    print("error: the array engine needs numpy and scipy; install them with: pip install -e '.[arrays]'", file=sys.stderr)
                    return 2
                grid = ArrayNetwork(grid)
                print(f"engine: arrays ({len(grid)} neurons, {len(grid.weight)} connections as vectors)", file=sys.stderr)
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
                    unstick=args.unstick,
                    unstick_target=args.unstick_target,
                    critic=args.critic,
                    late=args.late,
                )
                if loaded:
                    resume_teacher(teacher, data)
                teacher.epoch(input_bits, verbose=Neuron.verbose)  # the first epoch, with exploration, like every other
            else:
                run_epoch(grid, input_bits, verbose=Neuron.verbose, discharge=not args.carry_over)

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
                started = time.perf_counter()
                for epoch in range(2, args.epochs + 1):
                    if teacher:
                        teacher.epoch(verbose=Neuron.verbose)  # silent unless --verbose: printing is slower than learning
                        if epoch % report_every == 0 or epoch == args.epochs:
                            elapsed = time.perf_counter() - started
                            teacher.record(elapsed, (epoch - 1) / elapsed if elapsed else None)
                            print(f"epoch {epoch}: {teacher.status()}", file=sys.stderr)
                            save_checkpoint()
                    else:
                        run_epoch(grid, verbose=Neuron.verbose, discharge=not args.carry_over)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        mesh = getattr(grid, "mesh", grid)
        if args.omega > 0 and not isinstance(mesh, CartesianNodes):
            print(
                f"omega {args.omega:g}: {len(mesh.small_world_connections())} small-world "
                f"connections among {len(mesh.connections)}",
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
    if args.nodes < 0:
        print(f"error: --nodes cannot be negative, got {args.nodes}", file=sys.stderr)
        return 2
    seed = args.seed if args.seed is not None else random.randrange(2**31)
    common = dict(across=args.across, rows=args.rows, seed=seed, threshold=args.threshold,
                  minimum_potential=args.minimum_potential)
    print(f"seed {seed}", file=sys.stderr)
    if args.nodes == 0:
        nodes = CartesianNodes(layout="hex", **common)
        print(f"{nodes!r}", file=sys.stderr)
    else:
        nodes = CartesianNodes(layout="random", count=args.nodes, **common)
        print(f"{nodes!r} ({len(nodes) * CELL_AREA / (nodes.width * nodes.height):.2f} per unit cell)", file=sys.stderr)
    if args.reach < 0:
        print(f"error: --reach must not be negative, got {args.reach}", file=sys.stderr)
        return 2
    made = nodes.connect_within(reach=args.reach, weight=args.weight)
    print(
        f"wired: {made} one-way connections, {nodes.mean_out_degree():.1f} outgoing per neuron, every pair within "
        f"{args.reach:g} units; input and learning are not defined for a scatter (it has no rows)",
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


def _seed_worker(job: dict) -> dict:
    """One seed's headless run, in its own process. Returns a summary row."""
    Neuron.verbose = False
    seed, epochs = job["seed"], job["epochs"]
    if job.get("lattice"):
        settings = job["settings"]
        grid = CartesianNodes(
            across=settings["across"], rows=settings["rows"], seed=seed, threshold=settings["threshold"],
            minimum_potential=settings["minimum_potential"], permute=settings["permute"],
            weight_range=settings["weight_range"],
        )
        grid.connect_within(reach=job["lattice"]["reach"], weight=settings["weight"])
    elif not job.get("layers"):
        grid = GridOfNeurons(**job["settings"], seed=seed)
    if job.get("layers"):
        grid = HexColumns(layers=job["layers"], **job["settings"], seed=seed)
    if job.get("ecc"):
        grid.use_ecc(job["ecc"])
    if job.get("engine") == "arrays":
        from .arrays import ArrayNetwork
        grid = ArrayNetwork(grid)
    teacher = Teacher(grid, seed=seed, **job["teacher"])
    report_every = max(1, epochs // 10)
    started = time.perf_counter()
    rewards = [teacher.epoch(verbose=False)]
    for epoch in range(2, epochs + 1):
        rewards.append(teacher.epoch(verbose=False))
        if epoch % report_every == 0 or epoch == epochs:
            elapsed = time.perf_counter() - started
            teacher.record(elapsed, (epoch - 1) / elapsed if elapsed else None)
    if job["save"]:
        checkpoint(grid, job["save"], teacher)
    tail = rewards[-max(1, len(rewards) // 10):]  # the last tenth of the run
    return {
        "seed": seed,
        "to_date": teacher.accuracy_to_date,
        "recent": sum(tail) / len(tail),
        "epochs": teacher.epochs,
        "save": job["save"],
    }


def _run_seeds(args: argparse.Namespace) -> int:
    """--seeds N: N headless runs in parallel, one table at the end, a checkpoint per run."""
    if args.seeds < 1:
        print(f"error: --seeds needs at least 1, got {args.seeds}", file=sys.stderr)
        return 2
    if args.epochs < 2:
        print("error: --seeds needs --epochs of at least 2", file=sys.stderr)
        return 2
    base = args.seed if args.seed is not None else random.randrange(2**31 - args.seeds)
    seeds = list(range(base, base + args.seeds))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    settings = dict(
        across=args.across,
        rows=args.rows,
        weight=args.weight,
        threshold=args.threshold,
        omega=args.omega,
        permute=not args.no_permute,
        weight_range=(args.epsilon, 1.0) if args.positive_weights else (-1.0, 1.0),
        minimum_potential=args.minimum_potential,
    )
    teacher_kwargs = dict(
        target=args.target,
        lr=args.lr,
        sigma=args.sigma,
        eligibility=args.eligibility,
        homeostasis=args.homeostasis,
        target_rate=args.target_rate,
        threshold_range=tuple(args.threshold_range),
        carry_over=args.carry_over,
        unstick=args.unstick,
        unstick_target=args.unstick_target,
        critic=args.critic,
        late=args.late,
    )
    if args.no_learn:
        print("error: --seeds is for comparing learning runs; drop --no-learn", file=sys.stderr)
        return 2
    jobs = []
    for seed in seeds:
        save = None
        if not args.no_save:
            save = args.save_weights or str(Path("runs") / f"{stamp}-seed{seed}.json")
            if args.save_weights and args.seeds > 1:
                save = str(Path(args.save_weights).with_suffix("")) + f"-seed{seed}.json"
            Path(save).parent.mkdir(parents=True, exist_ok=True)
        lattice = {"reach": args.reach} if args.nodes is not None else None
        jobs.append({"seed": seed, "epochs": args.epochs, "settings": settings, "teacher": teacher_kwargs,
                     "save": save, "lattice": lattice, "ecc": args.ecc, "engine": args.engine or "objects",
                     "layers": args.layers})
    if args.engine == "arrays":
        try:
            import numpy, scipy  # noqa: F401
        except ImportError:
            print("error: the array engine needs numpy and scipy; install them with: pip install -e '.[arrays]'", file=sys.stderr)
            return 2
    workers = max(1, min(args.seeds, (os.cpu_count() or 2) - 1))
    wiring = (f"lattice, reach {args.reach:g}" if args.nodes is not None
              else f"{args.layers} layers of hexagonal columns, omega {args.omega:g}" if args.layers
              else f"hex grid, omega {args.omega:g}")
    print(
        f"{args.seeds} seeds from {base} on {workers} cores, {args.epochs:,} epochs each, "
        f"{args.across}x{args.rows} {wiring}",
        file=sys.stderr,
    )
    started = time.perf_counter()
    with Pool(workers) as pool:
        rows = pool.map(_seed_worker, jobs)
    rows.sort(key=lambda r: (r["recent"], r["to_date"]), reverse=True)
    print(f"{'seed':>11}  {'last tenth':>10}  {'to date':>8}  checkpoint")
    for r in rows:
        print(f"{r['seed']:>11}  {r['recent']:>10.1%}  {r['to_date']:>8.1%}  {r['save'] or '-'}")
    best = rows[0]
    print(
        f"best seed {best['seed']}: {best['recent']:.1%} over its last tenth "
        f"({time.perf_counter() - started:.0f}s wall)",
        file=sys.stderr,
    )
    return 0


def _ecc_from_checkpoint(data: dict) -> str | None:
    """The code a checkpoint used: a name, or the (6, 4) code for files written when ecc was a flag."""
    value = data.get("ecc")
    if value is True:
        return "parity64"
    return value or None
