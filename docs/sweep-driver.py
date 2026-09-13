#!/usr/bin/env python3
"""Sweep any of the dopamine and clock knobs on sustain_inputs, a million epochs per arm, everything recorded.

    .venv/bin/python docs/sweep-driver.py --name hops-sweep --hops 2 2.5 3 3.5 4 4.5 5 --alpha 5 --theta 5.5
    .venv/bin/python docs/sweep-driver.py --name release-sweep-2 --alpha 5 10 20 --theta 11 25 50 100
    .venv/bin/python docs/sweep-driver.py --name hops-sweep ... --summary     # summarise what is on disk, run nothing

The arms are the product of every list given (a knob given once is fixed
for the sweep). One process per arm on the array engine, seed 1 (paired),
the sustain_inputs problem. Everything a run normally records is kept under
runs/<name>/: the checkpoint (weights, spikes, dopamine state, the Teacher's
history), the per-epoch trace CSV (epoch, time, dopamine, expected, score),
and the run's log. When every arm is done the driver writes docs/<name>.md
(a table per arm) and two figures, docs/<name>-expected.png and
docs/<name>-score.png, so the sweep can be read against either score or
dopamine_expected. Arms already complete are skipped; partial ones restart.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
KNOBS = {  # knob -> (command-line flag, label)
    "alpha": ("--release-alpha", "alpha"),
    "theta": ("--release-theta", "theta (ms)"),
    "hops": ("--refractory-hops", "refractory hops"),
    "refractory": ("--refractory", "refractory (ms)"),
    "dopamine_tau": ("--dopamine-tau", "pool tau (ms)"),
    "expectation_tau": ("--expectation-tau", "expectation tau (ms)"),
    "interval": ("--interval", "interval (ms)"),
    "lr": ("--lr", "lr"),
    "sigma": ("--sigma", "sigma"),
    "punish_gain": ("--punish-gain", "punish gain"),
    "decay": ("--weight-decay", "weight decay"),
    "expectation_start": ("--expectation-start", "expectation start"),
    "bored_after": ("--bored-after", "bored after (ms)"),
    "tau": ("--tau", "tau (ms)"),
    "threshold": ("--threshold", "threshold"),
    "seed": ("--seed", "seed"),
}


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", required=True, help="the sweep's name: runs/<name>/ and docs/<name>.*")
    for knob in KNOBS:
        if knob != "seed":
            parser.add_argument(f"--{knob.replace('_', '-')}", type=float, nargs="+", default=None, metavar="V")
    parser.add_argument("--epochs", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, nargs="+", default=[1], help="one or more seeds; several make a seed axis")
    parser.add_argument("--engine", default="arrays")
    parser.add_argument("--problem", default="sustain_inputs", help="the problem every arm runs (default: sustain_inputs)")
    parser.add_argument("--order", default=None, help="release-first or update-first (fixed for the sweep)")
    parser.add_argument("--no-punish", action="store_true", help="pass --no-punish to every arm")
    parser.add_argument("--summary", action="store_true", help="summarise what is on disk; run nothing")
    return parser.parse_args()


def grid(args) -> tuple[list[str], list[dict]]:
    """The swept knobs (more than one value), and every arm as {knob: value} over all knobs given."""
    given = {knob: getattr(args, knob) for knob in KNOBS if getattr(args, knob) is not None}
    given["seed"] = list(args.seed)  # always an arm's last knob, so names read ...-seedN as before
    swept = [knob for knob, values in given.items() if len(values) > 1]
    arms = [dict(zip(given, values)) for values in itertools.product(*given.values())]
    return swept, arms


def arm_name(arm: dict, args) -> str:
    return "-".join(f"{knob}{value:g}" for knob, value in arm.items())


def run_arm(job: tuple) -> dict:
    arm, args = job
    out = ROOT / "runs" / args.name
    name = arm_name(arm, args)
    save, log = out / f"{name}.json", out / f"{name}.log"
    if save.exists():
        try:
            if json.loads(save.read_text())["epoch"] >= args.epochs:
                return {"arm": name, "skipped": True}
        except (ValueError, KeyError):
            pass
    for stale in (save, save.with_suffix(".csv")):
        if stale.exists():
            stale.unlink()  # an incomplete arm starts over: the trace is appended, so it must not carry old rows
    command = [PYTHON, "-m", "walnutbutter", "--problem", args.problem, "--headless", "--engine", args.engine,
               "--epochs", str(args.epochs), "--save-weights", str(save), "--report", "60"]
    for knob, value in arm.items():
        command += [KNOBS[knob][0], f"{value:g}"]
    if args.order:
        command += ["--order", args.order]
    if args.no_punish:
        command += ["--no-punish"]
    started = time.perf_counter()
    with open(log, "w") as handle:
        code = subprocess.call(command, stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT)
    return {"arm": name, "code": code, "seconds": round(time.perf_counter() - started)}


def read_trace(path: Path) -> dict[str, list[float]]:
    columns: dict[str, list[float]] = {"epoch": [], "time_ms": [], "dopamine": [], "expected": [], "score": []}
    with open(path) as handle:
        for row in csv.DictReader(handle):
            for key in columns:
                columns[key].append(float(row[key]) if row[key] != "" else float("nan"))
    return columns


def summarise(args) -> None:
    swept, arms = grid(args)
    out = ROOT / "runs" / args.name
    fixed = {knob: values[0] for knob, values in ((k, getattr(args, k)) for k in KNOBS) if values is not None and len(values) == 1}
    rows, traces = [], {}
    size = "8x10"  # replaced by the first checkpoint's own size below
    for arm in arms:
        name = arm_name(arm, args)
        save, trace = out / f"{name}.json", out / f"{name}.csv"
        label = ", ".join(f"{KNOBS[k][1].split(' (')[0]} {arm[k]:g}" for k in swept) or name
        if not (save.exists() and trace.exists()):
            rows.append({"arm": arm, "label": label, "missing": True})
            continue
        data = json.loads(save.read_text())
        size = f"{data['across']}x{data['rows']}"
        t = read_trace(trace)
        n = len(t["score"])
        tenth = max(1, n // 10)
        weights, pool = data["weights"], data["dopamine"]
        rows.append({
            "arm": arm, "label": label, "epochs": data["epoch"],
            "score_to_date": sum(t["score"]) / n, "score_last_tenth": sum(t["score"][-tenth:]) / tenth, "score_max": max(t["score"]),
            "expected_final": t["expected"][-1], "expected_last_tenth": sum(t["expected"][-tenth:]) / tenth,
            "expected_peak": max(t["expected"]), "dopamine_last_tenth": sum(t["dopamine"][-tenth:]) / tenth,
            "releases": pool["releases"], "total_released": pool["total"],
            "spikes": sum(data["spikes"]), "neurons_spiked": sum(1 for s in data["spikes"] if s),
            "weights_plus": sum(1 for w in weights if w >= 0.999), "weights_minus": sum(1 for w in weights if w <= -0.999),
        })
        traces[label] = t
    head = " | ".join(KNOBS[k][1] for k in swept)
    lines = [
        f"# Sweep {args.name} (September 12, 2026)", "",
        f"{args.problem}, {size} hex grid, {args.engine} engine, seed{'s' if len(args.seed) > 1 else ''} {', '.join(str(s) for s in args.seed)}, {args.epochs:,} epochs per arm, 20 ms epochs. "
        + ("Fixed: " + ", ".join(f"{KNOBS[k][1]} {v:g}" for k, v in fixed.items()) + ". " if fixed else "")
        + f"Swept: {', '.join(KNOBS[k][1] for k in swept)}. Everything else at the defaults in constants.py.",
        f"Driver: `sweep-driver.py`; figures: `{args.name}-expected.png`, `{args.name}-score.png`; every arm's checkpoint, "
        f"per-epoch trace (CSV) and log are under `runs/{args.name}/` (not in git).", "",
        f"| {head} | score to date | score last tenth | score max | expected final | expected last tenth | expected peak | "
        "dopamine last tenth | releases | total released | spikes | neurons spiked | weights at +1 | at -1 |",
        "|" + "---|" * (len(swept) + 14),
    ]
    for r in rows:
        cells = [f"{r['arm'][k]:g}" for k in swept]
        if r.get("missing"):
            lines.append("| " + " | ".join(cells) + " | (missing) |" + " |" * 13)
            continue
        lines.append("| " + " | ".join(cells) + f" | {r['score_to_date']:.3f} | {r['score_last_tenth']:.3f} | {r['score_max']:.3f} | "
                     f"{r['expected_final']:.4g} | {r['expected_last_tenth']:.4g} | {r['expected_peak']:.4g} | {r['dopamine_last_tenth']:.4g} | "
                     f"{r['releases']:,} | {r['total_released']:.4g} | {r['spikes']:,} | {r['neurons_spiked']} | {r['weights_plus']} | {r['weights_minus']} |")
    if "seed" in swept and len(swept) > 1:  # the seed-averaged view, per setting of the other swept knobs
        others = [k for k in swept if k != "seed"]
        groups: dict = {}
        for r in rows:
            if r.get("missing"):
                continue
            groups.setdefault(tuple(r["arm"][k] for k in others), []).append(r)
        lines += ["", f"Averaged over the {len(args.seed)} seeds (mean, then the range across seeds):", "",
                  "| " + " | ".join(KNOBS[k][1] for k in others) + " | seeds | score last tenth | range | score to date | range | spikes | pinned at +1 |",
                  "|" + "---|" * (len(others) + 7)]
        for key, group in groups.items():
            last = [r["score_last_tenth"] for r in group]
            date = [r["score_to_date"] for r in group]
            lines.append("| " + " | ".join(f"{v:g}" for v in key) + f" | {len(group)} | {sum(last) / len(last):.3f} | {min(last):.3f} to {max(last):.3f} | "
                         f"{sum(date) / len(date):.3f} | {min(date):.3f} to {max(date):.3f} | {sum(r['spikes'] for r in group) / len(group):,.0f} | "
                         f"{sum(r['weights_plus'] for r in group) / len(group):.1f} |")
    (ROOT / "docs" / f"{args.name}.md").write_text("\n".join(lines) + "\n")
    if traces:
        plot(traces, [r["label"] for r in rows], swept, arms, args, size)
    print("\n".join(lines))


def plot(traces: dict, labels: list[str], swept: list[str], arms: list[dict], args, size: str = "8x10") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#1a1a19", "#5f5e58", "#8a897f", "#e6e5df"
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    if len(swept) == 2:  # rows by the first swept knob, columns by the second; more than two: a wrapped list
        rows_of = sorted({arm[swept[0]] for arm in arms})
        cols_of = sorted({arm[swept[1]] for arm in arms})
        n_rows, n_cols = len(rows_of), len(cols_of)
        position = {labels[i]: (rows_of.index(arm[swept[0]]), cols_of.index(arm[swept[1]])) for i, arm in enumerate(arms)}
    else:
        n_cols = min(4, len(labels))
        n_rows = (len(labels) + n_cols - 1) // n_cols
        position = {label: divmod(i, n_cols) for i, label in enumerate(labels)}
    for which, suffix, series, colours, ylabel in (
        ("expected", "expected", ("expected", "dopamine"), (ORANGE, BLUE), "dopamine units"),
        ("score", "score", ("score",), (BLUE,), "score"),
    ):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols + 1, 2.4 * n_rows + 1), dpi=130, sharex=True,
                                 sharey=(which == "score"), squeeze=False)
        fig.patch.set_facecolor(SURFACE)
        for ax in axes.flat:
            ax.set_facecolor(SURFACE)
            ax.axis("off")
        for label in labels:
            i, j = position[label]
            ax = axes[i][j]
            ax.axis("on")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.tick_params(colors=INK2, labelsize=7, length=0)
            ax.grid(axis="y", color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            ax.set_title(label, loc="left", color=INK, fontsize=8)
            t = traces.get(label)
            if t is None:
                ax.text(0.5, 0.5, "missing", transform=ax.transAxes, color=MUTED, ha="center")
                continue
            every = max(1, len(t["epoch"]) // 2000)
            minutes = [m / 60000 for m in t["time_ms"][::every]]
            for key, colour in zip(series, colours):
                values = t[key]
                if key == "score":  # a moving average over 500 epochs, taken before thinning, so the line is readable
                    window = 500
                    running, smoothed = 0.0, []
                    for k, v in enumerate(values):
                        running += v - (values[k - window] if k >= window else 0.0)
                        smoothed.append(running / min(k + 1, window))
                    values = smoothed
                ax.plot(minutes, values[::every], color=colour, linewidth=1.2, label=key)
            if which == "score":
                ax.set_ylim(0, 1)
            if i == n_rows - 1:
                ax.set_xlabel("clock, minutes", color=INK2, fontsize=8)
            if j == 0:
                ax.set_ylabel(ylabel, color=INK2, fontsize=8)
        if which == "expected":
            axes[0][0].legend(loc="upper right", frameon=False, fontsize=7, labelcolor=INK2)
        fig.suptitle(f"{args.name}: {args.problem} {size}, seed{'s' if len(args.seed) > 1 else ''} {', '.join(str(s) for s in args.seed)}, {args.epochs:,} epochs per arm: "
                     f"{'dopamine_expected against the pool' if which == 'expected' else 'score, 500-epoch moving average'}",
                     x=0.01, ha="left", color=INK, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(ROOT / "docs" / f"{args.name}-{suffix}.png", facecolor=SURFACE)
        plt.close(fig)


def main() -> int:
    venv = ROOT / ".venv"
    if (venv / "bin" / "python").exists() and Path(sys.prefix).resolve() != venv.resolve():
        python = str(venv / "bin" / "python")
        os.execv(python, [python, __file__] + sys.argv[1:])  # run inside the project's venv (numpy, scipy, matplotlib)
    args = parse()
    out = ROOT / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    swept, arms = grid(args)
    if not args.summary:
        workers = min(len(arms), max(1, os.cpu_count() - 1))
        print(f"{len(arms)} arms on {workers} workers, {args.epochs:,} epochs each, sweeping {swept} -> {out}", flush=True)
        started = time.perf_counter()
        with Pool(workers) as pool:
            for result in pool.imap_unordered(run_arm, [(arm, args) for arm in arms]):
                print(f"[{time.perf_counter() - started:6.0f}s] {result}", flush=True)
    summarise(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
