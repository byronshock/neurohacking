"""Sweep the dopamine release gamma: alpha in {5, 10, 20} x theta in {11, 25, 50, 100} (the second sweep; the first, alpha 1.25-5 x theta 0.25-11, is release-sweep-1), a million epochs each.

Byron, September 12, 2026. One arm per process on the array engine, the
sustain_inputs problem, seed 1 (paired across arms). Everything a run normally
records is kept under runs/release-sweep/: the checkpoint (weights, spikes,
dopamine state, the Teacher's history), the per-epoch trace CSV (epoch, time,
dopamine, expected, score), and the run's log. When every arm is done the
driver writes docs/release-sweep.md (a table per arm) and two figures,
docs/release-sweep-expected.png and docs/release-sweep-score.png, so the
sweep can be read against either score or dopamine_expected.

    .venv/bin/python docs/release-sweep-driver.py            # run (skips arms already complete), then summarise
    .venv/bin/python docs/release-sweep-driver.py --summary  # summarise what is on disk, run nothing
"""

from __future__ import annotations

import csv
import itertools
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ALPHAS = (5.0, 10.0, 20.0)
THETAS = (11.0, 25.0, 50.0, 100.0)
EPOCHS = 1_000_000
SEED = 1
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "release-sweep"
PYTHON = sys.executable


def arm_name(alpha: float, theta: float) -> str:
    return f"a{alpha:g}-t{theta:g}-seed{SEED}"


def run_arm(arm: tuple[float, float]) -> dict:
    alpha, theta = arm
    name = arm_name(alpha, theta)
    save, log = OUT / f"{name}.json", OUT / f"{name}.log"
    if save.exists():
        try:
            if json.loads(save.read_text())["epoch"] >= EPOCHS:
                return {"arm": name, "skipped": True}
        except (ValueError, KeyError):
            pass
    for stale in (save, save.with_suffix(".csv")):
        if stale.exists():
            stale.unlink()  # an incomplete arm starts over: the trace is appended, so it must not carry old rows
    command = [
        PYTHON, "-m", "walnutbutter", "--problem", "sustain_inputs", "--headless", "--engine", "arrays",
        "--epochs", str(EPOCHS), "--seed", str(SEED), "--release-alpha", str(alpha), "--release-theta", str(theta),
        "--save-weights", str(save), "--report", "60",
    ]
    started = time.perf_counter()
    with open(log, "w") as handle:
        code = subprocess.call(command, stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT)
    return {"arm": name, "code": code, "seconds": round(time.perf_counter() - started)}


def read_trace(path: Path, every: int = 1) -> dict[str, list[float]]:
    columns: dict[str, list[float]] = {"epoch": [], "time_ms": [], "dopamine": [], "expected": [], "score": []}
    with open(path) as handle:
        for i, row in enumerate(csv.DictReader(handle)):
            if i % every:
                continue
            for key in columns:
                columns[key].append(float(row[key]) if row[key] != "" else float("nan"))
    return columns


def summarise() -> None:
    rows = []
    traces = {}
    for alpha, theta in itertools.product(ALPHAS, THETAS):
        name = arm_name(alpha, theta)
        save, trace = OUT / f"{name}.json", OUT / f"{name}.csv"
        if not (save.exists() and trace.exists()):
            rows.append({"alpha": alpha, "theta": theta, "missing": True})
            continue
        data = json.loads(save.read_text())
        t = read_trace(trace)
        n = len(t["score"])
        tenth = max(1, n // 10)
        weights = data["weights"]
        pool = data["dopamine"]
        rows.append({
            "alpha": alpha, "theta": theta, "epochs": data["epoch"],
            "score_to_date": sum(t["score"]) / n,
            "score_last_tenth": sum(t["score"][-tenth:]) / tenth,
            "score_max": max(t["score"]),
            "expected_final": t["expected"][-1],
            "expected_last_tenth": sum(t["expected"][-tenth:]) / tenth,
            "expected_peak": max(t["expected"]),
            "dopamine_last_tenth": sum(t["dopamine"][-tenth:]) / tenth,
            "releases": pool["releases"], "total_released": pool["total"],
            "spikes": sum(data["spikes"]), "neurons_spiked": sum(1 for s in data["spikes"] if s),
            "weights_plus": sum(1 for w in weights if w >= 0.999), "weights_minus": sum(1 for w in weights if w <= -0.999),
        })
        traces[(alpha, theta)] = t
    lines = [
        "# Dopamine release sweep 2: long theta (September 12, 2026)", "",
        f"sustain_inputs, 8x10 hex grid, array engine, seed {SEED} (paired), {EPOCHS:,} epochs per arm, 20 ms epochs.",
        "Release = gamma density of the refire delay past the refractory period; pool decay 20 ms; expectation window 10 min from 0;",
        "release-first; lr 0.03; sigma 0.1. Driver: `release-sweep-driver.py`; figures: `release-sweep-expected.png`, `release-sweep-score.png`;",
        "every arm's checkpoint, per-epoch trace (CSV) and log are under `runs/release-sweep/` (not in git).", "",
        "| alpha | theta (ms) | score to date | score last tenth | score max | expected final | expected last tenth | expected peak | dopamine last tenth | releases | total released | spikes | neurons spiked | weights at +1 | at -1 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("missing"):
            lines.append(f"| {r['alpha']:g} | {r['theta']:g} | (missing) | | | | | | | | | | | | |")
            continue
        lines.append(
            f"| {r['alpha']:g} | {r['theta']:g} | {r['score_to_date']:.3f} | {r['score_last_tenth']:.3f} | {r['score_max']:.3f} | "
            f"{r['expected_final']:.4g} | {r['expected_last_tenth']:.4g} | {r['expected_peak']:.4g} | {r['dopamine_last_tenth']:.4g} | "
            f"{r['releases']:,} | {r['total_released']:.4g} | {r['spikes']:,} | {r['neurons_spiked']} | {r['weights_plus']} | {r['weights_minus']} |"
        )
    (ROOT / "docs" / "release-sweep.md").write_text("\n".join(lines) + "\n")
    if traces:
        plot(traces)
    print("\n".join(lines))


def plot(traces: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#1a1a19", "#5f5e58", "#8a897f", "#e6e5df"
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    for which, filename, series, colour, ylabel in (
        ("expected", "release-sweep-expected.png", ("expected", "dopamine"), (ORANGE, BLUE), "dopamine units"),
        ("score", "release-sweep-score.png", ("score",), (BLUE,), "score"),
    ):
        fig, axes = plt.subplots(len(ALPHAS), len(THETAS), figsize=(16, 8), dpi=130, sharex=True, sharey=(which == "score"))
        fig.patch.set_facecolor(SURFACE)
        for i, alpha in enumerate(ALPHAS):
            for j, theta in enumerate(THETAS):
                ax = axes[i][j]
                ax.set_facecolor(SURFACE)
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
                for side in ("left", "bottom"):
                    ax.spines[side].set_color(GRID)
                ax.tick_params(colors=INK2, labelsize=7, length=0)
                ax.grid(axis="y", color=GRID, linewidth=0.6)
                ax.set_axisbelow(True)
                ax.set_title(f"alpha {alpha:g}, theta {theta:g} ms", loc="left", color=INK, fontsize=8)
                t = traces.get((alpha, theta))
                if t is None:
                    ax.text(0.5, 0.5, "missing", transform=ax.transAxes, color=MUTED, ha="center")
                    continue
                every = max(1, len(t["epoch"]) // 2000)
                minutes = [m / 60000 for m in t["time_ms"][::every]]
                for key, c in zip(series, colour):
                    values = t[key][::every]
                    if key == "score":  # a moving average over about 500 epochs, so the line is readable
                        window = max(1, 500 // every)
                        values = [sum(values[max(0, k - window + 1):k + 1]) / len(values[max(0, k - window + 1):k + 1]) for k in range(len(values))]
                    ax.plot(minutes, values, color=c, linewidth=1.2, label=key)
                if which == "score":
                    ax.set_ylim(0, 1)
                if i == len(ALPHAS) - 1:
                    ax.set_xlabel("clock, minutes", color=INK2, fontsize=8)
                if j == 0:
                    ax.set_ylabel(ylabel, color=INK2, fontsize=8)
        if which == "expected":
            axes[0][0].legend(loc="upper right", frameon=False, fontsize=7, labelcolor=INK2)
        fig.suptitle(f"Release gamma sweep, sustain_inputs 8x10 seed {SEED}, {EPOCHS:,} epochs per arm: "
                     f"{'dopamine_expected against the pool' if which == 'expected' else 'score, 500-epoch moving average'}",
                     x=0.01, ha="left", color=INK, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(ROOT / "docs" / filename, facecolor=SURFACE)
        plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if "--summary" not in sys.argv:
        arms = list(itertools.product(ALPHAS, THETAS))
        workers = min(len(arms), max(1, os.cpu_count() - 1))
        print(f"{len(arms)} arms on {workers} workers, {EPOCHS:,} epochs each -> {OUT}", flush=True)
        started = time.perf_counter()
        with Pool(workers) as pool:
            for result in pool.imap_unordered(run_arm, arms):
                print(f"[{time.perf_counter() - started:6.0f}s] {result}", flush=True)
    summarise()
    return 0


if __name__ == "__main__":
    sys.exit(main())
