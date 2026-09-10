"""Sweep tau on the default 8x10 mesh, array engine, 15 paired seeds per tau, 1M epochs each.

Phase 1: a log-spaced grid from 0.5 ms to infinity, baseline 5 included.
Phase 2: if the best finite tau is interior, refine with a 1-D Nelder-Mead on log(tau)
         (the simplex is an interval), same 15 seeds, at most 6 more evaluations.
Phase 3: confirm the best against the baseline at 3M epochs.
Results go to results.jsonl (one line per run) and summary.json; progress to sweep.log.
"""
import json, math, statistics, sys, time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
HERE = Path(__file__).parent
SEEDS = list(range(1, 16))
EPOCHS = 1_000_000
CONFIRM_EPOCHS = 3_000_000
GRID = [0.5, 1.0, 2.0, 3.5, 5.0, 7.0, 10.0, 15.0, 25.0, 50.0, math.inf]
WORKERS = 15

def log(msg):
    with open(HERE / "sweep.log", "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")

def run(job):
    tau, seed, epochs = job
    from walnutbutter.arrays import ArrayNetwork
    from walnutbutter.grid import GridOfNeurons
    from walnutbutter.learning import Teacher
    from walnutbutter.neuron import Neuron
    Neuron.verbose = False
    Neuron.tau = tau
    t = Teacher(ArrayNetwork(GridOfNeurons(seed=seed, weight=None)), seed=seed)
    tail_from = int(epochs * 0.9)
    tail_sum = 0.0
    for i in range(epochs):
        r = t.epoch(verbose=False)
        if i >= tail_from:
            tail_sum += r
    return {"tau": tau, "seed": seed, "epochs": epochs, "to_date": t.accuracy_to_date, "recent": tail_sum / (epochs - tail_from)}

def evaluate(pool, tau, epochs=EPOCHS, cache={}):
    key = (tau, epochs)
    if key in cache:
        return cache[key]
    t0 = time.time()
    rows = pool.map(run, [(tau, s, epochs) for s in SEEDS])
    with open(HERE / "results.jsonl", "a") as f:
        for r in rows:
            f.write(json.dumps({**r, "tau": "inf" if r["tau"] == math.inf else r["tau"]}) + "\n")
    recent = statistics.median(r["recent"] for r in rows)
    to_date = statistics.median(r["to_date"] for r in rows)
    cache[key] = {"tau": tau, "epochs": epochs, "median_recent": recent, "median_to_date": to_date,
                  "mean_recent": statistics.mean(r["recent"] for r in rows), "rows": rows}
    log(f"tau={tau:g} epochs={epochs}: median last tenth {recent:.4f}, median to date {to_date:.4f}, mean last tenth {cache[key]['mean_recent']:.4f} ({time.time()-t0:.0f} s)")
    return cache[key]

def score(result):
    return result["median_recent"]

def nelder_mead_1d(pool, x_lo, x_hi, max_evals=6):
    """Nelder-Mead on x = log(tau): the simplex is two points; reflect, expand, contract, shrink."""
    f = lambda x: score(evaluate(pool, math.exp(x)))
    pts = sorted([(f(x_lo), x_lo), (f(x_hi), x_hi)], reverse=True)  # best first (maximising)
    evals = 0
    while evals < max_evals:
        (fb, xb), (fw, xw) = pts
        xr = xb + (xb - xw)  # reflect the worst through the best
        fr = f(xr); evals += 1
        if fr > fb:
            xe = xb + 2 * (xb - xw)
            fe = f(xe); evals += 1
            pts = sorted([(fb, xb), (fe, xe) if fe > fr else (fr, xr)], reverse=True)
        else:
            xc = xb + 0.5 * (xw - xb)  # contract toward the best
            fc = f(xc); evals += 1
            pts = sorted([(fb, xb), (fc, xc) if fc > fw else (fw, xw)], reverse=True)
        if abs(pts[0][1] - pts[1][1]) < 0.15:  # within about 16% in tau: the noise floor of 15 seeds
            break
    return pts

if __name__ == "__main__":
    log(f"PHASE 1 start: taus {GRID}, {len(SEEDS)} seeds x {EPOCHS:,} epochs each, {WORKERS} workers")
    summary = {"grid": [], "refine": [], "confirm": []}
    with Pool(WORKERS) as pool:
        for tau in GRID:
            r = evaluate(pool, tau)
            summary["grid"].append({k: v for k, v in r.items() if k != "rows"})
        finite = [g for g in summary["grid"] if g["tau"] != math.inf]
        best = max(finite, key=score)
        i = [g["tau"] for g in finite].index(best["tau"])
        log(f"PHASE 1 done: best finite tau {best['tau']:g} (median last tenth {best['median_recent']:.4f}); baseline 5 -> {next(g for g in finite if g['tau']==5.0)['median_recent']:.4f}")
        if 0 < i < len(finite) - 1:
            lo, hi = finite[i - 1]["tau"], finite[i + 1]["tau"]
            log(f"PHASE 2 start: Nelder-Mead on log(tau) between {lo:g} and {hi:g}")
            pts = nelder_mead_1d(pool, math.log(lo) + (math.log(best['tau']) - math.log(lo)) / 2, math.log(hi) - (math.log(hi) - math.log(best['tau'])) / 2)
            summary["refine"] = [{"tau": math.exp(x), "median_recent": fx} for fx, x in pts]
            log(f"PHASE 2 done: simplex at tau {[round(math.exp(x), 3) for _, x in pts]} with {[round(fx, 4) for fx, _ in pts]}")
        else:
            log("PHASE 2 skipped: the best tau sits at the edge of the grid")
        cache_all = [dict(tau=r["tau"], score=score(r)) for r in [evaluate(pool, t) for t in set([g["tau"] for g in finite] + [x["tau"] for x in summary["refine"]])]]
        champion = max(cache_all, key=lambda r: r["score"])["tau"]
        log(f"PHASE 3 start: confirm tau {champion:g} against the baseline 5 at {CONFIRM_EPOCHS:,} epochs")
        for tau in ([champion, 5.0] if champion != 5.0 else [5.0]):
            r = evaluate(pool, tau, CONFIRM_EPOCHS)
            summary["confirm"].append({k: v for k, v in r.items() if k != "rows"})
    summary["champion"] = champion
    (HERE / "summary.json").write_text(json.dumps(summary, default=str, indent=1))
    log("DONE")
