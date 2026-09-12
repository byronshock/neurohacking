"""The Teacher: scores a trained problem's output every epoch, and runs the reinforce rule when asked.

Two rules exist (AUTHORITY.md §6). The default, **dopamine**, lives in
dopamine.py and runs inside the schedule as neurons refire; under it the
Teacher only scores and reports. The **reinforce** rule of the pre-alpha,
factored out behind `rule="reinforce"`, is the rest of this module.

The top row is the network's output. For each epoch a target pattern is
derived from the input pattern (by default its reverse) and the reward is
the fraction of output neurons that match it.

Nothing is traced back through the network. Instead each epoch:

1. **Explore.** Every neuron starts the epoch with a small random potential
   (Gaussian, standard deviation `sigma`), so the same input produces
   slightly different cascades from one epoch to the next.
2. **Run** the epoch and score it. The **advantage** is the reward minus a
   running average of recent rewards: how much better or worse than usual.
3. **Reinforce.** Every connection that carried a signal (its source fired)
   into a neuron that was not a forced input is moved by
   `lr * advantage * eligibility`, where the eligibility is the target's
   exploration noise (normalised).

**Late signals.** A signal that arrives after its target has already fired
is dropped on delivery (Neuron.receive) and has no effect on the epoch, yet
by default its connection is still updated: the rule is local, pre fired
and post fired in the same epoch, and the global reward says whether that
coincidence was good. Strictly that is a reward-modulated Hebbian term
riding on the perturbation estimator, a bias with respect to the reward
gradient, but it is the biological shape of the rule (local eligibility,
global signal) and on the 8x10 reversed task it learned faster on every
seed tried (100k epochs: last-tenth 0.76-0.91 with late signals against
0.61-0.75 without). `late` chooses what a late signal earns: "count" (the
default, the same update as one that landed), "ignore" (nothing: the
node-perturbation estimator proper, only the signals that landed) or
"depress" (the opposite update, the shape of spike-timing-dependent
plasticity, where a presynaptic spike after the postsynaptic one weakens
the synapse). A neuron nudged towards firing in an
   epoch that turned out better than usual gets stronger inputs from the
   neurons that fed it; in a worse epoch, weaker.

This is the REINFORCE / node-perturbation estimator of the reward gradient,
a three-factor rule: presynaptic activity x postsynaptic perturbation x
global reward. With `eligibility="hebb"` the perturbation is replaced by a
plain Hebbian term (+1 if the target fired, -1 if not) and no noise is
injected, which is the classic reward-modulated Hebbian rule.

Forced inputs are never adjusted and weights are kept within the grid's
weight_range, [-1, 1] by default.

**Homeostasis.** A neuron whose input sits far from its threshold is never
flipped by the exploration noise, gets no learning signal, and stays "stuck"
on or off. Every neuron therefore tracks its own firing rate and nudges its
threshold toward a target rate each epoch: firing too often raises the
threshold, too rarely lowers it. Neurons forced in wave 0 this epoch are
left out of both, exactly as reinforcement leaves their incoming weights
alone: their firing was not the network's doing. An unforced input neuron
is an ordinary neuron and is treated as one. The default
rate of 1e-6 toward a target of 0.5 is a very slow drift (a fully stuck
neuron moves its threshold by about 0.0006 per thousand epochs, so the
effect belongs to runs of millions of epochs); a rate of 0 switches it off. Thresholds may go negative, within `THRESHOLD_RANGE`.
"""

from __future__ import annotations

import random
from typing import Callable, Sequence

from .grid import GridOfNeurons
from .constants import (
    BASELINE_RATE, CRITIC, ELIGIBILITY, HOMEOSTASIS, LATE, LR, RATE_MEMORY, RULE, SIGMA, STUCK_ABOVE, STUCK_BELOW,
    TARGET, TARGET_RATE, THRESHOLD_RANGE, UNSTICK, UNSTICK_TARGET, WINDOW,
)
from .dopamine import Dopamine
from .monitor import run_epoch
from .neuron import Neuron

Target = Callable[[Sequence[bool]], list[bool]]

TARGETS: dict[str, Target] = {
    "reversed": lambda pattern: list(pattern)[::-1],
    "copy": lambda pattern: list(pattern),
    "all-off": lambda pattern: [False] * len(pattern),
    "all-on": lambda pattern: [True] * len(pattern),
}

RULES = ("dopamine", "reinforce")  # which learning rule runs (AUTHORITY.md §6)
ELIGIBILITIES = ("perturb", "hebb")
LATE_RULES = ("count", "ignore", "depress")  # what a signal that arrived after its target fired earns


def arrays(grid) -> bool:
    """True for the array engine (arrays.ArrayNetwork), which does its own vector updates."""
    return getattr(grid, "engine", "objects") == "arrays"


def output_row(grid: GridOfNeurons) -> list[Neuron]:
    """The network's output neurons, in word order: the top row, or a container's own output surface."""
    return grid.output_row()


def expected_outputs(grid: GridOfNeurons, target: str = "reversed") -> list[bool]:
    """What the top row should show for the grid's current input pattern."""
    if grid.input_pattern is None:
        raise ValueError("no input pattern set")
    return TARGETS[target](grid.input_pattern)


def output_fired(grid: GridOfNeurons) -> list[bool]:
    """Whether each output neuron is on, left to right, whichever engine runs the grid (see Network.output_fired)."""
    return grid.output_fired()


def output_errors(grid: GridOfNeurons, target: str = "reversed") -> dict[Neuron, int]:
    """Error per output neuron: +1 should have fired, -1 should not have, 0 correct."""
    return {
        neuron: int(want) - int(fired)
        for neuron, fired, want in zip(output_row(grid), output_fired(grid), expected_outputs(grid, target))
    }


def accuracy(grid: GridOfNeurons, target: str = "reversed") -> float:
    """Fraction of the output row that matches the target, 0 to 1. This is the reward."""
    fired = output_fired(grid)
    want = expected_outputs(grid, target)
    return sum(1 for f, w in zip(fired, want) if f == w) / len(want)


# --- reading the output row as a receiver would ------------------------------------


def read_output_word(grid: GridOfNeurons, target: str = "reversed") -> list[bool | None]:
    """Undo the target's arrangement and the permutation, then resolve each complement pair.

    The output row is what the network produced; the target says where each
    coded bit was meant to land (reversed: place j shows coded bit
    permutation[across-1-j]). Coded bit k and its complement k + half form a
    pair: if exactly one of them fired, the bit is read; if both or neither
    did, the bit is unreadable (None) and counts as an error for the code.
    """
    fired = output_fired(grid)
    width = len(fired)
    if target == "reversed":
        placed = fired[::-1]  # placed[i] is what place i of the input arrangement would show
    elif target == "copy":
        placed = fired
    else:
        raise ValueError(f"the output word can only be read for the reversed or copy target, not {target!r}")
    coded = [False] * width
    for i, k in enumerate(grid.permutation):
        coded[k] = placed[i]
    half = width // 2
    word: list[bool | None] = []
    for k in range(half):
        bit, complement = coded[k], coded[k + half]
        word.append(bit if bit != complement else None)
    return word


def decoded_output(grid: GridOfNeurons, target: str = "reversed") -> list[bool]:
    """The data bits a receiver would decode from the output row, after error correction if a code is on.

    Unreadable bits are taken as 0 before correction, so a single unreadable
    or wrong bit is repaired by a correcting code.
    """
    word = [False if b is None else b for b in read_output_word(grid, target)]
    if grid.code:
        return grid.code.decode(word)
    return word


def expected_data(grid: GridOfNeurons) -> list[bool]:
    """What the receiver should decode: the data bits when a code is on, else the raw input bits."""
    if grid.input_bits is None:
        raise ValueError("no input pattern set")
    return list(grid.input_data if grid.code else grid.input_bits)


def decoded_accuracy(grid: GridOfNeurons, target: str = "reversed") -> float:
    """Fraction of the corrected, decoded data bits that are right, 0 to 1."""
    want = expected_data(grid)
    got = decoded_output(grid, target)
    return sum(a == b for a, b in zip(got, want)) / len(want)


def decoded_exact(grid: GridOfNeurons, target: str = "reversed") -> float:
    """1 if the corrected, decoded data bits are all right, else 0."""
    return 1.0 if decoded_output(grid, target) == expected_data(grid) else 0.0


CRITICS = {
    "row": accuracy,  # fraction of the output row matching the target, neuron by neuron
    "decoded": decoded_accuracy,  # fraction of data bits right after reading and error-correcting the row
    "decoded-exact": decoded_exact,  # all data bits right after correction, or nothing
}


def reward(grid: GridOfNeurons, target: str = "reversed", critic: str = "row") -> float:
    """The scalar the network is judged by, according to the chosen critic."""
    return CRITICS[critic](grid, target)


def forced(neuron: Neuron) -> bool:
    """True if the neuron was forced to fire by the stimulus this epoch."""
    return neuron.forced


def update_rates(grid: GridOfNeurons) -> None:
    """Move every neuron's running firing-rate estimate toward what it did this epoch.

    A neuron forced this epoch is skipped: that firing says nothing about the network.
    """
    if arrays(grid):
        return grid.update_rates()
    for neuron in grid.all_neurons():
        if not forced(neuron):
            neuron.rate += RATE_MEMORY * ((1.0 if neuron.has_fired else 0.0) - neuron.rate)


def stuck_neurons(grid: GridOfNeurons) -> tuple[list[Neuron], list[Neuron]]:
    """Neurons whose running rate is (almost) always on, and always off (indices, for the array engine)."""
    if arrays(grid):
        return grid.stuck()
    on = [n for n in grid.all_neurons() if n.rate > STUCK_ABOVE]
    off = [n for n in grid.all_neurons() if n.rate < STUCK_BELOW]
    return on, off


def homeostasis(
    grid: GridOfNeurons, rate: float, target: float = TARGET_RATE, threshold_range: tuple[float, float] = THRESHOLD_RANGE
) -> int:
    """Nudge each neuron's threshold toward its target firing rate, except those forced this epoch.

    Returns the number of neurons moved.
    """
    if arrays(grid):
        return grid.homeostasis(rate, target, threshold_range)
    if rate <= 0:
        return 0
    low, high = threshold_range
    moved = 0
    for neuron in grid.all_neurons():
        if forced(neuron):
            continue
        threshold = neuron.threshold + rate * (neuron.rate - target)
        neuron.threshold = max(low, min(high, threshold))
        moved += 1
    return moved


def unstick_outputs(
    grid: GridOfNeurons,
    rate: float,
    target: float = 0.5,
    threshold_range: tuple[float, float] = THRESHOLD_RANGE,
) -> list[Neuron]:
    """Nudge the threshold of every *stuck* output neuron toward a target firing rate.

    Only output neurons whose running rate is beyond the stuck band (almost
    always on, or almost always off) are touched, and only while they are.
    A saturated output gets no learning signal because the exploration noise
    never changes whether it fires; moving its threshold back toward the
    region where the noise matters gives the rule a gradient there, and
    nothing else in the mesh is disturbed. Returns the neurons nudged (indices, for the array engine).
    """
    if arrays(grid):
        return grid.unstick_outputs(rate, target, threshold_range)
    if rate <= 0:
        return []
    low, high = threshold_range
    nudged = []
    for neuron in output_row(grid):
        if neuron.rate > STUCK_ABOVE or neuron.rate < STUCK_BELOW:
            neuron.threshold = max(low, min(high, neuron.threshold + rate * (neuron.rate - target)))
            nudged.append(neuron)
    return nudged


def delivered_signals(grid: GridOfNeurons) -> list:
    """Every signal delivered in the last epoch, in wave order, each exactly once.

    No deduplication is needed: a neuron fires at most once per epoch, so each
    of its active outgoing connections carries at most one signal. This
    includes signals that arrived after their target had fired; see `landed`.
    """
    return [signal for wave in grid.waves for signal in wave.signals()]


def landed(signal) -> bool:
    """True if the signal was taken in: its target had not fired before the wave it arrived in.

    A neuron ignores input once it has fired (Neuron.receive), but propagation
    still records the delivery. A signal from a later wave than the target's
    firing wave changed nothing and must earn no credit or blame.
    """
    fired_in = signal.target.fired_in_wave
    return fired_in is None or signal.wave <= fired_in


def delivered_connections(grid: GridOfNeurons) -> list:
    """Every connection that carried a signal in the last epoch, landed or not (see delivered_signals)."""
    return [connection for wave in grid.waves for connection in wave.delivered]


def reinforce(
    grid: GridOfNeurons,
    advantage: float,
    lr: float = LR,
    sigma: float = SIGMA,
    eligibility: str = ELIGIBILITY,
    late: str = LATE,
) -> int:
    """Apply the global-reward update for the epoch that has just run. Returns connections changed.

    `late` says what a signal that arrived after its target fired (see `landed`)
    earns: "count" the same update as one that landed, "ignore" none, or
    "depress" the opposite.
    """
    if eligibility not in ELIGIBILITIES:
        raise ValueError(f"unknown eligibility {eligibility!r}; choose from {', '.join(ELIGIBILITIES)}")
    if late not in LATE_RULES:
        raise ValueError(f"unknown late-signal rule {late!r}; choose from {', '.join(LATE_RULES)}")
    if arrays(grid):
        return grid.reinforce(advantage, lr, sigma, eligibility, late)
    if not advantage:
        return 0
    low, high = grid.weight_range
    step = lr * advantage
    perturb = eligibility == "perturb"
    changed = 0
    last_delivery: dict = {}  # once per connection per epoch, by its last delivery (a source may fire more than once)
    for wave in grid.waves:
        for connection in wave.delivered:
            last_delivery[connection] = wave.number
    for connection, arrived in last_delivery.items():
        if True:
            target = connection.target
            if target.forced:
                continue  # a forced input: its firing was not the network's doing
            fired_in = target.fired_in_wave
            if perturb:
                e = target.noise / sigma if sigma else 0.0
            else:
                e = 1.0 if target.has_fired else -1.0
            if late != "count" and fired_in is not None and arrived > fired_in:
                if late == "ignore":
                    continue  # dropped on arrival: it changed nothing this epoch
                e = -e  # arrived after the firing: weakened where an early one would be strengthened
            if not e:
                continue
            weight = connection.weight + step * e
            if weight < low:
                weight = low
            elif weight > high:
                weight = high
            connection.weight = weight
            changed += 1
    return changed


class Teacher:
    """Runs epochs with exploration noise and scores them; under the reinforce rule, also reinforces every connection.

    Use `teacher.epoch()` in place of `run_epoch(grid)`: it injects the
    exploration noise before the epoch and, with `rule="reinforce"`, applies
    the update after it. With the default `rule="dopamine"` the network
    learns by itself as it runs (a Dopamine is attached to the grid if it
    has none) and the Teacher scores, keeps the firing rates, homeostasis
    and un-sticking, and reports.
    """

    def __init__(
        self,
        grid: GridOfNeurons,
        target: str = TARGET,
        lr: float = LR,
        sigma: float = SIGMA,
        eligibility: str = ELIGIBILITY,
        baseline_rate: float = BASELINE_RATE,
        window: int = WINDOW,
        seed: int | None = None,
        homeostasis: float = HOMEOSTASIS,
        target_rate: float = TARGET_RATE,
        threshold_range: tuple[float, float] = THRESHOLD_RANGE,
        discharge: bool = False,
        unstick: float = UNSTICK,
        unstick_target: float = UNSTICK_TARGET,
        critic: str = CRITIC,
        late: str = LATE,
        rule: str = RULE,
    ):
        if rule not in RULES:
            raise ValueError(f"unknown learning rule {rule!r}; choose from {', '.join(RULES)}")
        self.rule = rule
        if rule == "dopamine" and getattr(grid, "dopamine", None) is None:
            grid.dopamine = Dopamine(lr=lr)
        if target not in TARGETS:
            raise ValueError(f"unknown target {target!r}; choose from {', '.join(TARGETS)}")
        if critic not in CRITICS:
            raise ValueError(f"unknown critic {critic!r}; choose from {', '.join(CRITICS)}")
        if critic != "row" and target not in ("reversed", "copy"):
            raise ValueError(f"the {critic} critic reads the output as a word, which needs the reversed or copy target")
        self.critic = critic
        if late not in LATE_RULES:
            raise ValueError(f"unknown late-signal rule {late!r}; choose from {', '.join(LATE_RULES)}")
        self.late = late  # what a signal arriving after its target fired earns
        if eligibility not in ELIGIBILITIES:
            raise ValueError(f"unknown eligibility {eligibility!r}; choose from {', '.join(ELIGIBILITIES)}")
        if lr < 0 or sigma < 0 or homeostasis < 0 or unstick < 0:
            raise ValueError("learning rate, sigma, homeostasis and unstick rates must not be negative")
        if not 0.0 < unstick_target < 1.0:
            raise ValueError(f"unstick target firing rate must be between 0 and 1, got {unstick_target}")
        self.unstick = unstick
        self.unstick_target = unstick_target
        self.unstuck_count = 0  # how many epoch-nudges the output un-sticking has applied
        self.history: list[dict] = []  # one entry per progress report; saved in checkpoints
        if not 0.0 < target_rate < 1.0:
            raise ValueError(f"target firing rate must be between 0 and 1, got {target_rate}")
        low, high = threshold_range
        if not low < high:
            raise ValueError(f"threshold range must run from low to high, got {threshold_range}")
        self.homeostasis = homeostasis
        self.target_rate = target_rate
        self.threshold_range = (float(low), float(high))
        self.discharge = discharge  # zero every potential between inputs instead of letting it leak
        self.grid = grid
        self.target = target
        self.lr = lr
        self.sigma = sigma if eligibility == "perturb" else 0.0
        self.eligibility = eligibility
        self.baseline_rate = baseline_rate
        self.window = window
        self.rng = random.Random(seed)
        self.epochs = 0
        self.total_reward = 0.0  # sum of every epoch's reward, for accuracy to date
        self.baseline: float | None = None  # running average reward: what "usual" looks like
        self.last_reward: float | None = None
        self.average: float | None = None  # exponential moving average over about `window` epochs
        self._trace = None  # a text file the per-epoch trace is written to (see trace_to)

    def trace_to(self, path) -> None:
        """Write one line per epoch to `path` (CSV, appended): epoch, time, dopamine, expected, score."""
        import os
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        self._trace = open(path, "a", buffering=1)
        if new:
            self._trace.write("epoch,time_ms,dopamine,expected,score\n")

    def close_trace(self) -> None:
        if self._trace is not None:
            self._trace.close()
            self._trace = None

    def epoch(self, bits: Sequence[bool] | None = None, verbose: bool = True) -> float:
        """Run one epoch with exploration noise, then learn from it. Returns its reward."""
        run_epoch(self.grid, bits, verbose=verbose, noise=self.sigma, rng=self.rng, discharge=self.discharge)
        return self.step()

    def step(self) -> float:
        """Score the epoch that has just run and reinforce. Returns its reward (accuracy)."""
        reward = CRITICS[self.critic](self.grid, self.target)
        if self.baseline is None:
            self.baseline = reward
        advantage = reward - self.baseline
        if self.rule == "reinforce":
            reinforce(self.grid, advantage, self.lr, self.sigma, self.eligibility, self.late)
        update_rates(self.grid)
        homeostasis(self.grid, self.homeostasis, self.target_rate, self.threshold_range)
        self.unstuck_count += len(unstick_outputs(self.grid, self.unstick, self.unstick_target, self.threshold_range))
        self.baseline += self.baseline_rate * (reward - self.baseline)
        self.epochs += 1
        if self._trace is not None:
            pool = self.grid.dopamine
            level, expected = ("", "") if pool is None else (f"{pool.peek(self.grid.horizon):.6g}", f"{pool.expected():.6g}")
            self._trace.write(f"{self.grid.epoch},{self.grid.time:g},{level},{expected},{reward:.6g}\n")
        self.total_reward += reward
        self.last_reward = reward
        if self.average is None:
            self.average = reward
        else:
            alpha = 2.0 / (self.window + 1)
            self.average = (1 - alpha) * self.average + alpha * reward
        return reward

    @property
    def accuracy_to_date(self) -> float | None:
        """Mean reward over every epoch taught so far."""
        return self.total_reward / self.epochs if self.epochs else None

    def record(self, elapsed: float | None = None, epochs_per_second: float | None = None) -> dict:
        """Append the current figures to the history (called at each progress report). Returns the entry."""
        on, off = stuck_neurons(self.grid)
        entry = {
            "epoch": self.grid.epoch,
            "elapsed": None if elapsed is None else round(elapsed, 1),
            "accuracy_to_date": None if self.accuracy_to_date is None else round(self.accuracy_to_date, 4),
            "recent": None if self.average is None else round(self.average, 4),
            "stuck_on": len(on),
            "stuck_off": len(off),
            "epochs_per_second": None if epochs_per_second is None else round(epochs_per_second),
        }
        self.history.append(entry)
        return entry

    def status(self) -> str:
        verb = "learning" if self.rule == "reinforce" else "scoring"  # under the dopamine rule the Teacher only scores
        if self.average is None:
            return f"{verb} {self.target}: no epochs yet"
        if self.rule == "dopamine":
            settings = f"{self.grid.dopamine.status()}, sigma {self.sigma:g}"
        else:
            settings = f"{self.eligibility}, lr {self.lr:g}, sigma {self.sigma:g}"
        if self.critic != "row":
            settings += f", critic {self.critic}"
        if self.late != "count":
            settings += f", late signals {self.late}d"
        if self.homeostasis:
            low, high = self.threshold_range
            settings += f", homeostasis {self.homeostasis:g} toward {self.target_rate:g} in [{low:g}, {high:g}]"
        if self.unstick:
            settings += f", unstick {self.unstick:g}"
        if self.discharge:
            settings += ", discharge"
        return (
            f"{verb} {self.target} ({settings}): "
            f"accuracy {self.accuracy_to_date:.1%} to date over {self.epochs:,} epochs, "
            f"{self.average:.0%} recent"
        )
