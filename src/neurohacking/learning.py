"""Teaching the network by global reinforcement: a scalar reward broadcast to every connection.

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
   exploration noise (normalised). A neuron nudged towards firing in an
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
on or off. Every neuron outside the input row therefore tracks its own
firing rate and nudges its threshold toward a target rate each epoch:
firing too often raises the threshold, too rarely lowers it. The default
rate of 1e-5 toward a target of 0.4 is a slow drift (a fully stuck neuron
moves its threshold by about 0.006 per thousand epochs); a rate of 0
switches it off. Thresholds may go negative, within `THRESHOLD_RANGE`.
"""

from __future__ import annotations

import random
from typing import Callable, Sequence

from .grid import GridOfNeurons
from .monitor import run_epoch
from .neuron import Neuron

Target = Callable[[Sequence[bool]], list[bool]]

TARGETS: dict[str, Target] = {
    "reversed": lambda pattern: list(pattern)[::-1],
    "copy": lambda pattern: list(pattern),
    "all-off": lambda pattern: [False] * len(pattern),
    "all-on": lambda pattern: [True] * len(pattern),
}

ELIGIBILITIES = ("perturb", "hebb")
RATE_MEMORY = 0.01  # per-epoch update of a neuron's running firing rate (about the last 100 epochs)
STUCK_BELOW, STUCK_ABOVE = 0.01, 0.99  # a neuron firing less or more often than this is "stuck"
THRESHOLD_RANGE = (-5.0, 5.0)  # default limits on what homeostasis may move a threshold to


def output_row(grid: GridOfNeurons) -> list[Neuron]:
    """The top row of neurons, left to right: the network's output."""
    return [grid.get_neuron_at(column, 0) for column in range(grid.columns)]


def expected_outputs(grid: GridOfNeurons, target: str = "reversed") -> list[bool]:
    """What the top row should show for the grid's current input pattern."""
    if grid.input_pattern is None:
        raise ValueError("no input pattern set")
    return TARGETS[target](grid.input_pattern)


def output_errors(grid: GridOfNeurons, target: str = "reversed") -> dict[Neuron, int]:
    """Error per output neuron: +1 should have fired, -1 should not have, 0 correct."""
    return {
        neuron: int(want) - int(neuron.has_fired)
        for neuron, want in zip(output_row(grid), expected_outputs(grid, target))
    }


def accuracy(grid: GridOfNeurons, target: str = "reversed") -> float:
    """Fraction of the output row that matches the target, 0 to 1. This is the reward."""
    errors = output_errors(grid, target)
    return sum(1 for e in errors.values() if e == 0) / len(errors)


def update_rates(grid: GridOfNeurons) -> None:
    """Move every neuron's running firing-rate estimate toward what it did this epoch."""
    for neuron in grid.neurons.values():
        neuron.rate += RATE_MEMORY * ((1.0 if neuron.has_fired else 0.0) - neuron.rate)


def stuck_neurons(grid: GridOfNeurons) -> tuple[list[Neuron], list[Neuron]]:
    """Neurons outside the input row whose running rate is (almost) always on, and always off."""
    inputs = set(grid.input_row())
    candidates = [n for n in grid.neurons.values() if n not in inputs]
    on = [n for n in candidates if n.rate > STUCK_ABOVE]
    off = [n for n in candidates if n.rate < STUCK_BELOW]
    return on, off


def homeostasis(
    grid: GridOfNeurons, rate: float, target: float = 0.4, threshold_range: tuple[float, float] = THRESHOLD_RANGE
) -> int:
    """Nudge each non-input neuron's threshold toward its target firing rate. Returns neurons moved."""
    if rate <= 0:
        return 0
    low, high = threshold_range
    inputs = set(grid.input_row())
    moved = 0
    for neuron in grid.neurons.values():
        if neuron in inputs:
            continue
        threshold = neuron.threshold + rate * (neuron.rate - target)
        neuron.threshold = max(low, min(high, threshold))
        moved += 1
    return moved


def delivered_connections(grid: GridOfNeurons) -> list:
    """Every connection that carried a signal in the last epoch, each exactly once.

    No deduplication is needed: a neuron fires at most once per epoch, so each
    of its active outgoing connections carries at most one signal.
    """
    return [signal.connection for wave in grid.waves for signal in wave.delivered]


def reinforce(
    grid: GridOfNeurons,
    advantage: float,
    lr: float = 0.03,
    sigma: float = 0.1,
    eligibility: str = "perturb",
) -> int:
    """Apply the global-reward update for the epoch that has just run. Returns connections changed."""
    if eligibility not in ELIGIBILITIES:
        raise ValueError(f"unknown eligibility {eligibility!r}; choose from {', '.join(ELIGIBILITIES)}")
    if not advantage:
        return 0
    low, high = grid.weight_range
    step = lr * advantage
    perturb = eligibility == "perturb"
    changed = 0
    for connection in delivered_connections(grid):
        target = connection.target
        if target.fired_in_wave == 0:
            continue  # a forced input: its firing was not the network's doing
        if perturb:
            e = target.noise / sigma if sigma else 0.0
        else:
            e = 1.0 if target.has_fired else -1.0
        if e:
            weight = connection.weight + step * e
            if weight < low:
                weight = low
            elif weight > high:
                weight = high
            connection.weight = weight
            changed += 1
    return changed


class Teacher:
    """Runs epochs with exploration noise, scores them, and reinforces every connection.

    Use `teacher.epoch()` in place of `run_epoch(grid)`: it injects the
    exploration noise before the epoch and applies the update after it.
    """

    def __init__(
        self,
        grid: GridOfNeurons,
        target: str = "reversed",
        lr: float = 0.03,
        sigma: float = 0.1,
        eligibility: str = "perturb",
        baseline_rate: float = 0.05,
        window: int = 200,
        seed: int | None = None,
        homeostasis: float = 1e-5,
        target_rate: float = 0.4,
        threshold_range: tuple[float, float] = THRESHOLD_RANGE,
    ):
        if target not in TARGETS:
            raise ValueError(f"unknown target {target!r}; choose from {', '.join(TARGETS)}")
        if eligibility not in ELIGIBILITIES:
            raise ValueError(f"unknown eligibility {eligibility!r}; choose from {', '.join(ELIGIBILITIES)}")
        if lr < 0 or sigma < 0 or homeostasis < 0:
            raise ValueError("learning rate, sigma and homeostasis rate must not be negative")
        if not 0.0 < target_rate < 1.0:
            raise ValueError(f"target firing rate must be between 0 and 1, got {target_rate}")
        low, high = threshold_range
        if not low < high:
            raise ValueError(f"threshold range must run from low to high, got {threshold_range}")
        self.homeostasis = homeostasis
        self.target_rate = target_rate
        self.threshold_range = (float(low), float(high))
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

    def epoch(self, bits: Sequence[bool] | None = None, verbose: bool = True) -> float:
        """Run one epoch with exploration noise, then learn from it. Returns its reward."""
        run_epoch(self.grid, bits, verbose=verbose, noise=self.sigma, rng=self.rng)
        return self.step()

    def step(self) -> float:
        """Score the epoch that has just run and reinforce. Returns its reward (accuracy)."""
        reward = accuracy(self.grid, self.target)
        if self.baseline is None:
            self.baseline = reward
        advantage = reward - self.baseline
        reinforce(self.grid, advantage, self.lr, self.sigma, self.eligibility)
        update_rates(self.grid)
        homeostasis(self.grid, self.homeostasis, self.target_rate, self.threshold_range)
        self.baseline += self.baseline_rate * (reward - self.baseline)
        self.epochs += 1
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

    def stuck(self) -> str:
        """Short summary of stuck neurons, e.g. '26 on + 17 off of 56 stuck'."""
        on, off = stuck_neurons(self.grid)
        total = len(self.grid.neurons) - self.grid.columns
        return f"{len(on)} on + {len(off)} off of {total} stuck"

    def status(self) -> str:
        if self.average is None:
            return f"learning {self.target}: no epochs yet"
        settings = f"{self.eligibility}, lr {self.lr:g}, sigma {self.sigma:g}"
        if self.homeostasis:
            low, high = self.threshold_range
            settings += f", homeostasis {self.homeostasis:g} toward {self.target_rate:g} in [{low:g}, {high:g}]"
        return (
            f"learning {self.target} ({settings}): "
            f"accuracy {self.accuracy_to_date:.1%} to date over {self.epochs:,} epochs, "
            f"{self.average:.0%} recent, {self.stuck()}"
        )
