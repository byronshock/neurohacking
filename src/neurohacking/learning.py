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

Forced inputs are never adjusted and weights are kept within [-1, 1].
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


def delivered_connections(grid: GridOfNeurons):
    """Every connection that carried a signal in the last epoch (each once)."""
    seen = set()
    for wave in grid.waves:
        for signal in wave.delivered:
            seen.add(signal.connection)
    return seen


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
    changed = 0
    for connection in delivered_connections(grid):
        target = connection.target
        if target.fired_in_wave == 0:
            continue  # a forced input: its firing was not the network's doing
        if eligibility == "perturb":
            e = target.noise / sigma if sigma else 0.0
        else:
            e = 1.0 if target.has_fired else -1.0
        if e:
            connection.weight = max(-1.0, min(1.0, connection.weight + lr * advantage * e))
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
    ):
        if target not in TARGETS:
            raise ValueError(f"unknown target {target!r}; choose from {', '.join(TARGETS)}")
        if eligibility not in ELIGIBILITIES:
            raise ValueError(f"unknown eligibility {eligibility!r}; choose from {', '.join(ELIGIBILITIES)}")
        if lr < 0 or sigma < 0:
            raise ValueError("learning rate and sigma must not be negative")
        self.grid = grid
        self.target = target
        self.lr = lr
        self.sigma = sigma if eligibility == "perturb" else 0.0
        self.eligibility = eligibility
        self.baseline_rate = baseline_rate
        self.window = window
        self.rng = random.Random(seed)
        self.epochs = 0
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
        self.baseline += self.baseline_rate * (reward - self.baseline)
        self.epochs += 1
        self.last_reward = reward
        if self.average is None:
            self.average = reward
        else:
            alpha = 2.0 / (self.window + 1)
            self.average = (1 - alpha) * self.average + alpha * reward
        return reward

    def status(self) -> str:
        if self.average is None:
            return f"learning {self.target}: no epochs yet"
        return (
            f"learning {self.target} ({self.eligibility}, lr {self.lr:g}): "
            f"accuracy {self.average:.0%} avg, {self.last_reward:.0%} last"
        )
