"""Teaching the network: what the top row should show, how well it does, and a rule to improve it.

The top row is the network's output. For each epoch a target pattern is
derived from the input pattern (by default its reverse), the output row is
compared with it, and the error is used to adjust weights.

The teaching rule backpropagates the output error through the epoch's waves.
Because signals only ever travel from earlier waves to later ones, the epoch
is a directed acyclic graph, and the error can be pushed back along it. Each
neuron's threshold is treated as if it passed the error straight through
(the "straight-through estimator" used to train binary networks):

* an output neuron that should have fired but did not has error +1; one
  that fired but should not have has error -1;
* every connection from a fired source into an erring neuron has its weight
  moved by `lr * error` (at the output row alone this is the perceptron rule);
* the error is passed upstream to each source, fired or not, scaled by the
  connection weight, so that neurons further back can learn as well.

The forced inputs (the neurons fired in wave 0) are never adjusted, and
weights are kept within [-1, 1].
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Sequence

from .grid import GridOfNeurons
from .neuron import Neuron

Target = Callable[[Sequence[bool]], list[bool]]

TARGETS: dict[str, Target] = {
    "reversed": lambda pattern: list(pattern)[::-1],
    "copy": lambda pattern: list(pattern),
    "all-off": lambda pattern: [False] * len(pattern),
    "all-on": lambda pattern: [True] * len(pattern),
}


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
    """Fraction of the output row that matches the target, 0 to 1."""
    errors = output_errors(grid, target)
    return sum(1 for e in errors.values() if e == 0) / len(errors)


def teach(grid: GridOfNeurons, lr: float = 0.05, target: str = "reversed") -> float:
    """Adjust weights from the last epoch's errors. Returns that epoch's accuracy (before teaching).

    Must be called after an epoch has run, while grid.waves still describes it.
    """
    errors = output_errors(grid, target)
    if all(e == 0 for e in errors.values()):
        return 1.0

    # When each neuron "happened": the wave it fired in, or for a neuron that
    # never fired, the last wave in which it received a signal.
    depth: dict[Neuron, int] = {}
    for wave in grid.waves:
        for signal in wave.delivered:
            if not signal.target.has_fired:
                depth[signal.target] = wave.number
    for neuron in grid.neurons.values():
        if neuron.has_fired:
            depth[neuron] = neuron.fired_in_wave
    top = max(depth.values(), default=0) + 1

    delta: dict[Neuron, float] = {}
    for neuron, error in errors.items():
        if error:
            delta[neuron] = float(error)
            depth.setdefault(neuron, top)  # an output nothing ever reached still counts

    by_depth: dict[int, list[Neuron]] = defaultdict(list)
    for neuron, d in depth.items():
        by_depth[d].append(neuron)

    for d in range(top, 0, -1):  # wave 0 holds the forced inputs: never adjusted
        for neuron in by_depth[d]:
            error = delta.get(neuron, 0.0)
            if not error:
                continue
            for connection in neuron.incoming:
                source = connection.source
                if source not in depth or depth[source] >= d:
                    continue  # the source did not happen before this neuron, so it cannot have contributed
                weight = connection.weight
                if source.has_fired:
                    connection.weight = max(-1.0, min(1.0, weight + lr * error))
                delta[source] = max(-1.0, min(1.0, delta.get(source, 0.0) + weight * error))

    return sum(1 for e in errors.values() if e == 0) / len(errors)


class Teacher:
    """Applies the teaching rule after each epoch and keeps a running accuracy."""

    def __init__(self, grid: GridOfNeurons, target: str = "reversed", lr: float = 0.05, window: int = 200):
        if target not in TARGETS:
            raise ValueError(f"unknown target {target!r}; choose from {', '.join(TARGETS)}")
        if lr < 0:
            raise ValueError(f"learning rate must not be negative, got {lr}")
        self.grid = grid
        self.target = target
        self.lr = lr
        self.window = window
        self.epochs = 0
        self.last_accuracy: float | None = None
        self.average: float | None = None  # exponential moving average over about `window` epochs

    def step(self) -> float:
        """Teach from the epoch that has just run. Returns its accuracy."""
        acc = teach(self.grid, self.lr, self.target)
        self.epochs += 1
        self.last_accuracy = acc
        if self.average is None:
            self.average = acc
        else:
            alpha = 2.0 / (self.window + 1)
            self.average = (1 - alpha) * self.average + alpha * acc
        return acc

    def status(self) -> str:
        if self.average is None:
            return f"learning {self.target}: no epochs yet"
        return f"learning {self.target} (lr {self.lr:g}): accuracy {self.average:.0%} avg, {self.last_accuracy:.0%} last"
