"""A population of neurons at Cartesian positions, with no grid structure.

Each neuron has an (x, y) position inside a bounding box, by default the
square from -1 to 1 on both axes. A neuron added with coordinates is placed
there; one added without is placed at random, each coordinate drawn
uniformly across the box from the container's seeded random stream.
Connections are not made here: the neurons are placed, and how they connect
is a separate decision.
"""

from __future__ import annotations

import math
import random
from typing import Iterator, Sequence

from .neuron import Neuron

Bounds = tuple[tuple[float, float], tuple[float, float]]  # ((x_min, x_max), (y_min, y_max))
UNIT_SQUARE: Bounds = ((-1.0, 1.0), (-1.0, 1.0))


class CartesianNodes:
    """Neurons at (x, y) positions inside `bounds`, placed explicitly or at random."""

    def __init__(
        self,
        count: int = 0,
        bounds: Bounds = UNIT_SQUARE,
        seed: int | None = None,
        threshold: float = 0.25,
        minimum_potential: float = -1.0,
    ):
        (x_min, x_max), (y_min, y_max) = bounds
        if not (x_min < x_max and y_min < y_max):
            raise ValueError(f"bounds must run from min to max on each axis, got {bounds}")
        if count < 0:
            raise ValueError(f"count must not be negative, got {count}")
        self.bounds: Bounds = ((float(x_min), float(x_max)), (float(y_min), float(y_max)))
        self.seed = seed
        self.threshold = threshold
        self.minimum_potential = minimum_potential
        self._rng = random.Random(seed)
        self.neurons: list[Neuron] = []
        for _ in range(count):
            self.add()

    # --- placing neurons --------------------------------------------------

    def add(
        self,
        x: float | None = None,
        y: float | None = None,
        name: str | None = None,
        threshold: float | None = None,
    ) -> Neuron:
        """Add one neuron. Coordinates left as None are drawn at random within the bounds."""
        (x_min, x_max), (y_min, y_max) = self.bounds
        x = self._rng.uniform(x_min, x_max) if x is None else float(x)
        y = self._rng.uniform(y_min, y_max) if y is None else float(y)
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            raise ValueError(f"({x}, {y}) is outside the bounds {self.bounds}")
        neuron = Neuron(
            name if name is not None else f"Node_{len(self.neurons)}",
            threshold=self.threshold if threshold is None else threshold,
            minimum_potential=self.minimum_potential,
        )
        neuron.position = (x, y)
        self.neurons.append(neuron)
        return neuron

    def contains(self, x: float, y: float) -> bool:
        (x_min, x_max), (y_min, y_max) = self.bounds
        return x_min <= x <= x_max and y_min <= y <= y_max

    # --- looking around ---------------------------------------------------

    def positions(self) -> list[tuple[float, float]]:
        return [neuron.position for neuron in self.neurons]

    @staticmethod
    def distance(a: Neuron, b: Neuron) -> float:
        """Euclidean distance between two placed neurons."""
        return math.dist(a.position, b.position)

    def nearest(self, x: float, y: float, count: int = 1, exclude: Neuron | None = None) -> list[Neuron]:
        """The `count` neurons closest to (x, y), nearest first."""
        candidates = [n for n in self.neurons if n is not exclude]
        return sorted(candidates, key=lambda n: math.dist(n.position, (x, y)))[:count]

    def within(self, x: float, y: float, radius: float, exclude: Neuron | None = None) -> list[Neuron]:
        """Every neuron within `radius` of (x, y), nearest first."""
        found = [n for n in self.neurons if n is not exclude and math.dist(n.position, (x, y)) <= radius]
        return sorted(found, key=lambda n: math.dist(n.position, (x, y)))

    # --- state ------------------------------------------------------------

    def reset(self, discharge: bool = True) -> None:
        for neuron in self.neurons:
            neuron.reset(discharge)

    def fired_neurons(self) -> list[Neuron]:
        return [n for n in self.neurons if n.has_fired]

    # --- container protocol -----------------------------------------------

    def __len__(self) -> int:
        return len(self.neurons)

    def __iter__(self) -> Iterator[Neuron]:
        return iter(self.neurons)

    def __getitem__(self, index: int) -> Neuron:
        return self.neurons[index]

    def __repr__(self) -> str:
        (x_min, x_max), (y_min, y_max) = self.bounds
        return f"CartesianNodes({len(self.neurons)} neurons in [{x_min:g}, {x_max:g}] x [{y_min:g}, {y_max:g}])"
