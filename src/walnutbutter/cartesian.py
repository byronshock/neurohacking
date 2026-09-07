"""A population of neurons at Cartesian positions, with no grid structure.

Positions are measured in *unit distances*: one unit is the natural scale of
the network, the step between neighbouring cells in the hex mesh, and a
population can span many of them.

By default the neurons form a hexagonal lattice of `columns` x `rows` points
at unit spacing (pointy-top: odd rows shifted half a unit right, rows
sqrt(3)/2 apart), centred on the origin, so each neuron's six nearest
neighbours are exactly one unit away. With `layout="random"`, `count`
neurons are instead placed uniformly at random inside a rectangular region
of `width` x `height` units. Neurons added with coordinates go exactly
there, inside the region or not. Connections are not made here: the neurons
are placed, and how they connect (typically by distance) is a separate
decision.
"""

from __future__ import annotations

import math
import random
from typing import Iterator

from .connection import Connection
from .network import Network
from .neuron import Neuron


ROW_SPACING = math.sqrt(3) / 2  # distance between rows of a unit-spaced hexagonal lattice
TOLERANCE = 1e-9  # lattice distances are 1 only up to rounding


def gaussian_density(distance: float, sigma: float = 1.0) -> float:
    """The isotropic 2D Gaussian probability density with standard deviation `sigma`, at `distance` from the origin."""
    return math.exp(-0.5 * (distance / sigma) ** 2) / (2.0 * math.pi * sigma * sigma)


def connection_probability(distance: float, sigma: float = 1.0, scale: float = 1.0) -> float:
    """Probability of a forward connection between two neurons `distance` units apart.

    Proportional to the 2D Gaussian density with standard deviation `sigma`
    (default 1, the standard normal) at that distance wherever the distance is
    nonzero, scaled so the probability approaches `scale` (default 1) as the
    distance approaches zero: exp(-d^2 / 2 sigma^2). With sigma 1, one unit
    apart gives 0.61, two units 0.135, three units 0.011; a larger sigma
    reaches further. Two neurons at the same position have probability exactly
    zero: they never project onto each other. Never above 1.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if distance <= TOLERANCE:
        return 0.0
    return min(1.0, scale * gaussian_density(distance, sigma) / gaussian_density(0.0, sigma))


class CartesianNodes(Network):
    """Neurons at (x, y) positions in unit distances: a hexagonal lattice by default, or random."""

    def __init__(
        self,
        columns: int = 8,
        rows: int = 10,
        layout: str = "hex",
        count: int | None = None,
        width: float | None = None,
        height: float | None = None,
        seed: int | None = None,
        threshold: float = 0.25,
        minimum_potential: float = -1.0,
        permute: bool = True,
        weight_range: tuple[float, float] = (-1.0, 1.0),
    ):
        if layout not in ("hex", "random"):
            raise ValueError(f"layout must be 'hex' or 'random', got {layout!r}")
        self.layout = layout
        self.columns = columns
        self.rows = rows
        self.seed = seed
        self.threshold = threshold
        self.minimum_potential = minimum_potential
        self._rng = random.Random(seed)
        self.neurons: list[Neuron] = []
        self.connections: dict[int, Connection] = {}  # by ID from 1, like the grid
        self._lattice: dict[tuple[int, int], Neuron] = {}
        self._init_network(columns, weight_range)
        if layout == "hex":
            if columns < 1 or rows < 1:
                raise ValueError(f"a lattice needs at least one column and one row, got {columns}x{rows}")
            if count is not None:
                raise ValueError("count applies to layout='random'; a hex lattice has columns x rows neurons")
            # The region is the lattice's footprint plus half a unit all round.
            self.width = float(width) if width is not None else columns + 0.5 + 1.0
            self.height = float(height) if height is not None else (rows - 1) * ROW_SPACING + 1.0
            self._place_lattice()
        else:
            self.width = float(width) if width is not None else float(columns)
            self.height = float(height) if height is not None else float(rows)
            count = 64 if count is None else count
            if count < 0:
                raise ValueError(f"count must not be negative, got {count}")
            for _ in range(count):
                self.add()
        if not (self.width > 0 and self.height > 0):
            raise ValueError(f"the region needs a positive width and height, got {self.width} x {self.height}")
        if permute and layout == "hex":
            self._rng.shuffle(self.permutation)

    def _place_lattice(self) -> None:
        """Columns x rows neurons at unit spacing, odd rows shifted half a unit, centred on the origin."""
        raw = [(c + 0.5 * (r % 2), r * ROW_SPACING, c, r) for r in range(self.rows) for c in range(self.columns)]
        xs = [x for x, _, _, _ in raw]
        ys = [y for _, y, _, _ in raw]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        for x, y, c, r in raw:
            neuron = self.add(x - cx, y - cy, name=f"Node_{c}_{r}")
            self._lattice[(c, r)] = neuron

    def node_at(self, column: int, row: int) -> Neuron | None:
        """The lattice neuron at (column, row), counted from the bottom-left; None for random layouts."""
        return self._lattice.get((column, row))

    def get_neuron_at(self, column: int, row: int) -> Neuron | None:
        """The lattice neuron at (column, row) counted from the top-left, like the hex grid.

        Row 0 is the top row (the output); row rows-1 is the bottom row (the input).
        """
        return self._lattice.get((column, self.rows - 1 - row))

    @property
    def region(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """((x_min, x_max), (y_min, y_max)) of the random placement region."""
        return (-self.width / 2, self.width / 2), (-self.height / 2, self.height / 2)

    # --- placing neurons --------------------------------------------------

    def add(
        self,
        x: float | None = None,
        y: float | None = None,
        name: str | None = None,
        threshold: float | None = None,
    ) -> Neuron:
        """Add one neuron. Coordinates left as None are drawn uniformly from the placement region."""
        (x_min, x_max), (y_min, y_max) = self.region
        x = self._rng.uniform(x_min, x_max) if x is None else float(x)
        y = self._rng.uniform(y_min, y_max) if y is None else float(y)
        neuron = Neuron(
            name if name is not None else f"Node_{len(self.neurons)}",
            threshold=self.threshold if threshold is None else threshold,
            minimum_potential=self.minimum_potential,
        )
        neuron.position = (x, y)
        self.neurons.append(neuron)
        return neuron

    def in_region(self, x: float, y: float) -> bool:
        """True if (x, y) lies inside the random placement region."""
        (x_min, x_max), (y_min, y_max) = self.region
        return x_min <= x <= x_max and y_min <= y <= y_max

    # --- looking around, in unit distances ----------------------------------

    def positions(self) -> list[tuple[float, float]]:
        return [neuron.position for neuron in self.neurons]

    def extent(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Bounding box of where the neurons actually are (the region if there are none)."""
        if not self.neurons:
            return self.region
        xs = [x for x, _ in self.positions()]
        ys = [y for _, y in self.positions()]
        return (min(xs), max(xs)), (min(ys), max(ys))

    @staticmethod
    def distance(a: Neuron, b: Neuron) -> float:
        """Distance between two placed neurons, in unit distances."""
        return math.dist(a.position, b.position)

    def nearest(self, x: float, y: float, count: int = 1, exclude: Neuron | None = None) -> list[Neuron]:
        """The `count` neurons closest to (x, y), nearest first."""
        candidates = [n for n in self.neurons if n is not exclude]
        return sorted(candidates, key=lambda n: math.dist(n.position, (x, y)))[:count]

    def within(self, x: float, y: float, radius: float = 1.0, exclude: Neuron | None = None) -> list[Neuron]:
        """Every neuron within `radius` unit distances of (x, y), nearest first (with a rounding tolerance)."""
        found = [n for n in self.neurons if n is not exclude and math.dist(n.position, (x, y)) <= radius + TOLERANCE]
        return sorted(found, key=lambda n: math.dist(n.position, (x, y)))

    def neighbours(self, neuron: Neuron, radius: float = 1.0) -> list[Neuron]:
        """Every other neuron within `radius` unit distances of `neuron`, nearest first."""
        return self.within(*neuron.position, radius=radius, exclude=neuron)

    # --- wiring by distance -------------------------------------------------

    def connect_by_distance(
        self,
        sigma: float = 1.0,
        scale: float = 1.0,
        weight: float | None = 1.0,
        weight_range: tuple[float, float] = (-1.0, 1.0),
    ) -> int:
        """Wire the population: A -> B with probability connection_probability(|AB|, sigma), each direction drawn independently.

        Every ordered pair of distinct neurons is considered once. A pair whose
        probability is zero (the same position, or a separation so large the
        density has underflowed) is never connected. `sigma` is the standard
        deviation of the Gaussian receptive field, in unit distances (default 1).
        `weight` is given to every new connection; None draws each uniformly
        from `weight_range`. Draws come from the container's seeded stream.
        Returns the number of connections made.
        """
        if scale < 0:
            raise ValueError(f"scale must not be negative, got {scale}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.receptive_field_sigma = sigma
        low, high = weight_range
        made = 0
        for source in self.neurons:
            for target in self.neurons:
                if target is source:
                    continue
                p = connection_probability(math.dist(source.position, target.position), sigma, scale)
                if p == 0.0 or self._rng.random() >= p:
                    continue
                w = self._rng.uniform(low, high) if weight is None else weight
                connection_id = len(self.connections) + 1
                self.connections[connection_id] = source.connect(target, connection_id, w)
                made += 1
        return made

    def get_connection(self, connection_id: int) -> Connection | None:
        return self.connections.get(connection_id)

    def mean_out_degree(self) -> float:
        return sum(len(n.outgoing) for n in self.neurons) / len(self.neurons) if self.neurons else 0.0

    # --- container protocol -----------------------------------------------

    def __len__(self) -> int:
        return len(self.neurons)

    def __iter__(self) -> Iterator[Neuron]:
        return iter(self.neurons)

    def __getitem__(self, index: int) -> Neuron:
        return self.neurons[index]

    def __repr__(self) -> str:
        if self.layout == "hex":
            return f"CartesianNodes({self.columns}x{self.rows} hexagonal lattice, {len(self.neurons)} neurons at unit spacing)"
        return f"CartesianNodes({len(self.neurons)} neurons at random in a {self.width:g} x {self.height:g} unit region)"
