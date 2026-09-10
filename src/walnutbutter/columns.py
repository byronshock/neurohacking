"""Hexagonal columns in R3: the prespread butter.

The plane is tiled with hexagonal cells, one neuron per cell, and every cell
is extruded into a **column** of `layers` neurons. The whole thing is one
unit across a cell: neurons sit half a unit apart in the plane, and the
layers of a column an eighth of a unit apart in z (a quarter of the cell
spacing, so a column reads as a compact stack, much closer than
neighbours). The guaranteed radius of one unit (plus epsilon) takes in a
neuron's six neighbours in its layer and their second ring, the same
eighteen cells the hex grid has always wired.

Connection rule (Byron's, September 10, 2026):

    same position:                                 never
    horizontal distance <= 1 + epsilon:            always
    otherwise:                                     small-world shortcuts only (omega)

The distance is measured in the plane only, so the guarantee holds at any
height: a neuron is wired to every neuron of its own column and of the
eighteen columns around it, in every layer. Everything further away is
reached only by shortcuts, drawn as on the grid: random source to random
non-neighbour target until they are the fraction omega of all connections.

The input butter is the bottom layer and the output butter the top layer.
With one layer the two coincide, and the network keeps the grid's
convention (bottom row in, top row out) so that a one-layer stack is the
hex grid itself: the same neurons, the same connections, and with the same
seed the same shortcuts, weights and permutation.
"""

from __future__ import annotations

import math
import random

from .connection import Connection
from .grid import DIRECTIONS, DIRECTIONS2, offset_to_axial
from .network import Network
from .neuron import Neuron

SPACING = 0.5  # between neighbouring cells: a cell is one unit across
ROW_SPACING = SPACING * math.sqrt(3) / 2
LAYER_SPACING = SPACING / 4  # between layers of a column: much closer than neighbours, a column of four is one cell tall
GUARANTEED = 1.0  # horizontal radius, in units, within which butter always connects


class HexColumns(Network):
    """`across` x `rows` hexagonal columns of `layers` neurons each, wired by the rule above."""

    def __init__(
        self,
        across: int = 8,
        rows: int = 10,
        layers: int = 1,
        weight: float | None = None,
        threshold: float = 0.25,
        seed: int | None = None,
        omega: float = 0.2,
        permute: bool = True,
        weight_range: tuple[float, float] = (-1.0, 1.0),
        minimum_potential: float = -1.0,
        epsilon: float = 1e-6,
    ):
        if across < 1 or rows < 1 or layers < 1:
            raise ValueError(f"columns need at least one neuron across, one row and one layer, got {across}x{rows}x{layers}")
        if not 0.0 <= omega < 1.0:
            raise ValueError(f"omega must be between 0 and 1 (exclusive), got {omega}")
        if epsilon < 0:
            raise ValueError(f"epsilon must not be negative, got {epsilon}")
        self.across, self.rows, self.layers = across, rows, layers
        self.weight = weight
        self.threshold = threshold
        self.minimum_potential = minimum_potential
        self.omega = omega
        self.epsilon = epsilon
        self.seed = seed
        self._rng = random.Random(seed)
        self.neurons: dict[tuple[int, int, int], Neuron] = {}  # (place, row, layer) -> Neuron
        self.connections: dict[int, Connection] = {}
        self._init_network(across, weight_range)
        self._place()
        self._wire(1.0 if weight is None else weight)
        self._add_shortcuts(1.0 if weight is None else weight)
        if weight is None:
            self.randomize_weights()
        self.permutation = list(range(self.input_width()))
        if permute:
            self._rng.shuffle(self.permutation)

    # --- building -----------------------------------------------------------

    def _place(self) -> None:
        """One neuron per cell per layer, row-major within a layer, layer 0 (the bottom) first.

        Cells are laid out exactly as the hex grid lays them out: the same axial
        coordinates from the same centring, so a row is shifted half a cell when
        the grid shifts it. Positions are in units, half a unit per cell.
        """
        self._axial: dict[tuple[int, int, int], tuple[int, int]] = {}
        self._by_axial: dict[tuple[int, int, int], Neuron] = {}  # (q, r, layer) -> Neuron
        centre_place, centre_row = self.across // 2, self.rows // 2
        for layer in range(self.layers):
            for row in range(self.rows):
                for place in range(self.across):
                    q, r = offset_to_axial(place - centre_place, row - centre_row)
                    neuron = Neuron(f"Column_{place}_{row}_L{layer}", threshold=self.threshold, minimum_potential=self.minimum_potential)
                    x = SPACING * (q + r / 2.0)  # pointy-top axial to the plane
                    y = -ROW_SPACING * r  # row 0 at the top, like the grid
                    neuron.position = (x, y, LAYER_SPACING * layer)
                    self.neurons[(place, row, layer)] = neuron
                    self._axial[(place, row, layer)] = (q, r)
                    self._by_axial[(q, r, layer)] = neuron

    def _ring(self, key: tuple[int, int, int], directions, layer: int):
        """The cells one step (DIRECTIONS) or two steps (DIRECTIONS2) from `key`, in `layer`, in the grid's order."""
        q, r = self._axial[key]
        for dq, dr in directions:
            other = self._by_axial.get((q + dq, r + dr, layer))
            if other is not None:
                yield other

    def _wire(self, weight: float) -> None:
        """Every guaranteed pair, each ordered pair once.

        Within a layer the first ring is wired for every neuron, then the second
        ring, in the grid's order, so a one-layer stack gets the grid's
        connection ids. Then, for each neuron, the same cells in the other
        layers (its own column first), nearest layer first. Every pair is
        checked against the distance rule itself; the visiting order only
        decides the ids.
        """
        reach = GUARANTEED + self.epsilon

        def link(source: Neuron, target: Neuron) -> None:
            if source is target or source.connection_to(target) is not None:
                return
            if self.horizontal_distance(source, target) > reach:
                return
            connection_id = len(self.connections) + 1
            self.connections[connection_id] = source.connect(target, connection_id, weight, kind=self._kind(source, target))

        for key, neuron in self.neurons.items():
            for other in self._ring(key, DIRECTIONS, key[2]):
                link(neuron, other)
        for key, neuron in self.neurons.items():
            for other in self._ring(key, DIRECTIONS2, key[2]):
                link(neuron, other)
        if self.layers > 1:
            for key, neuron in self.neurons.items():
                place, row, layer = key
                for z in sorted((z for z in range(self.layers) if z != layer), key=lambda z: (abs(z - layer), z)):
                    link(neuron, self.neurons[(place, row, z)])
                    for other in self._ring(key, DIRECTIONS, z):
                        link(neuron, other)
                    for other in self._ring(key, DIRECTIONS2, z):
                        link(neuron, other)

    @staticmethod
    def _kind(a: Neuron, b: Neuron) -> str:
        same_layer = a.position[2] == b.position[2]
        same_column = a.position[0] == b.position[0] and a.position[1] == b.position[1]
        if same_column:
            return "column"
        d = HexColumns.horizontal_distance(a, b)
        ring = "local" if d <= SPACING + 1e-9 else "local2"
        return ring if same_layer else ring + "-up"

    def is_guaranteed(self, a: Neuron, b: Neuron) -> bool:
        """The rule: not the same position, and within the guaranteed horizontal radius."""
        if a.position == b.position:
            return False
        return self.horizontal_distance(a, b) <= GUARANTEED + self.epsilon

    def _add_shortcuts(self, weight: float) -> None:
        """Small-world shortcuts as on the grid: the fraction omega of all connections, from random
        neurons to random neurons that are not guaranteed partners and not already targets."""
        local_count = len(self.connections)
        wanted = round(self.omega * local_count / (1.0 - self.omega))
        neurons = list(self.neurons.values())
        attempts_left = 100 * (wanted + 1)
        while wanted > 0 and attempts_left > 0:
            attempts_left -= 1
            source = self._rng.choice(neurons)
            target = self._rng.choice(neurons)
            if target is source or self.is_guaranteed(source, target):
                continue
            if source.connection_to(target) is not None:
                continue
            connection_id = len(self.connections) + 1
            self.connections[connection_id] = source.connect(target, connection_id, weight, kind="small_world")
            wanted -= 1

    def randomize_weights(self) -> None:
        low, high = self.weight_range
        for connection in self.connections.values():
            connection.weight = self._rng.uniform(low, high)

    # --- geometry -----------------------------------------------------------

    @staticmethod
    def horizontal_distance(a: Neuron, b: Neuron) -> float:
        """Distance in the plane, ignoring height, in units."""
        return math.hypot(a.position[0] - b.position[0], a.position[1] - b.position[1])

    @staticmethod
    def distance(a: Neuron, b: Neuron) -> float:
        return math.dist(a.position, b.position)

    def get_neuron_at(self, place: int, row: int, layer: int = 0) -> Neuron | None:
        """The neuron `place` along `row` (row 0 at the top) in `layer` (0 at the bottom)."""
        return self.neurons.get((place, row, layer))

    def column(self, place: int, row: int) -> list[Neuron]:
        """The neurons of one column, bottom layer first."""
        return [self.neurons[(place, row, layer)] for layer in range(self.layers)]

    def layer(self, index: int) -> list[Neuron]:
        """Every neuron of one layer, row-major (row 0 first, left to right)."""
        return [self.neurons[(place, row, index)] for row in range(self.rows) for place in range(self.across)]

    def all_neurons(self):
        return self.neurons.values()

    def small_world_connections(self) -> list[Connection]:
        return [c for c in self.connections.values() if c.kind == "small_world"]

    def local_connections(self) -> list[Connection]:
        return [c for c in self.connections.values() if c.kind != "small_world"]

    def connections_of_kind(self, kind: str) -> list[Connection]:
        return [c for c in self.connections.values() if c.kind == kind]

    def mean_out_degree(self) -> float:
        return len(self.connections) / len(self.neurons)

    # --- input and output butter -------------------------------------------------

    def input_row(self) -> list[Neuron]:
        """The input butter: the bottom layer, or with one layer the bottom row (the grid's convention)."""
        if self.layers == 1:
            return [self.neurons[(place, self.rows - 1, 0)] for place in range(self.across)]
        return self.layer(0)

    def output_row(self) -> list[Neuron]:
        """The output butter: the top layer, or with one layer the top row."""
        if self.layers == 1:
            return [self.neurons[(place, 0, 0)] for place in range(self.across)]
        return self.layer(self.layers - 1)

    def input_width(self) -> int:
        return self.across if self.layers == 1 else self.across * self.rows

    def __len__(self) -> int:
        return len(self.neurons)

    def __repr__(self) -> str:
        shape = f"{self.across}x{self.rows}" + (f"x{self.layers} layers" if self.layers > 1 else "")
        return f"HexColumns({shape}, {len(self.neurons)} neurons, {len(self.connections)} connections, omega {self.omega:g})"
