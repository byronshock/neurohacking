from __future__ import annotations

import random

from .connection import Connection
from .inputs import complement_code
from .neuron import Neuron
from .propagation import Wave, propagate

# The six neighbours of a cell in axial coordinates (q, r).
DIRECTIONS = [
    (1, 0), (0, 1), (-1, 1),
    (-1, 0), (0, -1), (1, -1),
]

# The twelve cells at hex distance 2: the neighbours of a cell's neighbours,
# other than the cell itself and its own neighbours.
DIRECTIONS2 = [
    (2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -1),
    (-2, 0), (-2, 1), (-2, 2), (-1, 2), (0, 2), (1, 1),
]


def hex_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Number of steps between two axial cells."""
    dq, dr = b[0] - a[0], b[1] - a[1]
    return max(abs(dq), abs(dr), abs(dq + dr))


def offset_to_axial(column: int, row: int) -> tuple[int, int]:
    """Convert a (column, row) position to axial (q, r).

    Rows are laid out "odd-r": every odd row is shifted half a cell to the
    right, which is what lets whole pointy-top hexagons fill a rectangle.
    Python's floor division makes this work for negative rows as well.
    """
    return column - (row - (row & 1)) // 2, row


def axial_to_offset(q: int, r: int) -> tuple[int, int]:
    """The inverse of offset_to_axial."""
    return q + (r - (r & 1)) // 2, r


class GridOfNeurons:
    """A rectangle of `columns` x `rows` hexagonal cells, each holding a Neuron.

    Every neuron is connected to its six neighbours (kind "local") and to the
    twelve neighbours of those neighbours (kind "local2"), one way in each
    direction, plus the small-world shortcuts chosen by omega. Cells are stored
    by axial coordinates, centred so that the middle cell is (0, 0). That cell
    is the origin used by activate_origin().
    """

    def __init__(
        self,
        columns: int = 8,
        rows: int = 10,
        weight: float | None = 1.0,
        threshold: float = 0.25,
        seed: int | None = None,
        omega: float = 0.05,
        permute: bool = True,
        weight_range: tuple[float, float] = (-1.0, 1.0),
    ):
        """Build the mesh.

        `weight` is given to every connection; pass None to draw each weight
        independently and uniformly from `weight_range` instead. The range is
        also what learning clips weights to; (epsilon, 1) keeps them positive.
        `threshold` is given to every neuron. `omega` is the proportion of all
        connections that are small-world shortcuts (0 <= omega < 1): after the
        local mesh is built, shortcuts from random neurons to random
        non-neighbours are added until they make up that fraction of the total.
        `permute` draws a random permutation of the columns, fixed for the life
        of the grid, that scrambles every input pattern onto the bottom row.
        `seed` makes the shortcuts, the random weights, the permutation and the
        random inputs all reproducible.
        """
        if columns < 1 or rows < 1:
            raise ValueError(f"grid needs at least one column and one row, got {columns}x{rows}")
        if not 0.0 <= omega < 1.0:
            raise ValueError(f"omega must be at least 0 and less than 1, got {omega}")
        low, high = weight_range
        if not low < high:
            raise ValueError(f"weight range must run from low to high, got {weight_range}")
        self.weight_range = (float(low), float(high))
        self.columns = columns
        self.rows = rows
        self.weight = weight  # fixed weight for every connection, or None for random
        self.threshold = threshold  # firing threshold given to every neuron
        self.omega = omega
        self.seed = seed
        self._rng = random.Random(seed)  # one stream for shortcuts, then weights
        self.neurons: dict[tuple[int, int], Neuron] = {}  # Maps axial (q, r) to Neuron
        self.connections: dict[int, Connection] = {}  # Maps connection ID (from 1) to Connection
        self.waves: list[Wave] = []  # Waves of the most recent propagation
        self.input_pattern: list[bool] | None = None  # one bit per column, applied to the bottom row
        self.input_bits: list[bool] | None = None  # the raw bits before complement coding
        self.input_coded: list[bool] | None = None  # the complement-coded bits before permutation
        self.permutation: list[int] = list(range(columns))  # bottom-row column i shows coded bit permutation[i]
        self.epoch = 0  # how many inputs have been presented
        self.directions = DIRECTIONS
        self.create_grid()  # Initialize the grid
        self._add_small_world_connections(omega)
        if weight is None:
            self.randomize_weights()
        if permute:
            self._rng.shuffle(self.permutation)

    # --- building ---------------------------------------------------------

    def create_grid(self):
        """Create one neuron per cell of the rectangle, then connect neighbours."""
        centre_column, centre_row = self.columns // 2, self.rows // 2
        for row in range(self.rows):
            for column in range(self.columns):
                q, r = offset_to_axial(column - centre_column, row - centre_row)
                neuron = Neuron(f"Neuron_{q}_{r}", threshold=self.threshold)
                neuron.position = (q, r)
                self.neurons[(q, r)] = neuron

        self._establish_connections(1.0 if self.weight is None else self.weight)

    def _establish_connections(self, weight: float = 1.0):
        """Create one one-way Connection from every neuron to each cell within two steps.

        First every neuron is connected to its six neighbours ("local"), then to
        the twelve neighbours of its neighbours ("local2"). Neurons A and B
        therefore get two connections, A -> B and B -> A, each with its own ID
        (from 1) and weight. Every connection is stored in self.connections and
        on both neurons.
        """
        for neuron in self.neurons.values():
            for neighbor in self.get_neighbors(neuron):
                if neuron.connection_to(neighbor) is None:
                    connection_id = len(self.connections) + 1
                    self.connections[connection_id] = neuron.connect(neighbor, connection_id, weight)
        for neuron in self.neurons.values():
            for neighbor in self.get_second_neighbors(neuron):
                if neuron.connection_to(neighbor) is None:
                    connection_id = len(self.connections) + 1
                    self.connections[connection_id] = neuron.connect(neighbor, connection_id, weight, kind="local2")

    def _add_small_world_connections(self, omega: float) -> None:
        """Add shortcuts until they are the fraction `omega` of all connections.

        With L local connections, S = omega * L / (1 - omega) shortcuts make
        S / (L + S) == omega. Each shortcut runs one way from a random neuron to a
        random neuron that is neither itself, one of its six neighbours, nor a
        target it already connects to.
        """
        local_count = len(self.connections)
        wanted = round(omega * local_count / (1.0 - omega))
        neurons = list(self.neurons.values())
        weight = 1.0 if self.weight is None else self.weight
        attempts_left = 100 * (wanted + 1)  # a tiny mesh may have no valid targets
        while wanted > 0 and attempts_left > 0:
            attempts_left -= 1
            source = self._rng.choice(neurons)
            target = self._rng.choice(neurons)
            if target is source or self._are_neighbours(source, target):
                continue
            if source.connection_to(target) is not None:
                continue
            connection_id = len(self.connections) + 1
            self.connections[connection_id] = source.connect(target, connection_id, weight, kind="small_world")
            wanted -= 1

    def _are_neighbours(self, a: Neuron, b: Neuron) -> bool:
        """True if b is within two steps of a, i.e. already reached by a local connection."""
        return hex_distance(a.position, b.position) <= 2

    def small_world_connections(self) -> list[Connection]:
        """The shortcut connections added for omega, in ID order."""
        return [c for c in self.connections.values() if c.kind == "small_world"]

    def local_connections(self) -> list[Connection]:
        """The connections within the mesh (first and second ring), in ID order."""
        return [c for c in self.connections.values() if c.kind in ("local", "local2")]

    def first_ring_connections(self) -> list[Connection]:
        """Connections to immediate neighbours, in ID order."""
        return [c for c in self.connections.values() if c.kind == "local"]

    def second_ring_connections(self) -> list[Connection]:
        """Connections to neighbours of neighbours, in ID order."""
        return [c for c in self.connections.values() if c.kind == "local2"]

    def randomize_weights(
        self, low: float | None = None, high: float | None = None, seed: int | None = None
    ) -> None:
        """Give every connection its own weight, drawn uniformly between low and high.

        The bounds default to the grid's weight_range. Each direction between two
        neurons gets an independent draw. Weights are assigned in connection-ID
        order, so the same seed always gives the same mesh. With no seed here,
        the grid's own seeded stream is used.
        """
        low = self.weight_range[0] if low is None else low
        high = self.weight_range[1] if high is None else high
        rng = self._rng if seed is None else random.Random(seed)
        for connection in self.connections.values():
            connection.weight = rng.uniform(low, high)

    def clip_weight(self, weight: float) -> float:
        """Keep a weight inside the grid's weight_range."""
        low, high = self.weight_range
        return max(low, min(high, weight))

    # --- lookup -----------------------------------------------------------

    def get_neighbors(self, neuron: Neuron) -> list:
        """Return the list of neighboring neurons in the grid."""
        q, r = neuron.position
        neighbors = []
        for dq, dr in self.directions:
            neighbor_pos = (q + dq, r + dr)
            if neighbor_pos in self.neurons:
                neighbors.append(self.neurons[neighbor_pos])
        return neighbors

    def get_second_neighbors(self, neuron: Neuron) -> list:
        """The neurons two steps away: neighbours of neighbours, excluding the neighbours themselves."""
        q, r = neuron.position
        return [self.neurons[(q + dq, r + dr)] for dq, dr in DIRECTIONS2 if (q + dq, r + dr) in self.neurons]

    def get_neuron(self, q: int, r: int) -> Neuron | None:
        """Retrieve a neuron by axial coordinates."""
        return self.neurons.get((q, r))

    def get_neuron_at(self, column: int, row: int) -> Neuron | None:
        """Retrieve a neuron by (column, row), counted from the top-left cell."""
        q, r = offset_to_axial(column - self.columns // 2, row - self.rows // 2)
        return self.neurons.get((q, r))

    def get_origin_neuron(self) -> Neuron | None:
        """Return the neuron at the centre of the rectangle, axial (0, 0)."""
        return self.neurons.get((0, 0))

    def get_connection(self, connection_id: int) -> Connection | None:
        """Retrieve a connection by its ID."""
        return self.connections.get(connection_id)

    def connection_between(self, source: Neuron, target: Neuron) -> Connection | None:
        """The connection running from `source` to `target`, or None if there is none."""
        return source.connection_to(target)

    # --- input ------------------------------------------------------------

    def input_row(self) -> list[Neuron]:
        """The bottom row of neurons, left to right: the network's input."""
        return [self.get_neuron_at(column, self.rows - 1) for column in range(self.columns)]

    def set_input(self, pattern) -> None:
        """Store the input pattern: one boolean per column of the bottom row."""
        pattern = [bool(b) for b in pattern]
        if len(pattern) != self.columns:
            raise ValueError(f"input pattern has {len(pattern)} bits but the mesh has {self.columns} columns")
        self.input_pattern = pattern

    def set_input_bits(self, bits) -> None:
        """Set the input from raw bits (half the columns): complement-code them, then permute.

        Bottom-row column i receives coded bit permutation[i]. With the identity
        permutation (permute=False) the coded bits land in order.
        """
        if self.columns % 2:
            raise ValueError(f"complement coding needs an even number of columns, got {self.columns}")
        bits = [bool(b) for b in bits]
        if len(bits) != self.columns // 2:
            raise ValueError(f"expected {self.columns // 2} input bits for {self.columns} columns, got {len(bits)}")
        coded = complement_code(bits)
        self.set_input([coded[i] for i in self.permutation])
        self.input_bits = bits
        self.input_coded = coded

    def new_random_input(self) -> list[bool]:
        """Draw fresh raw bits from the grid's seeded stream and set them as the input.

        Because the stream is the same one used for shortcuts and weights, a
        seed reproduces the whole sequence of inputs, not just the first.
        """
        if self.columns % 2:
            raise ValueError(f"complement coding needs an even number of columns, got {self.columns}")
        bits = [self._rng.random() < 0.5 for _ in range(self.columns // 2)]
        self.set_input_bits(bits)
        return bits

    def input_neurons(self) -> list[Neuron]:
        """The bottom-row neurons whose input bit is 1 (empty if no pattern is set)."""
        if self.input_pattern is None:
            return []
        return [neuron for neuron, bit in zip(self.input_row(), self.input_pattern) if bit]

    def fire_input(self) -> list[Wave]:
        """Force the input neurons to fire and propagate the signal wave by wave."""
        if self.input_pattern is None:
            raise ValueError("no input pattern set; call set_input() first")
        self.epoch += 1
        return self.propagate(fire=self.input_neurons())

    # --- running ----------------------------------------------------------

    def activate_origin(self) -> list[Wave]:
        """Fire the origin neuron and propagate the signal wave by wave."""
        origin = self.get_origin_neuron()
        if origin:
            print(f"\nOrigin neuron {origin.name} has {len(origin.outgoing)} connections")
            return self.propagate(fire=[origin])
        return []

    def propagate(self, fire=(), inputs=None) -> list[Wave]:
        """Run one epoch from the given stimulus (see propagation.propagate) and keep its waves."""
        self.waves = propagate(fire=fire, inputs=inputs)
        return self.waves

    def reset(self):
        """Clear every neuron's fired state and potential so a signal can be sent again."""
        for neuron in self.neurons.values():
            neuron.reset()
        self.waves = []

    def fired_neurons(self) -> list:
        """Return the neurons that have fired since the last reset."""
        return [n for n in self.neurons.values() if n.has_fired]
