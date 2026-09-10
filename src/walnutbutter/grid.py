from __future__ import annotations

import random

from .connection import Connection
from .network import Network
from .neuron import Neuron
from .propagation import Wave

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


def offset_to_axial(place: int, row: int) -> tuple[int, int]:
    """Convert a (place, row) position to axial (q, r).

    Rows are laid out "odd-r": every odd row is shifted half a cell to the
    right, which is what lets whole pointy-top hexagons fill a rectangle.
    Python's floor division makes this work for negative rows as well.
    """
    return place - (row - (row & 1)) // 2, row


def axial_to_offset(q: int, r: int) -> tuple[int, int]:
    """The inverse of offset_to_axial."""
    return q + (r - (r & 1)) // 2, r


class GridOfNeurons(Network):
    """A rectangle of `across` x `rows` hexagonal cells, each holding a Neuron.

    Every neuron is connected to its six neighbours (kind "local") and to the
    twelve neighbours of those neighbours (kind "local2"), one way in each
    direction, plus the small-world shortcuts chosen by omega. Cells are stored
    by axial coordinates, centred so that the middle cell is (0, 0). That cell
    is the origin used by activate_origin().
    """

    def __init__(
        self,
        across: int = 8,
        rows: int = 10,
        weight: float | None = 1.0,
        threshold: float = 0.25,
        seed: int | None = None,
        omega: float = 0.2,
        permute: bool = True,
        weight_range: tuple[float, float] = (-1.0, 1.0),
        minimum_potential: float = -1.0,
    ):
        """Build the mesh.

        `weight` is given to every connection; pass None to draw each weight
        independently and uniformly from `weight_range` instead. The range is
        also what learning clips weights to; (epsilon, 1) keeps them positive.
        `threshold` is given to every neuron. `omega` is the proportion of all
        connections that are small-world shortcuts (0 <= omega < 1): after the
        local mesh is built, shortcuts from random neurons to random
        non-neighbours are added until they make up that fraction of the total.
        `permute` draws a random permutation of the places across, fixed for the life
        of the grid, that scrambles every input pattern onto the bottom row.
        `seed` makes the shortcuts, the random weights, the permutation and the
        random inputs all reproducible.
        """
        if across < 1 or rows < 1:
            raise ValueError(f"grid needs at least one neuron across and one row, got {across}x{rows}")
        if not 0.0 <= omega < 1.0:
            raise ValueError(f"omega must be at least 0 and less than 1, got {omega}")
        self.across = across
        self.rows = rows
        self.weight = weight  # fixed weight for every connection, or None for random
        self.threshold = threshold  # firing threshold given to every neuron
        self.minimum_potential = minimum_potential  # floor on every neuron's potential
        self.omega = omega
        self.seed = seed
        self._rng = random.Random(seed)  # one stream for shortcuts, then weights
        self.neurons: dict[tuple[int, int], Neuron] = {}  # Maps axial (q, r) to Neuron
        self.connections: dict[int, Connection] = {}  # Maps connection ID (from 1) to Connection
        self._init_network(across, weight_range)
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
        centre_column, centre_row = self.across // 2, self.rows // 2
        for row in range(self.rows):
            for place in range(self.across):
                q, r = offset_to_axial(place - centre_column, row - centre_row)
                neuron = Neuron(f"Neuron_{q}_{r}", threshold=self.threshold, minimum_potential=self.minimum_potential)
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

    def get_neuron_at(self, place: int, row: int) -> Neuron | None:
        """Retrieve a neuron by (place, row), counted from the top-left cell."""
        q, r = offset_to_axial(place - self.across // 2, row - self.rows // 2)
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

    def all_neurons(self):
        return self.neurons.values()

    # --- running ----------------------------------------------------------

    def activate_origin(self) -> list[Wave]:
        """Fire the origin neuron and propagate the signal wave by wave."""
        origin = self.get_origin_neuron()
        if origin:
            print(f"\nOrigin neuron {origin.name} has {len(origin.outgoing)} connections")
            return self.propagate(fire=[origin])
        return []
