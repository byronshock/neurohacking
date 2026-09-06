from __future__ import annotations

import random

from .connection import Connection
from .neuron import Neuron
from .propagation import Wave, propagate

# The six neighbours of a cell in axial coordinates (q, r).
DIRECTIONS = [
    (1, 0), (0, 1), (-1, 1),
    (-1, 0), (0, -1), (1, -1),
]


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

    Cells are stored by axial coordinates, centred so that the middle cell is
    (0, 0). That cell is the origin used by activate_origin().
    """

    def __init__(
        self,
        columns: int = 24,
        rows: int = 20,
        weight: float | None = 1.0,
        threshold: float = 0.25,
        seed: int | None = None,
        omega: float = 0.0,
    ):
        """Build the mesh.

        `weight` is given to every connection; pass None to draw each weight
        independently from a uniform distribution between -1 and 1 instead.
        `threshold` is given to every neuron. `omega` is the proportion of all
        connections that are small-world shortcuts (0 <= omega < 1): after the
        local mesh is built, shortcuts from random neurons to random
        non-neighbours are added until they make up that fraction of the total.
        `seed` makes both the shortcuts and the random weights reproducible.
        """
        if columns < 1 or rows < 1:
            raise ValueError(f"grid needs at least one column and one row, got {columns}x{rows}")
        if not 0.0 <= omega < 1.0:
            raise ValueError(f"omega must be at least 0 and less than 1, got {omega}")
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
        self.directions = DIRECTIONS
        self.create_grid()  # Initialize the grid
        self._add_small_world_connections(omega)
        if weight is None:
            self.randomize_weights()

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
        """Create one one-way Connection from every neuron to each of its neighbours.

        Neighbouring neurons A and B therefore get two connections, A -> B and
        B -> A, each with its own ID (from 1) and weight. Every connection is
        stored in self.connections and on both neurons.
        """
        for neuron in self.neurons.values():
            for neighbor in self.get_neighbors(neuron):
                if neuron.connection_to(neighbor) is None:
                    connection_id = len(self.connections) + 1
                    self.connections[connection_id] = neuron.connect(neighbor, connection_id, weight)

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
        (q1, r1), (q2, r2) = a.position, b.position
        return (q2 - q1, r2 - r1) in DIRECTIONS

    def small_world_connections(self) -> list[Connection]:
        """The shortcut connections added for omega, in ID order."""
        return [c for c in self.connections.values() if c.kind == "small_world"]

    def local_connections(self) -> list[Connection]:
        """The connections between grid neighbours, in ID order."""
        return [c for c in self.connections.values() if c.kind == "local"]

    def randomize_weights(self, low: float = -1.0, high: float = 1.0, seed: int | None = None) -> None:
        """Give every connection its own weight, drawn uniformly between low and high.

        Each direction between two neurons gets an independent draw. Weights are
        assigned in connection-ID order, so the same seed always gives the same mesh.
        With no seed here, the grid's own seeded stream is used.
        """
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
