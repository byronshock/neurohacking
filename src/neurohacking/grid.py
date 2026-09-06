from __future__ import annotations

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

    def __init__(self, columns: int = 24, rows: int = 20, weight: float = 1.0, threshold: float = 1.0):
        if columns < 1 or rows < 1:
            raise ValueError(f"grid needs at least one column and one row, got {columns}x{rows}")
        self.columns = columns
        self.rows = rows
        self.weight = weight  # weight given to every connection
        self.threshold = threshold  # firing threshold given to every neuron
        self.neurons: dict[tuple[int, int], Neuron] = {}  # Maps axial (q, r) to Neuron
        self.connections: dict[int, Connection] = {}  # Maps connection ID (from 1) to Connection
        self.waves: list[Wave] = []  # Waves of the most recent propagation
        self.directions = DIRECTIONS
        self.create_grid()  # Initialize the grid

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

        self._establish_connections(self.weight)  # Call once after grid is fully initialized

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
