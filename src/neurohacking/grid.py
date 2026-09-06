from __future__ import annotations

from .connection import Connection
from .neuron import Neuron
from .propagation import Wave, propagate


class GridOfNeurons:
    def __init__(self, size: int = 5, weight: float = 1.0, threshold: float = 1.0):
        self.size = size
        self.weight = weight  # weight given to every connection
        self.threshold = threshold  # firing threshold given to every neuron
        self.neurons = {}  # Maps (q, r) axial coordinates to Neuron instances
        self.connections: dict[int, Connection] = {}  # Maps connection ID (from 1) to Connection
        self.waves: list[Wave] = []  # Waves of the most recent propagation
        self.directions = [
            (1, 0), (0, 1), (-1, 1),
            (-1, 0), (0, -1), (1, -1)
        ]
        self.create_grid()  # Initialize the grid

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

    def create_grid(self):
        """Generate a hexagonal grid of neurons within the given size."""
        for q in range(-self.size, self.size + 1):
            for r in range(-self.size, self.size + 1):
                # Hexagon in axial coordinates: |q|, |r| and |q + r| all within size.
                if max(abs(q), abs(r), abs(q + r)) <= self.size:
                    neuron = Neuron(f"Neuron_{q}_{r}", threshold=self.threshold)
                    neuron.position = (q, r)
                    self.neurons[(q, r)] = neuron

        self._establish_connections(self.weight)  # Call once after grid is fully initialized

    def get_neighbors(self, neuron: "Neuron") -> list:
        """Return the list of neighboring neurons in the grid."""
        q, r = neuron.position
        neighbors = []
        for dq, dr in self.directions:
            neighbor_pos = (q + dq, r + dr)
            if neighbor_pos in self.neurons:
                neighbors.append(self.neurons[neighbor_pos])
        return neighbors

    def get_neuron(self, q: int, r: int) -> "Neuron":
        """Retrieve a neuron at specific axial coordinates."""
        return self.neurons.get((q, r))

    def get_connection(self, connection_id: int) -> Connection | None:
        """Retrieve a connection by its ID."""
        return self.connections.get(connection_id)

    def connection_between(self, source: Neuron, target: Neuron) -> Connection | None:
        """The connection running from `source` to `target`, or None if there is none."""
        return source.connection_to(target)

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

    def get_origin_neuron(self):
        """Return the neuron at the origin (0, 0)."""
        return self.neurons.get((0, 0))

    def reset(self):
        """Clear every neuron's fired state and potential so a signal can be sent again."""
        for neuron in self.neurons.values():
            neuron.reset()
        self.waves = []

    def fired_neurons(self) -> list:
        """Return the neurons that have fired since the last reset."""
        return [n for n in self.neurons.values() if n.has_fired]
