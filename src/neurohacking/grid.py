from .neuron import Neuron

class GridOfNeurons:
    def __init__(self, size: int = 5):
        self.size = size
        self.neurons = {}  # Maps (q, r) axial coordinates to Neuron instances
        self.directions = [
            (1, 0), (0, 1), (-1, 1),
            (-1, 0), (0, -1), (1, -1)
        ]
        self.create_grid()  # Initialize the grid

    def _establish_connections(self):
        """Establish connections between neurons without recursive references."""
        visited = set()  # Track neurons already processed
        for neuron in self.neurons.values():
            if neuron not in visited:
                neighbors = self.get_neighbors(neuron)
                for neighbor in neighbors:
                    neuron.connect(neighbor)
                visited.add(neuron)

    def create_grid(self):
        """Generate a hexagonal grid of neurons within the given size."""
        for q in range(-self.size, self.size + 1):
            for r in range(-self.size, self.size + 1):
                # Hexagon in axial coordinates: |q|, |r| and |q + r| all within size.
                if max(abs(q), abs(r), abs(q + r)) <= self.size:
                    neuron = Neuron(f"Neuron_{q}_{r}")
                    neuron.position = (q, r)
                    self.neurons[(q, r)] = neuron

        self._establish_connections()  # Call once after grid is fully initialized
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

    def activate_origin(self):
        """Activate the origin neuron to start signal propagation."""
        origin = self.get_origin_neuron()
        if origin:
            print(f"\nOrigin neuron {origin.name} has {len(origin.connections)} connections")
            origin.activate()

    def get_origin_neuron(self):
        """Return the neuron at the origin (0, 0)."""
        return self.neurons.get((0, 0))

    def reset(self):
        """Clear every neuron's has_fired flag so a signal can be sent again."""
        for neuron in self.neurons.values():
            neuron.reset()

    def fired_neurons(self) -> list:
        """Return the neurons that have fired since the last reset."""
        return [n for n in self.neurons.values() if n.has_fired]
