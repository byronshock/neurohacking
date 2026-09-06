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

    def create_grid(self):
        """Generate a hexagonal grid of neurons within the given size."""
        for q in range(-self.size, self.size + 1):
            for r in range(-self.size, self.size + 1):
                if abs(q) + abs(r) <= self.size:
                    neuron = Neuron(f"Neuron_{q}_{r}")
                    neuron.position = (q, r)
                    self.neurons[(q, r)] = neuron

        # Establish connections between neurons
        for neuron in self.neurons.values():
            neighbors = self.get_neighbors(neuron)
            for neighbor in neighbors:
                neuron.connect(neighbor)

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
