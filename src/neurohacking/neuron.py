class Neuron:
    def __init__(self, name: str = "Neuron"):
        self.name = name
        self.position = None  # (q, r) axial coordinates, set by the grid
        self.connections = []  # List of (target_neuron, is_active) tuples
        self.has_fired = False

    def connect(self, target: "Neuron"):
        """Connect this neuron to the target neuron with an active pathway."""
        self.connections.append((target, True))  # Store as (target, is_active)

    def activate(self):
        """Receive a signal and pass it on to connected neurons, once only.

        A neuron that has already fired ignores further signals. Without this
        guard, neighbouring neurons would re-trigger each other forever.
        """
        if self.has_fired:
            return
        self.has_fired = True
        print(f"{self.name} received a signal.")
        for target_neuron, is_active in self.connections:
            if is_active:
                target_neuron.activate()

    def reset(self):
        """Allow this neuron to fire again."""
        self.has_fired = False

    def list_connections(self):
        print(f"I am {self.name}, and I am connected to:")
        for connection in self.connections:
            target_neuron, _is_active = connection
            print(f"    {target_neuron.name}")
