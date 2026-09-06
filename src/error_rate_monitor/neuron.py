class Neuron:
    def __init__(self, name: str = "Neuron"):
        self.name = name
        self.connections = []  # List of (target_neuron, is_active) tuples

    def connect(self, target: "Neuron"):
        """Connect this neuron to the target neuron with an active pathway."""
        self.connections.append((target, True))  # Store as (target, is_active)

    def activate(self):
        """Trigger signal propagation to all connected neurons with active pathways."""
        print(f"{self.name} received a signal.")
        for connection in self.connections:
            target_neuron, is_active = connection
            if is_active:
                connection = (target_neuron, False)
                target_neuron.activate()
                """Simulate receiving a signal (for demonstration purposes)."""
                print(f"{self.name} received a signal.")

    def list_connections(self):
        print(f"I am {self.name}, and I am connected to:")
        for connection in self.connections:
            target_neuron, _is_active = connection
            print(f"    {target_neuron.name}")
