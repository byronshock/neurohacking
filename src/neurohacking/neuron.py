from __future__ import annotations

from .connection import Connection


class Neuron:
    def __init__(self, name: str = "Neuron"):
        self.name = name
        self.position = None  # (q, r) axial coordinates, set by the grid
        self.connections: list[Connection] = []  # shared with the neuron at the other end
        self.has_fired = False

    def connect(self, target: Neuron, connection_id: int = 0) -> Connection:
        """Create an active two-way connection to target.

        The same Connection object is stored on both neurons, so either side can
        switch it off. Returns it so a registry can record it by ID.
        """
        connection = Connection(connection_id, self, target)
        self.connections.append(connection)
        target.connections.append(connection)
        return connection

    def connection_to(self, other: Neuron) -> Connection | None:
        """The connection joining this neuron to `other`, or None if there is none."""
        for connection in self.connections:
            if connection.other(self) is other:
                return connection
        return None

    def neighbours(self) -> list[Neuron]:
        """The neurons at the far end of each connection."""
        return [connection.other(self) for connection in self.connections]

    def activate(self):
        """Receive a signal and pass it on along active connections, once only.

        A neuron that has already fired ignores further signals. Without this
        guard, neighbouring neurons would re-trigger each other forever.
        """
        if self.has_fired:
            return
        self.has_fired = True
        print(f"{self.name} received a signal.")
        for connection in self.connections:
            if connection.is_active:
                connection.other(self).activate()

    def reset(self):
        """Allow this neuron to fire again."""
        self.has_fired = False

    def list_connections(self):
        print(f"I am {self.name}, and I am connected to:")
        for connection in self.connections:
            state = "" if connection.is_active else " (inactive)"
            print(f"    #{connection.id} {connection.other(self).name}{state}")
