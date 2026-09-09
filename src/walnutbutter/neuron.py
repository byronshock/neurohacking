from __future__ import annotations

from .connection import Connection


class Neuron:
    verbose = False  # class-wide: print a line each time any neuron fires (off unless asked: walnutbutter -v)

    def __init__(self, name: str = "Neuron", threshold: float = 0.25, minimum_potential: float = -1.0):
        self.name = name
        self.position = None  # (q, r) axial coordinates, set by the grid
        self.outgoing: list[Connection] = []  # connections this neuron sends signals along
        self.incoming: list[Connection] = []  # connections that deliver signals to this neuron
        self.threshold = float(threshold)  # total weighted input needed to fire
        self.minimum_potential = float(minimum_potential)  # inhibition can push the potential no lower than this
        self.potential = 0.0  # weighted input received since the last reset
        self.noise = 0.0  # exploration noise this epoch started with (see learning.py)
        self.touched_stamp = 0  # last wave (a global stamp) in which a signal reached this neuron
        self.rate = 0.5  # running estimate of how often this neuron fires per epoch (see learning.py)
        self.has_fired = False
        self.fired_in_wave: int | None = None  # set by fire(); None until it fires

    def connect(
        self, target: Neuron, connection_id: int = 0, weight: float = 1.0, kind: str = "local"
    ) -> Connection:
        """Create a one-way connection from this neuron to `target`.

        The Connection is stored in this neuron's outgoing list and the target's
        incoming list. Returns it so a registry can record it by ID.
        """
        connection = Connection(connection_id, self, target, weight, kind=kind)
        self.outgoing.append(connection)
        target.incoming.append(connection)
        return connection

    def connection_to(self, target: Neuron) -> Connection | None:
        """The outgoing connection to `target`, or None if there is none."""
        for connection in self.outgoing:
            if connection.target is target:
                return connection
        return None

    def targets(self) -> list[Neuron]:
        """The neurons this neuron can send a signal to."""
        return [connection.target for connection in self.outgoing]

    def sources(self) -> list[Neuron]:
        """The neurons that can send a signal to this neuron."""
        return [connection.source for connection in self.incoming]

    def receive(self, amount: float) -> None:
        """Take in weighted input. A neuron that has already fired ignores it.

        Receiving never fires the neuron by itself; the propagation loop checks
        `ready` once every signal in the wave has been delivered. Negative
        weights push the potential down (an inhibitory connection).
        """
        if self.has_fired:
            return
        self.potential += amount
        if self.potential < self.minimum_potential:  # inhibition saturates at the floor
            self.potential = self.minimum_potential

    @property
    def ready(self) -> bool:
        """True if this neuron has enough input to fire and has not fired yet."""
        return not self.has_fired and self.potential >= self.threshold

    def fire(self, wave: int = 0) -> list[Connection]:
        """Mark this neuron as fired in `wave` and return the connections to signal along.

        This does not deliver anything: the caller (see propagation.py) queues
        the returned connections so that all of a wave's signals are delivered
        before any neuron in the next wave decides whether to fire.
        """
        self.has_fired = True
        self.fired_in_wave = wave
        if Neuron.verbose:
            print(f"{self.name} fired in wave {wave}.")
        return [connection for connection in self.outgoing if connection.is_active]

    def reset(self, discharge: bool = True) -> None:
        """Allow this neuron to fire again and, by default, clear its potential.

        With `discharge=False` only a neuron that fired is cleared; one that did
        not fire keeps the sub-threshold input it has accumulated, so charge
        carries over from epoch to epoch until it eventually fires (an optional
        mode; see run_epoch's `discharge` and the --carry-over flag).
        """
        if self.has_fired or discharge:
            self.potential = 0.0
        self.has_fired = False
        self.fired_in_wave = None
        self.noise = 0.0

    def list_connections(self) -> None:
        print(f"I am {self.name}, and I send signals to:")
        for connection in self.outgoing:
            state = "" if connection.is_active else " (inactive)"
            print(f"    #{connection.id} {connection.target.name}, weight {connection.weight:g}{state}")
