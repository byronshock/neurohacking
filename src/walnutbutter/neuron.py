from __future__ import annotations

import math

from .connection import Connection


class Neuron:
    """A leaky integrate-and-fire neuron with an absolute refractory period, on a clock in nominal milliseconds.

    The leak is lazy: nothing happens to a quiet neuron. When a signal arrives
    at time `now`, the potential is first decayed for the time since it was
    last brought up to date (`potential *= exp(-(now - last_update) / tau)`),
    then the signal is added. A neuron that fired within the last
    `refractory` milliseconds ignores every signal, forced stimulus included.
    `tau` and `refractory` are global properties of neurons, one value for the
    whole network. Propagation is instantaneous, so the clock only advances
    between inputs.
    """

    verbose = False  # class-wide: print a line each time any neuron fires (off unless asked: walnutbutter -v)
    tau = 5.0  # leak time constant, nominal milliseconds; math.inf switches the leak off
    refractory = 5.0  # absolute refractory period, nominal milliseconds

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
        self.has_fired = False  # fired in the current cascade
        self.fired_in_wave: int | None = None  # set by fire(); None until it fires
        self.fired_at: float | None = None  # clock time of the last spike, across cascades
        self.last_update = 0.0  # clock time the potential was last brought up to date

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

    def refractory_at(self, now: float) -> bool:
        """True if the neuron fired within the refractory period before `now` (a neuron that fired *at* now included)."""
        return self.fired_at is not None and now < self.fired_at + Neuron.refractory

    def leak(self, now: float) -> None:
        """Bring the potential up to `now`: decay it for the time since the last update. Lazy, so call it on arrival."""
        elapsed = now - self.last_update
        if elapsed > 0.0:
            if Neuron.tau != math.inf:
                self.potential *= math.exp(-elapsed / Neuron.tau)
            self.last_update = now

    def receive(self, amount: float, now: float | None = None) -> None:
        """Take in weighted input at time `now`: leak first, then integrate.

        A neuron that has already fired in this cascade, or is still refractory
        from an earlier one, ignores it (forced stimulus included). Receiving
        never fires the neuron by itself; the propagation loop checks
        readiness once every signal in the wave has been delivered, after
        `settle()` has applied the floor to the wave's total. Negative weights
        push the potential down (an inhibitory connection). Without `now` the
        clock is not consulted: no leak, and only `has_fired` blocks.
        """
        if self.has_fired:
            return
        if now is not None:
            if self.refractory_at(now):
                return
            self.leak(now)
        self.potential += amount

    def settle(self) -> None:
        """Apply the floor: inhibition saturates at `minimum_potential`.

        Called once per wave on every neuron that received a signal, so the
        floor acts on the wave's summed input and the result does not depend
        on the order the signals arrived in.
        """
        if self.potential < self.minimum_potential:
            self.potential = self.minimum_potential

    @property
    def ready(self) -> bool:
        """True if this neuron has enough input to fire and has not fired in this cascade."""
        return not self.has_fired and self.potential >= self.threshold

    def can_fire(self, now: float | None) -> bool:
        """Ready, and not refractory at `now` (the clock is ignored when `now` is None)."""
        if now is not None and self.refractory_at(now):
            return False
        return self.ready

    def fire(self, wave: int = 0, now: float | None = None) -> list[Connection]:
        """Mark this neuron as fired in `wave` at time `now` and return the connections to signal along.

        This does not deliver anything: the caller (see propagation.py) queues
        the returned connections so that all of a wave's signals are delivered
        before any neuron in the next wave decides whether to fire.
        """
        self.has_fired = True
        self.fired_in_wave = wave
        self.potential = 0.0  # the spike resets the potential
        if now is not None:
            self.fired_at = now
            self.last_update = now
        if Neuron.verbose:
            print(f"{self.name} fired in wave {wave}.")
        return [connection for connection in self.outgoing if connection.is_active]

    def reset(self, discharge: bool = False) -> None:
        """Start a new cascade: allow the neuron to fire again.

        A neuron that did not fire keeps its sub-threshold potential, which the
        leak then erodes over the time to the next input (a spike has already
        reset the potential of one that fired). With `discharge=True` every
        potential is zeroed instead, the old epoch-by-epoch behaviour.
        `fired_at` is kept: the refractory period outlives the cascade.
        """
        if discharge:
            self.potential = 0.0
        self.has_fired = False
        self.fired_in_wave = None
        self.noise = 0.0

    def list_connections(self) -> None:
        print(f"I am {self.name}, and I send signals to:")
        for connection in self.outgoing:
            state = "" if connection.is_active else " (inactive)"
            print(f"    #{connection.id} {connection.target.name}, weight {connection.weight:g}{state}")
