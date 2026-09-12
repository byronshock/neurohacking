from __future__ import annotations

from .clock import slack
from .connection import Connection
from .constants import MINIMUM_POTENTIAL, REFRACTORY, REFRACTORY_HOPS, THRESHOLD


class Neuron:
    """An integrate-and-fire neuron with an absolute refractory period, on a clock in nominal milliseconds.

    AUTHORITY.md §0 and §5: the neuron integrates delta functions (a signal
    adds its weight to the potential, and nothing else ever changes it: no
    leak, not even a lazy one), fires when the potential reaches its
    threshold, and resets. A neuron that fired within the last `refractory`
    milliseconds ignores every signal, forced stimulus included. It
    remembers its last two spike times: the gap between them is what its
    dopamine release depends on (dopamine.py). `refractory` and
    `refractory_hops` are global properties of neurons, one value for the
    whole network; a signal takes `hop()` milliseconds to travel a connection.
    """

    verbose = False  # class-wide: print a line each time any neuron fires (off unless asked: walnutbutter -v)
    refractory = REFRACTORY  # absolute refractory period, nominal milliseconds (see constants.py)
    refractory_hops = REFRACTORY_HOPS  # refractory period / hop time (see constants.py)

    @classmethod
    def hop(cls) -> float:
        """The time a signal takes to travel one connection: refractory / refractory_hops."""
        return cls.refractory / cls.refractory_hops

    def __init__(self, name: str = "Neuron", threshold: float = THRESHOLD, minimum_potential: float = MINIMUM_POTENTIAL):
        self.name = name
        self.position = None  # (q, r) axial coordinates, set by the grid
        self.outgoing: list[Connection] = []  # connections this neuron sends signals along
        self.incoming: list[Connection] = []  # connections that deliver signals to this neuron
        self.threshold = float(threshold)  # total weighted input needed to fire
        self.minimum_potential = float(minimum_potential)  # inhibition can push the potential no lower than this
        self.potential = 0.0  # weighted input received since the last spike
        self.noise = 0.0  # exploration noise added at the last input (see learning.py)
        self.touched_stamp = 0  # last wave (a global stamp) in which a signal reached this neuron
        self.rate = 0.5  # running estimate of how often this neuron fires per epoch (the reinforce rule)
        self.has_fired = False  # fired in the current epoch
        self.fired_in_wave: int | None = None  # the wave of the current epoch it (last) fired in; None until it fires
        self.forced = False  # forced to fire by the stimulus in the current epoch
        self.fired_at: float | None = None  # clock time of the last spike, across epochs
        self.previous_fired_at: float | None = None  # clock time of the spike before that
        self.spikes = 0  # how many times this neuron has ever fired

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
        """True if the neuron fired within the refractory period before `now` (a neuron that fired *at* now included).

        The clock's slack applies: a spike returning the instant the period
        ends, give or take a rounding error, finds the neuron recovered.
        """
        return self.fired_at is not None and now + slack(now) < self.fired_at + Neuron.refractory

    def receive(self, amount: float, now: float | None = None) -> bool:
        """Take in weighted input at time `now`: integrate it, unless refractory. Returns whether it was integrated.

        Receiving never fires the neuron by itself; the schedule checks
        readiness once every signal in the wave has been delivered, after
        `settle()` has applied the floor to the wave's total. Negative weights
        push the potential down (an inhibitory connection). Without `now` the
        clock is not consulted and nothing blocks.
        """
        if now is not None and self.refractory_at(now):
            return False
        self.potential += amount
        return True

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
        """True if this neuron has enough input to fire."""
        return self.potential >= self.threshold

    def can_fire(self, now: float | None) -> bool:
        """Ready, and not refractory at `now` (the clock is ignored when `now` is None)."""
        if now is not None and self.refractory_at(now):
            return False
        return self.ready

    def fire(self, wave: int = 0, now: float | None = None) -> list[Connection]:
        """Spike: mark this neuron as fired in `wave` at time `now` and return the connections to signal along.

        This does not deliver anything: the caller (see propagation.py)
        schedules the returned connections one hop later, so that all of a
        wave's signals are delivered before any neuron in the next wave
        decides whether to fire.
        """
        self.has_fired = True
        self.fired_in_wave = wave
        self.potential = 0.0  # the spike resets the potential
        self.spikes += 1
        if now is not None:
            self.previous_fired_at = self.fired_at
            self.fired_at = now
        if Neuron.verbose:
            print(f"{self.name} fired in wave {wave}.")
        return [connection for connection in self.outgoing if connection.is_active]

    def reset(self, discharge: bool = False) -> None:
        """Start a new epoch: clear the fired-this-epoch state.

        The potential is kept: a neuron that did not fire keeps its
        sub-threshold charge for as long as it takes (there is no leak), and
        a spike has already reset the potential of one that fired. With
        `discharge=True` every potential is zeroed instead. Spike times are
        kept: the refractory period outlives the epoch.
        """
        if discharge:
            self.potential = 0.0
        self.has_fired = False
        self.fired_in_wave = None
        self.forced = False
        self.noise = 0.0

    def list_connections(self) -> None:
        print(f"I am {self.name}, and I send signals to:")
        for connection in self.outgoing:
            state = "" if connection.is_active else " (inactive)"
            print(f"    #{connection.id} {connection.target.name}, weight {connection.weight:g}{state}")
