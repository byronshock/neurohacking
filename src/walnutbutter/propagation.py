"""Signalling on a schedule: a time-ordered queue of signals, processed a wave at a time.

AUTHORITY.md §4. A signal takes one hop (`Neuron.hop()`, the refractory
period over REFRACTORY_HOPS) to travel a connection. Firing a neuron does
not call its neighbours: each active outgoing connection is scheduled to
deliver one hop later, and the schedule is a heap of events by time. A
**wave** is everything scheduled for one moment: the signals arriving, and
the stimulus if an input lands then. Two signals computed to arrive within
the clock's tolerance of each other are the same moment (clock.py), and an
input's exact time anchors the wave it joins.

A wave is processed in two phases:

1. deliver every signal for that time, adding weights to the potentials of
   the targets that are not refractory, then let each touched neuron settle
   (apply its floor to the wave's total);
2. fire every forced neuron that is not refractory, then every touched
   neuron whose potential now meets its threshold and is not refractory.

Because all deliveries finish before any firing decision is made, the result
cannot depend on the order in which neurons or connections are stored. A
neuron may fire as often as its refractory period allows; nothing else
limits it. Each connection stamps the time of the last signal its target
integrated (`Connection.last_signal`): the synapse's only trace.

The schedule outlives an epoch: signals due at or after the horizon the
network runs to wait for the next epoch, where they join the next input's
waves. `propagate()` runs a standalone cascade from a stimulus to
exhaustion, for small experiments and tests.
"""

from __future__ import annotations

import heapq
import itertools
import math
from typing import Callable, Iterable, Mapping

from .clock import before, slack
from .connection import Connection
from .constants import INTERVAL
from .neuron import Neuron

_wave_stamps = itertools.count(1)  # a fresh stamp per wave, across all schedules
EXTERNAL, STIMULUS, SIGNAL = 0, 1, 2  # event kinds, in the order they are taken within a wave


class Signal:
    """A record of one delivery: the connection, the wave it was delivered in, its target and amount.

    Built on request by `Wave.signals()` for inspection and the reinforce
    rule; the schedule itself queues bare connections.
    """

    __slots__ = ("connection", "wave", "target", "amount")

    def __init__(self, connection: Connection, wave: int):
        self.connection = connection
        self.wave = wave
        self.target = connection.target
        self.amount = connection.weight

    def __repr__(self) -> str:
        return f"Signal({self.connection!r}, wave {self.wave})"


class Wave:
    """What happened at one moment: its time, the connections whose signals were delivered, and the neurons that fired."""

    __slots__ = ("number", "time", "delivered", "fired")

    def __init__(
        self, number: int, time: float = 0.0, delivered: list[Connection] | None = None, fired: list[Neuron] | None = None
    ):
        self.number = number
        self.time = time
        self.delivered: list[Connection] = [] if delivered is None else delivered
        self.fired: list[Neuron] = [] if fired is None else fired

    def signals(self) -> list[Signal]:
        """The deliveries of this wave as Signal records."""
        return [Signal(connection, self.number) for connection in self.delivered]

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Wave)
            and (self.number, self.time, self.delivered, self.fired) == (other.number, other.time, other.delivered, other.fired)
        )

    def __repr__(self) -> str:
        return f"Wave(number={self.number}, time={self.time:g}, delivered={self.delivered!r}, fired={self.fired!r})"


class Schedule:
    """The time-ordered queue of everything still to happen: signals in flight, stimuli and external inputs due."""

    def __init__(self):
        self._heap: list[tuple] = []  # (time, kind, sequence, payload)
        self._sequence = itertools.count()

    def __len__(self) -> int:
        return len(self._heap)

    def signal(self, connection: Connection, time: float) -> None:
        """A signal along `connection` arriving at `time`."""
        heapq.heappush(self._heap, (float(time), SIGNAL, next(self._sequence), connection))

    def stimulus(self, neuron: Neuron, time: float) -> None:
        """`neuron` is forced to fire at `time`, refractory period permitting."""
        heapq.heappush(self._heap, (float(time), STIMULUS, next(self._sequence), neuron))

    def external(self, neuron: Neuron, amount: float, time: float) -> None:
        """An external input of `amount` delivered to `neuron` at `time`, which fires it only if it reaches threshold."""
        heapq.heappush(self._heap, (float(time), EXTERNAL, next(self._sequence), (neuron, amount)))

    def next_time(self) -> float | None:
        return self._heap[0][0] if self._heap else None

    def pending(self) -> list[tuple[float, Connection]]:
        """The signals in flight, by time then order: what a checkpoint records."""
        return [(time, payload) for time, kind, _, payload in sorted(self._heap, key=lambda e: e[:3]) if kind == SIGNAL]

    def clear(self) -> None:
        self._heap.clear()

    def run(
        self,
        until: float = math.inf,
        waves: list[Wave] | None = None,
        on_wave: Callable[[Wave], None] | None = None,
    ) -> list[Wave]:
        """Process every wave due before `until`, appending to and returning `waves`.

        `on_wave` is called after each wave has fired, with the wave: this is
        where the refires learn (dopamine.py). Waves are numbered on from the
        length of `waves`.
        """
        waves = [] if waves is None else waves
        hop = Neuron.hop()
        heap = self._heap
        while heap and before(heap[0][0], until):
            first = heap[0][0]
            limit = first + slack(first)
            batch = []
            time = None  # an input's exact time anchors the wave; otherwise the earliest signal's
            while heap and heap[0][0] <= limit:
                event = heapq.heappop(heap)
                batch.append(event)
                if event[1] != SIGNAL and time is None:
                    time = event[0]
            if time is None:
                time = first
            wave = Wave(len(waves), time)
            mark = next(_wave_stamps)
            touched: list[Neuron] = []
            forced: list[Neuron] = []
            for _, kind, _, payload in batch:
                if kind == SIGNAL:
                    wave.delivered.append(payload)
                    target = payload.target
                    if target.receive(payload.weight, time):
                        payload.last_signal = time
                        if target.touched_stamp != mark:  # each neuron once per wave, without a set
                            target.touched_stamp = mark
                            touched.append(target)
                elif kind == STIMULUS:
                    forced.append(payload)
                else:
                    neuron, amount = payload
                    if neuron.receive(amount, time) and neuron.touched_stamp != mark:
                        neuron.touched_stamp = mark
                        touched.append(neuron)
            for neuron in touched:
                neuron.settle()  # the floor applies to the wave's total, whatever order it arrived in
            for neuron in forced:
                if not neuron.refractory_at(time):  # a stimulus listed twice fires once: the first spike makes it refractory
                    self._fire(neuron, wave, hop)
                    neuron.forced = True
            for neuron in touched:
                if neuron.can_fire(time):
                    self._fire(neuron, wave, hop)
            waves.append(wave)
            if on_wave is not None:
                on_wave(wave)
        return waves

    def _fire(self, neuron: Neuron, wave: Wave, hop: float) -> None:
        """Fire one neuron and schedule its outgoing signals one hop later."""
        for connection in neuron.fire(wave.number, wave.time):
            self.signal(connection, wave.time + hop)
        wave.fired.append(neuron)


def propagate(
    fire: Iterable[Neuron] = (),
    inputs: Mapping[Neuron, float] | None = None,
    now: float = 0.0,
    until: float | None = None,
) -> list[Wave]:
    """Run one cascade from a stimulus at clock time `now` and return its waves.

    `fire` lists neurons forced to fire at `now` regardless of threshold (an
    external stimulus); a neuron still refractory at `now` ignores it.
    `inputs` maps neurons to external input amounts delivered at `now`,
    which fire the neuron only if it reaches threshold. Later waves follow
    one hop apart until nothing is left in flight or the clock reaches
    `until` (default: one INTERVAL after `now`). A bound is needed because
    activity can sustain itself: with a refractory period of a few hops, a
    loop brings a neuron's own spike back to refire it, forever.
    """
    schedule = Schedule()
    for neuron in fire:
        schedule.stimulus(neuron, now)
    for neuron, amount in (inputs or {}).items():
        schedule.external(neuron, amount, now)
    return schedule.run(now + INTERVAL if until is None else until)
