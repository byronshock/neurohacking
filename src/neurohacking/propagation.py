"""Wave-by-wave signal propagation driven by a message queue.

Firing a neuron does not call its neighbours, so there is no recursion and no
recursion-depth limit. Instead each active outgoing connection becomes a
Signal in a first-in-first-out queue, tagged with the wave it belongs to.

A wave is processed in two phases:

1. deliver every signal for that wave, adding weights to target potentials;
2. fire every neuron whose potential now meets its threshold.

Because all deliveries finish before any firing decision is made, the result
cannot depend on the order in which neurons or connections are stored. The
signals produced by the neurons that fire in wave n are delivered in wave n+1.
Wave 0 is the external stimulus that starts the epoch.
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Mapping

from .connection import Connection
from .neuron import Neuron

_wave_stamps = itertools.count(1)  # a fresh stamp per wave, across all propagations


class Signal:
    """One message in the queue: the connection's weight, to be delivered in `wave`.

    The target and amount are copied from the connection when the signal is
    created, so delivering it costs plain attribute reads.
    """

    __slots__ = ("connection", "wave", "target", "amount")

    def __init__(self, connection: Connection, wave: int):
        self.connection = connection
        self.wave = wave
        self.target = connection.target
        self.amount = connection.weight

    def __repr__(self) -> str:
        return f"Signal({self.connection!r}, wave {self.wave})"


@dataclass
class Wave:
    """What happened in one wave: the signals delivered and the neurons that fired."""

    number: int
    delivered: list[Signal] = field(default_factory=list)
    fired: list[Neuron] = field(default_factory=list)


def propagate(
    fire: Iterable[Neuron] = (),
    inputs: Mapping[Neuron, float] | None = None,
) -> list[Wave]:
    """Run one epoch and return its waves.

    `fire` lists neurons forced to fire in wave 0 regardless of threshold (an
    external stimulus). `inputs` maps neurons to external input amounts, which
    are delivered in wave 0 and fire the neuron only if it reaches threshold.
    """
    queue: deque[Signal] = deque()
    waves: list[Wave] = []

    # Wave 0: the external stimulus. Deliver all inputs first, then fire.
    wave = Wave(number=0)
    touched: list[Neuron] = []
    for neuron, amount in (inputs or {}).items():
        neuron.receive(amount)
        touched.append(neuron)
    for neuron in fire:
        if not neuron.has_fired:
            _fire(neuron, wave, queue)
    _fire_ready(touched, wave, queue)
    waves.append(wave)

    # Later waves: drain the queue one wave at a time.
    while queue:
        wave = Wave(number=len(waves))
        stamp = next(_wave_stamps)
        touched = []
        delivered = wave.delivered
        popleft = queue.popleft
        while queue and queue[0].wave == wave.number:
            signal = popleft()
            target = signal.target
            target.receive(signal.amount)
            delivered.append(signal)
            if target.touched_stamp != stamp:  # each neuron once per wave, without a set
                target.touched_stamp = stamp
                touched.append(target)
        _fire_ready(touched, wave, queue)
        waves.append(wave)

    return waves


def _fire_ready(candidates: Iterable[Neuron], wave: Wave, queue: deque[Signal]) -> None:
    """Fire each candidate whose potential has reached its threshold.

    Candidates from a wave are already unique; the stimulus list in wave 0 may
    repeat a neuron, which is harmless because a fired neuron is never ready.
    """
    for neuron in candidates:
        if neuron.ready:
            _fire(neuron, wave, queue)


def _fire(neuron: Neuron, wave: Wave, queue: deque[Signal]) -> None:
    """Fire one neuron and queue its outgoing signals for the next wave."""
    for connection in neuron.fire(wave.number):
        queue.append(Signal(connection, wave.number + 1))
    wave.fired.append(neuron)
