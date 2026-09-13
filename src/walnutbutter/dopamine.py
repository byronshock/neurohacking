"""Dopamine: the reward, produced locally by refiring neurons and consumed globally (AUTHORITY.md §0, §6).

A neuron whose previous spike was at t_prev and which fires again at t
releases according to the gamma density of its delay d = t - t_prev -
refractory past the end of its refractory period, d^(alpha-1) exp(-d/theta)
/ (Gamma(alpha) theta^alpha), averaged over the hop that follows the delay
(the schedule's own resolution): the gamma mass in [d, d + hop) over hop.
That is the density itself as the hop shrinks, and it stays finite for
alpha < 1, where the density is infinite at zero delay and an instant
refire is the common case. With alpha 2 and theta 1 ms the release is
about 0.3 for an instant refire, peaks a little later, then decays. The
releases pool into one global value that decays lazily with `tau` and is
read without being depleted. The network's expectation of it is an
exponential moving average of the value with time constant
`expectation_tau` (10 minutes by default), starting from 0, brought up to
date at every wave (Byron, September 12, 2026; §6.3). The reinforcement is
proportional to the value minus the expectation.

Learning happens at the refire, with the wave: the refires in a wave
release together, then every incoming connection of each refiring neuron
that carried a signal it integrated since its previous spike moves by
lr * (value - expectation) * release (§6.4-6.6). `order` says whether the
wave's releases join the pool before the value is read (release-first,
default) or after (update-first). With `punish` (default on), an input
neuron whose bit this epoch is 0, one that should not fire, has the sign of
its update reversed when it refires: above-expected dopamine punishes it
for firing instead of rewarding it (Byron, September 12, 2026), and
`punish_gain` times as hard as a reward. Its release is unchanged; hidden
and forced neurons are unchanged. `decay` is the fraction every weight
moves toward zero each epoch: synapses that forget on their own, so that
sustained activity has to be earned (AUTHORITY.md §6.8). The object engine's update is `learn()`
below; the array engine does the same arithmetic on vectors (arrays.py).
"""

from __future__ import annotations

import math

from .constants import (
    DOPAMINE_EXPECTATION_START, DOPAMINE_EXPECTATION_TAU, DOPAMINE_ORDER, DOPAMINE_PUNISH, DOPAMINE_PUNISH_GAIN,
    DOPAMINE_RELEASE_ALPHA, DOPAMINE_RELEASE_THETA, DOPAMINE_TAU, LR, WEIGHT_DECAY,
)
from .neuron import Neuron

ORDERS = ("release-first", "update-first")


def gamma_cdf(a: float, x: float) -> float:
    """The regularised lower incomplete gamma P(a, x): the mass of a gamma(a, 1) distribution below x.

    Pure Python (series below a + 1, a continued fraction above; Numerical
    Recipes' gser and gcf), so both engines compute the same bits and the
    object engine needs no scipy.
    """
    if x <= 0.0:
        return 0.0
    log_prefactor = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1.0:
        term = total = 1.0 / a
        ap = a
        for _ in range(1000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-16:
                break
        return total * math.exp(log_prefactor)
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return 1.0 - math.exp(log_prefactor) * h


class Dopamine:
    """The global dopamine value, its expectation, and the arithmetic of a wave of refires."""

    def __init__(
        self,
        tau: float = DOPAMINE_TAU,
        release_alpha: float = DOPAMINE_RELEASE_ALPHA,
        release_theta: float = DOPAMINE_RELEASE_THETA,
        order: str = DOPAMINE_ORDER,
        lr: float = LR,
        expectation_tau: float = DOPAMINE_EXPECTATION_TAU,
        punish: bool = DOPAMINE_PUNISH,
        punish_gain: float = DOPAMINE_PUNISH_GAIN,
        decay: float = WEIGHT_DECAY,
        expectation_start: float = DOPAMINE_EXPECTATION_START,
    ):
        if punish_gain < 0 or not 0.0 <= decay < 1.0:
            raise ValueError(f"the punishment gain must not be negative and the decay must be in [0, 1), got {punish_gain} and {decay}")
        if tau <= 0 or expectation_tau <= 0:
            raise ValueError(f"dopamine time constants must be positive, got tau {tau} and expectation tau {expectation_tau}")
        if release_alpha <= 0 or release_theta <= 0:
            raise ValueError(f"the release gamma needs positive alpha and theta, got {release_alpha} and {release_theta}")
        if order not in ORDERS:
            raise ValueError(f"unknown dopamine order {order!r}; choose from {', '.join(ORDERS)}")
        if lr < 0:
            raise ValueError("the learning rate must not be negative")
        self.tau = float(tau)
        self.release_alpha = float(release_alpha)
        self.release_theta = float(release_theta)
        self._release_cache: dict[tuple[float, float], float] = {}  # delays repeat (hops and intervals), so remember them
        self.order = order
        self.lr = float(lr)
        self.expectation_tau = float(expectation_tau)
        self.punish = bool(punish)  # reverse the update of an input neuron that should not fire
        self.punish_gain = float(punish_gain)  # and make it this many times as large
        self.decay = float(decay)  # the fraction every weight moves toward zero each epoch
        self.expectation = float(expectation_start)  # the expected dopamine trace: an exponential moving average of the value
        self.level = 0.0  # the global value, as of `updated`
        self.updated = 0.0  # clock time the level was last brought up to date
        self.total = 0.0  # every unit ever released: the actual counts to date
        self.releases = 0  # how many refires have released
        self.updates = 0  # how many refires have moved their weights

    # --- the value ------------------------------------------------------------

    def level_at(self, time: float) -> float:
        """The value at `time`: the lazy decay since it was last brought up to date."""
        elapsed = time - self.updated
        if elapsed > 0.0:
            self.level *= math.exp(-elapsed / self.tau)
            self.updated = time
        return self.level

    def peek(self, time: float) -> float:
        """The value at `time` without bringing the pool up to date: a read that changes nothing (the trace uses it)."""
        elapsed = time - self.updated
        return self.level * math.exp(-elapsed / self.tau) if elapsed > 0.0 else self.level

    def expected(self, time: float | None = None) -> float:
        """The internal expectation as of the last wave (it moves only when a wave is processed, see step)."""
        return self.expectation

    def _expect(self, elapsed: float) -> None:
        """Move the expectation toward the current value: the exponential window, over the time since the last wave."""
        if elapsed > 0.0:
            self.expectation += (1.0 - math.exp(-elapsed / self.expectation_tau)) * (self.level - self.expectation)

    def advantage(self, time: float) -> float:
        """What an update at `time` would see, read without changing anything."""
        return self.peek(time) - self.expectation

    def release_amount(self, delay: float) -> float:
        """What a refire `delay` ms past the end of its refractory period releases: the gamma density averaged over the next hop."""
        delay = max(0.0, delay)
        hop = Neuron.hop()
        key = (round(delay, 9), hop)
        amount = self._release_cache.get(key)
        if amount is None:
            low = gamma_cdf(self.release_alpha, delay / self.release_theta)
            high = gamma_cdf(self.release_alpha, (delay + hop) / self.release_theta)
            amount = (high - low) / hop
            self._release_cache[key] = amount
        return amount

    def delay_of(self, previous: float, now: float) -> float:
        return now - previous - Neuron.refractory

    def _add(self, amount: float, count: int) -> None:
        self.level += amount
        self.total += amount
        self.releases += count

    # --- a wave --------------------------------------------------------------------

    def step(self, time: float, releases: list[float], update) -> float:
        """One wave at `time`: `releases` are the amounts its refires release, `update(advantage)` moves their weights.

        Returns the advantage the update saw. The releases are summed exactly
        (order-independent), so both engines add the same number.
        """
        elapsed = time - self.updated
        self.level_at(time)
        amount = math.fsum(releases)
        if self.order == "release-first":
            self._add(amount, len(releases))
        advantage = self.level - self.expectation
        if releases:
            update(advantage)
            self.updates += len(releases)
        if self.order != "release-first":
            self._add(amount, len(releases))
        self._expect(elapsed)  # then the expectation catches up a little toward what it just saw
        return advantage

    def status(self) -> str:
        return (
            f"dopamine {self.level:.3g} (expected {self.expected(self.updated):.3g}), "
            f"{self.releases:,} releases, {self.updates:,} updates"
        )

    def state(self) -> dict:
        """What a checkpoint records."""
        return {
            "tau": self.tau, "release_alpha": self.release_alpha, "release_theta": self.release_theta, "order": self.order, "lr": self.lr,
            "expectation_tau": self.expectation_tau, "expectation": self.expectation, "punish": self.punish,
            "punish_gain": self.punish_gain, "decay": self.decay,
            "level": self.level, "updated": self.updated, "total": self.total,
            "releases": self.releases, "updates": self.updates,
        }

    @classmethod
    def from_state(cls, data: dict) -> "Dopamine":
        dopamine = cls(tau=data["tau"], release_alpha=data.get("release_alpha", DOPAMINE_RELEASE_ALPHA),
                       release_theta=data.get("release_theta", DOPAMINE_RELEASE_THETA), order=data["order"], lr=data["lr"],
                       expectation_tau=data.get("expectation_tau", DOPAMINE_EXPECTATION_TAU), punish=data.get("punish", DOPAMINE_PUNISH),
                       punish_gain=data.get("punish_gain", DOPAMINE_PUNISH_GAIN), decay=data.get("decay", WEIGHT_DECAY))
        dopamine.level, dopamine.updated, dopamine.total = data["level"], data["updated"], data["total"]
        dopamine.expectation = data.get("expectation", DOPAMINE_EXPECTATION_START)
        dopamine.releases, dopamine.updates = data["releases"], data["updates"]
        return dopamine


def learn(dopamine: Dopamine, wave, weight_range: tuple[float, float]) -> float:
    """The object engine's learning for one wave: its refires release, then move their gated incoming weights.

    A refire is a firing neuron with a previous spike. The gate: an incoming
    connection counts if its last integrated signal came after that
    previous spike. Returns the advantage the wave saw.
    """
    refires = []
    for neuron in wave.fired:
        previous = neuron.previous_fired_at
        if previous is None:
            continue
        refires.append((neuron, dopamine.release_amount(dopamine.delay_of(previous, wave.time))))
    low, high = weight_range
    lr, punish, gain = dopamine.lr, dopamine.punish, dopamine.punish_gain

    def update(advantage: float) -> None:
        if not advantage:
            return
        for neuron, release in refires:
            step = lr * advantage * release
            if punish and neuron.should_fire is False:
                step = -gain * step  # it should not have fired: the reward is reversed, and outweighs a reward
            since = neuron.previous_fired_at
            for connection in neuron.incoming:
                if connection.is_active and connection.last_signal is not None and connection.last_signal > since:
                    weight = connection.weight + step
                    connection.weight = low if weight < low else high if weight > high else weight

    return dopamine.step(wave.time, [release for _, release in refires], update)
