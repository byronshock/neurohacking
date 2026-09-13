"""The array engine: the same network as vectors and a sparse matrix.

`ArrayNetwork` wraps a mesh built the usual way (a `GridOfNeurons`, a
`HexColumns` or a `CartesianNodes`) and runs it with numpy and scipy instead
of neuron objects. Neurons become vectors of length N (potential, threshold,
noise, firing rate, spike times, the wave each fired in this epoch);
connections become vectors of length E in connection-id order (source,
target, weight, active, the stamp of the last signal integrated) and a
sparse N x N matrix for delivery. A wave is one sparse matrix-vector
product: the neurons whose signals arrive now, as a 0/1 vector, times the
weights, gives every neuron the sum of its incoming signals; those not
refractory take it, the floor is applied, and whoever has reached threshold
fires, its signals scheduled one hop later.

The schedule (AUTHORITY.md §4) is a heap of distinct times, each with the
0/1 vector of sources whose signals arrive then and, when an input lands
then, the vector of neurons forced. The object engine (`propagation.py`)
and this one are meant to be the same network: same topology, same ids,
same rules, interchangeable checkpoints, and `tests/test_arrays.py` runs
them side by side. They differ only in the order floating-point additions
happen, so on the rare wave where a potential sits within rounding of a
threshold the two can decide differently and their runs diverge from
there, like two seeds. Exponentials are taken with `math.exp` on both
sides so that the dopamine arithmetic agrees to the last bit.

The mesh stays attached as `mesh`: `sync_to_mesh()` copies the arrays back
into its neuron and connection objects, which is how checkpoints are
written and the visualizer draws an array network. Nothing prints per
neuron in this engine.
"""

from __future__ import annotations

import heapq
import math
import random

import numpy as np
from scipy.sparse import csr_array

from .constants import RATE_MEMORY, STUCK_ABOVE, STUCK_BELOW
from .exploration import TWO_PI, uniforms
from .network import Network
from .neuron import Neuron
from .clock import TOLERANCE, before, slack
from .propagation import Wave


def slack_v(times: np.ndarray) -> np.ndarray:
    """clock.slack, as a vector."""
    return TOLERANCE * np.maximum(1.0, np.abs(times))


class ArrayWave:
    """What happened in one wave of the array engine: its time and the indices of the neurons that fired."""

    __slots__ = ("number", "time", "fired")

    def __init__(self, number: int, time: float, fired: np.ndarray):
        self.number = number
        self.time = time
        self.fired = fired

    def __repr__(self) -> str:
        return f"ArrayWave(number={self.number}, time={self.time:g}, fired={self.fired.tolist()})"


class ArrayNetwork(Network):
    """A mesh run as arrays. Build the mesh first, then wrap it: `ArrayNetwork(GridOfNeurons(seed=1))`."""

    engine = "arrays"

    def __init__(self, mesh):
        self.mesh = mesh
        self.across, self.rows = mesh.across, mesh.rows
        self._init_network(mesh.across, mesh.weight_range)
        self.permutation = list(mesh.permutation)
        self.ecc = mesh.ecc
        self.epoch = mesh.epoch
        self.time, self.interval, self.input_time = mesh.time, mesh.interval, mesh.input_time
        self.horizon = mesh.horizon
        self.dopamine = mesh.dopamine  # shared: one pool, whichever engine runs
        self.readout, self.read, self.read_window, self.coding = mesh.readout, mesh.read, mesh.read_window, mesh.coding
        self.rule = mesh.rule
        self.seed = mesh.seed
        self.threshold = mesh.threshold
        self.minimum_potential = mesh.minimum_potential
        self._rng = random.Random()
        self._rng.setstate(mesh._rng.getstate())  # the same input sequence as the mesh would draw
        for name in ("input_pattern", "input_bits", "input_coded", "input_data"):
            setattr(self, name, getattr(mesh, name))

        neurons = list(mesh.all_neurons())
        self.neurons_list = neurons
        self.index = {neuron: i for i, neuron in enumerate(neurons)}
        n = len(neurons)
        self.potential = np.array([x.potential for x in neurons], dtype=float)
        self.threshold_v = np.array([x.threshold for x in neurons], dtype=float)
        self.floor = np.array([x.minimum_potential for x in neurons], dtype=float)
        self.noise = np.array([x.noise for x in neurons], dtype=float)
        self.rate = np.array([x.rate for x in neurons], dtype=float)
        self.fired_wave = np.full(n, -1, dtype=np.int64)  # -1: has not fired this epoch
        for x in neurons:
            if x.has_fired:
                self.fired_wave[self.index[x]] = x.fired_in_wave
        self.forced = np.array([x.forced for x in neurons], dtype=bool)
        self.sign = np.array([-1.0 if x.should_fire is False else 1.0 for x in neurons])  # -1: an input neuron that should not fire this epoch
        # the clock: when each neuron last spiked (-inf: never), the spike before that, and how many spikes ever
        self.fired_at = np.array([-np.inf if x.fired_at is None else x.fired_at for x in neurons], dtype=float)
        self.previous_fired_at = np.array([-np.inf if x.previous_fired_at is None else x.previous_fired_at for x in neurons], dtype=float)
        self.spikes = np.array([x.spikes for x in neurons], dtype=np.int64)
        self.last_update = np.array([x.last_update for x in neurons], dtype=float)  # the lazy leak's bookkeeping

        e = len(mesh.connections)
        connections = [mesh.connections[i] for i in range(1, e + 1)]
        self.source = np.array([self.index[c.source] for c in connections], dtype=np.int64)
        self.target = np.array([self.index[c.target] for c in connections], dtype=np.int64)
        self.weight = np.array([c.weight for c in connections], dtype=float)
        self.active = np.array([c.is_active for c in connections], dtype=bool)
        self.last_signal = np.array([-np.inf if c.last_signal is None else c.last_signal for c in connections], dtype=float)
        self.delivered_wave = np.full(e, -1, dtype=np.int64)  # the wave of this epoch each connection delivered in
        self.eligibility = np.array([c.eligibility for c in connections], dtype=float)  # what each synapse has earned this epoch
        self._active_edges = np.flatnonzero(self.active)
        self._build_matrix()

        self.input_index = np.array([self.index[x] for x in mesh.input_row()], dtype=np.int64)
        self.output_index = np.array([self.index[x] for x in mesh.output_row()], dtype=np.int64)
        self.waves: list[ArrayWave] = []
        # the schedule: distinct times, each with the sources whose signals arrive then and the neurons forced then
        self._times: list[float] = []
        self._arrivals: dict[float, np.ndarray] = {}
        self._stimuli: dict[float, np.ndarray] = {}
        for time, connection in mesh.schedule.pending():
            self._arrive(self.index[connection.source], time)

    # --- the matrix -------------------------------------------------------

    def _build_matrix(self) -> None:
        """Delivery matrix W[target, source] = weight over active edges, and its 0/1 shadow for "touched"."""
        n = len(self.neurons_list)
        edges = self._active_edges
        order = np.lexsort((self.source[edges], self.target[edges]))  # CSR order: by target, then source
        self._order = edges[order]  # edge id (0-based) at each data position
        rows, cols = self.target[self._order], self.source[self._order]
        counts = np.bincount(rows, minlength=n)
        indptr = np.concatenate(([0], np.cumsum(counts)))
        # Rows 0..n-1 carry the weights (W[target, source]); rows n..2n-1 carry a 1 per edge, so one
        # product gives both the summed input and whether a neuron was touched at all.
        data = np.concatenate((self.weight[self._order], np.ones(len(cols))))
        indices = np.concatenate((cols, cols)).astype(np.int32)
        indptr2 = np.concatenate((indptr, indptr[1:] + indptr[-1])).astype(np.int32)
        self._matrix = csr_array((data, indices, indptr2), shape=(2 * n, n))
        self._edge_count = len(cols)
        self._out_degree = np.bincount(self.source[edges], minlength=n).astype(float)  # active edges out of each neuron
        self._matrix_dirty = False

    def _refresh_matrix(self) -> None:
        if self._matrix_dirty:
            self._matrix.data[: self._edge_count] = self.weight[self._order]
            self._matrix_dirty = False

    # --- the schedule ----------------------------------------------------------

    def _key(self, time: float) -> float:
        time = float(time)
        if time not in self._arrivals and time not in self._stimuli:
            heapq.heappush(self._times, time)
        return time

    def _arrive(self, source: int, time: float) -> None:
        time = self._key(time)
        if time not in self._arrivals:
            self._arrivals[time] = np.zeros(len(self.neurons_list))
        self._arrivals[time][source] = 1.0

    def _arrive_all(self, firing: np.ndarray, time: float) -> None:
        time = self._key(time)
        if time not in self._arrivals:
            self._arrivals[time] = np.zeros(len(self.neurons_list))
        self._arrivals[time] += firing

    def _stimulate(self, forced: np.ndarray, time: float) -> None:
        time = self._key(time)
        if time not in self._stimuli:
            self._stimuli[time] = np.zeros(len(self.neurons_list), dtype=bool)
        self._stimuli[time] |= forced

    def pending(self) -> list[tuple[float, int]]:
        """Signals in flight as (time, source index), by time: what sync_to_mesh puts back on the mesh's schedule."""
        return [(time, int(i)) for time in sorted(self._arrivals) for i in np.flatnonzero(self._arrivals[time] > 0)]

    def _run(self, until: float = math.inf) -> list[ArrayWave]:
        """Process every wave due before `until` (see propagation.Schedule.run), appending to this epoch's waves."""
        potential, threshold, floor, fired_wave = self.potential, self.threshold_v, self.floor, self.fired_wave
        n = len(potential)
        hop = Neuron.hop()
        out_degree = self._out_degree
        while self._times and before(self._times[0], until):
            self._refresh_matrix()
            first = heapq.heappop(self._times)
            limit = first + slack(first)
            firing = self._arrivals.pop(first, None)
            stimulus = self._stimuli.pop(first, None)
            time = first if stimulus is None else first
            while self._times and self._times[0] <= limit:  # the same moment, within the clock's slack
                other = heapq.heappop(self._times)
                more = self._arrivals.pop(other, None)
                if more is not None:
                    firing = more if firing is None else firing + more
                forced_too = self._stimuli.pop(other, None)
                if forced_too is not None:
                    stimulus = forced_too if stimulus is None else stimulus | forced_too
                    time = other  # an input's exact time anchors the wave
            number = len(self.waves)
            refractory = self.fired_at + Neuron.refractory > time + slack(time)  # fired within the refractory period before now
            take = np.zeros(n, dtype=bool)
            if firing is not None:
                both = self._matrix @ firing
                incoming, touched = both[:n], both[n:] > 0
                take = touched & ~refractory
                self.leak(time, take)  # the lazy leak: a neuron is brought up to date when a signal reaches it
                potential[take] = np.maximum(potential[take] + incoming[take], floor[take])
                delivering = self.active & (firing[self.source] > 0)
                self.delivered_wave[delivering] = number
                self.last_signal[delivering & take[self.target]] = time
            fire_forced = (stimulus & ~refractory) if stimulus is not None else np.zeros(n, dtype=bool)
            if Neuron.bored_after > 0.0:  # threshold homeostasis: the threshold falls with the silence since the last spike
                since = np.where(self.fired_at == -np.inf, 0.0, self.fired_at)
                facing = threshold - threshold * (time - since) / Neuron.bored_after
            else:
                facing = threshold
            fired = fire_forced | (~refractory & (self.potential_at(time) >= facing))  # everyone, touched or not, like the object engine
            idx = np.flatnonzero(fired)
            if len(idx):
                self.previous_fired_at[idx] = self.fired_at[idx]
                self.fired_at[idx] = time
                self.last_update[idx] = time
                potential[idx] = 0.0  # the spike resets the potential
                fired_wave[idx] = number
                self.spikes[idx] += 1
                self.forced |= fire_forced
                if out_degree @ fired > 0:  # someone has active connections out: signals in flight
                    self._arrive_all(fired.astype(float), time + hop)
            self.waves.append(ArrayWave(number, time, idx))
            if self.dopamine is not None:
                self._learn(time, idx)  # every wave, fired or not: the pool decays in the same steps as the object engine
        return self.waves

    # --- the epoch ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.neurons_list)

    @property
    def neurons(self):
        """The mesh's neuron container, for code that only counts or lists them (see sync_to_mesh)."""
        return self.mesh.neurons

    @property
    def connections(self):
        return self.mesh.connections

    def all_neurons(self):
        return self.neurons_list

    def get_neuron_at(self, place: int, row: int, *args):
        return self.mesh.get_neuron_at(place, row, *args)

    def input_row(self):
        return self.mesh.input_row()

    def output_row(self):
        return self.mesh.output_row()

    def input_width(self) -> int:
        return self.mesh.input_width()

    def has_fired(self) -> np.ndarray:
        return self.fired_wave >= 0

    def total_spikes(self) -> int:
        return int(self.spikes.sum())

    def reset(self, discharge: bool = False) -> None:
        if discharge:
            self.potential[:] = 0.0
        self.fired_wave[:] = -1
        self.forced[:] = False
        if self.rule == "teacher":
            self.eligibility[:] = 0.0  # a new epoch earns its own credit
        self.noise[:] = 0.0
        self.delivered_wave[:] = -1
        self.waves = []

    def leak(self, now: float, which=None) -> None:
        """Bring potentials up to `now` (all, or the boolean mask `which`): one vector operation, the same arithmetic as the objects."""
        mask = np.ones(len(self.potential), dtype=bool) if which is None else which
        elapsed = now - self.last_update
        moving = mask & (elapsed > 0.0)
        if moving.any():
            if Neuron.tau != math.inf:
                self.potential[moving] *= np.exp(-elapsed[moving] / Neuron.tau)
            self.last_update[moving] = now

    def potential_at(self, now: float) -> np.ndarray:
        """Every potential as it stands at `now`, decayed for the time since its last update; changes nothing."""
        elapsed = np.maximum(now - self.last_update, 0.0)
        if Neuron.tau == math.inf:
            return self.potential
        return self.potential * np.exp(-elapsed / Neuron.tau)

    def perturb(self, sigma: float, rng, now: float | None = None) -> None:
        """Exploration: the same Box-Muller draws as the object engine (see exploration.py), done as a vector."""
        now = self.input_time if now is None else now
        if now is not None:
            self.leak(now)
        n = len(self.neurons_list)
        draws = np.array(uniforms(rng, n))
        angle = TWO_PI * draws[0::2]
        radius = sigma * np.sqrt(-2.0 * np.log(1.0 - draws[1::2]))
        noise = np.empty(len(draws))
        noise[0::2] = np.cos(angle) * radius
        noise[1::2] = np.sin(angle) * radius
        self.noise = noise[:n]
        np.maximum(self.potential + self.noise, self.floor, out=self.potential)

    def set_input(self, pattern, time: float | None = None) -> None:
        super().set_input(pattern, time)
        self.sign[:] = 1.0
        self.sign[self.input_index[~np.asarray(self.input_pattern, dtype=bool)]] = -1.0

    def fire_input(self, until: float | None = None) -> list[ArrayWave]:
        if self.input_pattern is None:
            raise ValueError("no input pattern set; call set_input() first")
        self.time = self.input_time if self.input_time is not None else self.next_time()
        self.epoch += 1
        forced = np.zeros(len(self.neurons_list), dtype=bool)
        forced[self.input_index[np.asarray(self.input_pattern, dtype=bool)]] = True
        self._stimulate(forced, self.time)
        self.horizon = self.time + self.interval if until is None else float(until)
        waves = self._run(self.horizon)
        self.forget()
        return waves

    def forget(self) -> None:
        if self.dopamine is not None and self.dopamine.decay > 0.0:
            self.weight *= 1.0 - self.dopamine.decay
            self._matrix_dirty = True

    def propagate(self, fire=(), inputs=None, now: float | None = None, until: float | None = None) -> list[ArrayWave]:
        """Run a cascade from the mesh neurons `fire`, forced at `now` (default: the clock), up to `until` (default: one interval)."""
        if inputs:
            raise NotImplementedError("external input amounts are not vectorised; use the object engine")
        now = self.time if now is None else now
        forced = np.zeros(len(self.neurons_list), dtype=bool)
        forced[[self.index[neuron] for neuron in fire]] = True
        self._stimulate(forced, now)
        self.horizon = now + self.interval if until is None else float(until)
        return self._run(self.horizon)

    def output_times(self) -> list[float | None]:
        return [float(self.fired_at[i]) if self.fired_wave[i] >= 0 else None for i in self.output_index]

    def fired_neurons(self):
        self.sync_to_mesh()
        return self.mesh.fired_neurons()

    def output_fired(self) -> list[bool]:
        fired_at = self.fired_at[self.output_index]
        if self.read == "again":
            return (fired_at > self.time + slack(self.time)).tolist()
        if self.read == "window" and self.read_window is not None:
            return (fired_at + slack_v(fired_at) >= self.horizon - self.read_window).tolist()
        return (self.fired_wave[self.output_index] >= 0).tolist()

    # --- learning: dopamine ---------------------------------------------------------

    def _learn(self, time: float, idx: np.ndarray) -> float:
        """The refires among the neurons `idx` firing at `time` release, then move their gated incoming weights (dopamine.learn)."""
        dopamine = self.dopamine
        previous = self.previous_fired_at[idx]
        refired = previous > -np.inf
        ridx = idx[refired]
        delays = time - previous[refired] - Neuron.refractory
        releases = [dopamine.release_amount(d) for d in delays.tolist()]  # math.exp, like the object engine
        n = len(self.potential)
        refired_v = np.zeros(n, dtype=bool)
        refired_v[ridx] = True

        def gated() -> np.ndarray:
            return self.active & refired_v[self.target] & (self.last_signal > self.previous_fired_at[self.target])

        def earn(_advantage: float) -> None:
            """The teacher rule: remember what each gated synapse earned; the signal comes at the read."""
            earned = np.zeros(n)
            earned[ridx] = np.array(releases)
            mask = gated()
            self.eligibility[mask] += earned[self.target[mask]]

        def update(advantage: float) -> None:
            if not advantage:
                return
            step = np.zeros(n)
            step[ridx] = dopamine.lr * advantage * np.array(releases)
            if dopamine.punish:
                step *= np.where(self.sign < 0, -dopamine.punish_gain, 1.0)  # a should-not-fire refire: reversed, and outweighing a reward
            mask = gated()
            low, high = self.weight_range
            self.weight[mask] = np.clip(self.weight[mask] + step[self.target[mask]], low, high)
            self._matrix_dirty = True

        return dopamine.step(time, releases, earn if self.rule == "teacher" else update)

    def apply_teacher(self, signal: float, lr: float) -> int:
        """An external teacher's signal at the read (see dopamine.apply_teacher), as one vector operation."""
        earned = self.eligibility != 0.0
        moved = 0
        if signal:
            low, high = self.weight_range
            self.weight[earned] = np.clip(self.weight[earned] + lr * signal * self.eligibility[earned], low, high)
            self._matrix_dirty = True
            moved = int(earned.sum())
        self.eligibility[:] = 0.0
        return moved

    # --- learning: the reinforce rule, factored out ---------------------------------

    def reinforce(self, advantage: float, lr: float, sigma: float, eligibility: str, late: str) -> int:
        """The global-reward update, edge by edge, all at once. Returns connections changed."""
        if not advantage:
            return 0
        fired_target = self.fired_wave[self.target]
        mask = (self.delivered_wave >= 0) & ~self.forced[self.target]  # delivered this epoch, and not into a forced neuron
        if eligibility == "perturb":
            e = self.noise[self.target] / sigma if sigma else np.zeros(len(self.target))
        else:
            e = np.where(fired_target >= 0, 1.0, -1.0)
        if late != "count":
            arrived_late = (fired_target >= 0) & (self.delivered_wave > fired_target)
            if late == "ignore":
                mask &= ~arrived_late
            else:
                e = np.where(arrived_late, -e, e)
        mask &= e != 0
        low, high = self.weight_range
        self.weight[mask] = np.clip(self.weight[mask] + lr * advantage * e[mask], low, high)
        self._matrix_dirty = True
        return int(mask.sum())

    def update_rates(self) -> None:
        unforced = ~self.forced
        fired = (self.fired_wave[unforced] >= 0).astype(float)
        self.rate[unforced] += RATE_MEMORY * (fired - self.rate[unforced])

    def homeostasis(self, rate: float, target: float, threshold_range: tuple[float, float]) -> int:
        if rate <= 0:
            return 0
        unforced = ~self.forced
        low, high = threshold_range
        self.threshold_v[unforced] = np.clip(self.threshold_v[unforced] + rate * (self.rate[unforced] - target), low, high)
        return int(unforced.sum())

    def unstick_outputs(self, rate: float, target: float, threshold_range: tuple[float, float]) -> list[int]:
        if rate <= 0:
            return []
        idx = self.output_index
        stuck = idx[(self.rate[idx] > STUCK_ABOVE) | (self.rate[idx] < STUCK_BELOW)]
        low, high = threshold_range
        self.threshold_v[stuck] = np.clip(self.threshold_v[stuck] + rate * (self.rate[stuck] - target), low, high)
        return stuck.tolist()

    def stuck(self) -> tuple[list[int], list[int]]:
        return np.flatnonzero(self.rate > STUCK_ABOVE).tolist(), np.flatnonzero(self.rate < STUCK_BELOW).tolist()

    # --- back to objects ------------------------------------------------------

    def sync_to_mesh(self) -> None:
        """Copy the arrays into the mesh's neurons, connections and schedule (for checkpoints and drawing)."""
        for i, neuron in enumerate(self.neurons_list):
            wave = int(self.fired_wave[i])
            neuron.potential = float(self.potential[i])
            neuron.threshold = float(self.threshold_v[i])
            neuron.noise = float(self.noise[i])
            neuron.rate = float(self.rate[i])
            neuron.has_fired = wave >= 0
            neuron.fired_in_wave = wave if wave >= 0 else None
            neuron.forced = bool(self.forced[i])
            neuron.should_fire = None if neuron.should_fire is None else bool(self.sign[i] > 0)
            neuron.fired_at = None if self.fired_at[i] == -np.inf else float(self.fired_at[i])
            neuron.previous_fired_at = None if self.previous_fired_at[i] == -np.inf else float(self.previous_fired_at[i])
            neuron.last_update = float(self.last_update[i])
            neuron.spikes = int(self.spikes[i])
        connections = self.mesh.connections
        for i, (weight, last) in enumerate(zip(self.weight.tolist(), self.last_signal.tolist()), start=1):
            connections[i].weight = weight
            connections[i].last_signal = None if last == -np.inf else last
        mesh = self.mesh
        mesh.epoch = self.epoch
        mesh.time, mesh.interval, mesh.input_time, mesh.horizon = self.time, self.interval, self.input_time, self.horizon
        mesh.dopamine, mesh.rule = self.dopamine, self.rule
        for i, weight in enumerate(self.eligibility.tolist(), start=1):
            connections[i].eligibility = weight
        mesh.waves = [Wave(w.number, w.time, [], [self.neurons_list[i] for i in w.fired.tolist()]) for w in self.waves]
        mesh.schedule.clear()
        for time, i in self.pending():
            for connection in self.neurons_list[i].outgoing:
                if connection.is_active:
                    mesh.schedule.signal(connection, time)
        for name in ("input_pattern", "input_bits", "input_coded", "input_data"):
            setattr(mesh, name, getattr(self, name))
        mesh._rng.setstate(self._rng.getstate())

    def __repr__(self) -> str:
        return f"ArrayNetwork({self.mesh!r}, {len(self.neurons_list)} neurons, {len(self.weight)} connections)"
