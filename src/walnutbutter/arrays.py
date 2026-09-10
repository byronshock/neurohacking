"""The array engine: the same network as vectors and a sparse matrix.

`ArrayNetwork` wraps a mesh built the usual way (a `GridOfNeurons` or a
`CartesianNodes`) and runs it with numpy and scipy instead of neuron
objects. Neurons become vectors of length N (potential, threshold, noise,
firing rate, the wave each fired in); connections become vectors of length
E in connection-id order (source, target, weight, active) and a sparse
N x N matrix for delivery. A wave is one sparse matrix-vector product: the
neurons that fired this wave, as a 0/1 vector, times the weights, gives
every neuron the sum of its incoming signals; those not yet fired take it,
the floor is applied, and whoever has reached threshold fires next wave.

The object engine (`propagation.py`) and this one are meant to be the same
network: same topology, same ids, same rules, interchangeable checkpoints,
and `tests/test_arrays.py` runs them side by side. They differ only in the
order floating-point additions happen, so on the rare epoch where a
potential sits within rounding of a threshold the two can decide
differently and their runs diverge from there, like two seeds.

The mesh stays attached as `mesh`: `sync_to_mesh()` copies the arrays back
into its neuron and connection objects, which is how checkpoints are
written and the visualizer draws an array network. Nothing prints per
neuron in this engine.
"""

from __future__ import annotations

import random

import numpy as np
from scipy.sparse import csr_array

from .exploration import TWO_PI, uniforms
from .learning_rules import RATE_MEMORY, STUCK_ABOVE, STUCK_BELOW
from .network import Network
from .neuron import Neuron
from .propagation import Wave


class ArrayWave:
    """What happened in one wave of the array engine: the indices of the neurons that fired."""

    __slots__ = ("number", "fired")

    def __init__(self, number: int, fired: np.ndarray):
        self.number = number
        self.fired = fired

    def __repr__(self) -> str:
        return f"ArrayWave(number={self.number}, fired={self.fired.tolist()})"


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
        self.noise = np.zeros(n)
        self.rate = np.array([x.rate for x in neurons], dtype=float)
        self.fired_wave = np.full(n, -1, dtype=np.int64)  # -1: has not fired this epoch
        for x in neurons:
            if x.has_fired:
                self.fired_wave[self.index[x]] = x.fired_in_wave
        # the clock: when each neuron last spiked (-inf: never) and when its potential was last brought up to date
        self.fired_at = np.array([-np.inf if x.fired_at is None else x.fired_at for x in neurons], dtype=float)
        self.last_update = np.array([x.last_update for x in neurons], dtype=float)

        e = len(mesh.connections)
        connections = [mesh.connections[i] for i in range(1, e + 1)]
        self.source = np.array([self.index[c.source] for c in connections], dtype=np.int64)
        self.target = np.array([self.index[c.target] for c in connections], dtype=np.int64)
        self.weight = np.array([c.weight for c in connections], dtype=float)
        self.active = np.array([c.is_active for c in connections], dtype=bool)
        self._active_edges = np.flatnonzero(self.active)
        self._build_matrix()

        self.input_index = np.array([self.index[x] for x in mesh.input_row()], dtype=np.int64)
        self.output_index = np.array([self.index[x] for x in mesh.output_row()], dtype=np.int64)
        self.waves: list[ArrayWave] = []

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

    def reset(self, discharge: bool = False) -> None:
        if discharge:
            self.potential[:] = 0.0
        self.fired_wave[:] = -1
        self.noise[:] = 0.0
        self.waves = []

    def leak(self, now: float) -> None:
        """Bring every potential up to `now`. The object engine does this lazily per neuron on arrival;
        here it is one vector operation, which is the same arithmetic (decays compose)."""
        elapsed = now - self.last_update
        moving = elapsed > 0.0
        if moving.any():
            if Neuron.tau != np.inf:
                self.potential[moving] *= np.exp(-elapsed[moving] / Neuron.tau)
            self.last_update[moving] = now

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

    def fire_input(self) -> list[ArrayWave]:
        if self.input_pattern is None:
            raise ValueError("no input pattern set; call set_input() first")
        self.time = self.input_time if self.input_time is not None else self.next_time()
        self.epoch += 1
        forced = self.input_index[np.asarray(self.input_pattern, dtype=bool)]
        self.waves = self.propagate_forced(forced, self.time)
        return self.waves

    def output_times(self) -> list[float | None]:
        return [float(self.fired_at[i]) if self.fired_wave[i] >= 0 else None for i in self.output_index]

    def propagate_forced(self, forced: np.ndarray, now: float | None = None) -> list[ArrayWave]:
        """Run one cascade at clock time `now` from the neurons `forced` to fire in wave 0. Returns the waves."""
        self._refresh_matrix()
        matrix = self._matrix
        potential, threshold, floor, fired_wave = self.potential, self.threshold_v, self.floor, self.fired_wave
        n = len(potential)
        if now is None:
            now = self.time
        self.leak(now)
        refractory = self.fired_at + Neuron.refractory > now  # fired within the refractory period before now
        forced = forced[~refractory[forced]]  # a refractory neuron ignores the stimulus too
        firing = np.zeros(n, dtype=float)
        firing[forced] = 1.0
        fired_wave[forced] = 0
        self.fired_at[forced] = now
        potential[forced] = 0.0  # the spike resets the potential
        waves = [ArrayWave(0, np.sort(forced))]
        out_degree = self._out_degree
        while out_degree @ firing > 0:  # signals in flight: the firing neurons have active connections out
            number = len(waves)
            both = matrix @ firing
            incoming, touched = both[:n], both[n:] > 0
            unfired = fired_wave < 0
            take = touched & unfired & ~refractory
            potential[take] = np.maximum(potential[take] + incoming[take], floor[take])
            ready = take & (potential >= threshold)
            fired_wave[ready] = number
            self.fired_at[ready] = now
            potential[ready] = 0.0
            waves.append(ArrayWave(number, np.flatnonzero(ready)))  # recorded even if nobody fired, like the object engine
            if not ready.any():
                break
            firing = ready.astype(float)
        return waves

    def fired_neurons(self):
        self.sync_to_mesh()
        return self.mesh.fired_neurons()

    def output_fired(self) -> list[bool]:
        return (self.fired_wave[self.output_index] >= 0).tolist()

    # --- learning -----------------------------------------------------------

    def reinforce(self, advantage: float, lr: float, sigma: float, eligibility: str, late: str) -> int:
        """The global-reward update, edge by edge, all at once. Returns connections changed."""
        if not advantage:
            return 0
        fw = self.fired_wave
        fired_source, fired_target = fw[self.source], fw[self.target]
        mask = self.active & (fired_source >= 0) & (fired_target != 0)  # delivered, and not into a forced input
        if eligibility == "perturb":
            e = self.noise[self.target] / sigma if sigma else np.zeros(len(self.target))
        else:
            e = np.where(fired_target >= 0, 1.0, -1.0)
        if late != "count":
            arrived_late = (fired_target >= 0) & (fired_source + 1 > fired_target)
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
        unforced = self.fired_wave != 0
        fired = (self.fired_wave[unforced] >= 0).astype(float)
        self.rate[unforced] += RATE_MEMORY * (fired - self.rate[unforced])

    def homeostasis(self, rate: float, target: float, threshold_range: tuple[float, float]) -> int:
        if rate <= 0:
            return 0
        unforced = self.fired_wave != 0
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
        """Copy the arrays into the mesh's neurons and connections (for checkpoints and drawing)."""
        for i, neuron in enumerate(self.neurons_list):
            wave = int(self.fired_wave[i])
            neuron.potential = float(self.potential[i])
            neuron.threshold = float(self.threshold_v[i])
            neuron.noise = float(self.noise[i])
            neuron.rate = float(self.rate[i])
            neuron.has_fired = wave >= 0
            neuron.fired_in_wave = wave if wave >= 0 else None
            neuron.fired_at = None if self.fired_at[i] == -np.inf else float(self.fired_at[i])
            neuron.last_update = float(self.last_update[i])
        connections = self.mesh.connections
        for i, weight in enumerate(self.weight.tolist(), start=1):
            connections[i].weight = weight
        mesh = self.mesh
        mesh.epoch = self.epoch
        mesh.time, mesh.interval, mesh.input_time = self.time, self.interval, self.input_time
        mesh.waves = [Wave(w.number, [], [self.neurons_list[i] for i in w.fired.tolist()]) for w in self.waves]
        for name in ("input_pattern", "input_bits", "input_coded", "input_data"):
            setattr(mesh, name, getattr(self, name))
        mesh._rng.setstate(self._rng.getstate())

    def __repr__(self) -> str:
        return f"ArrayNetwork({self.mesh!r}, {len(self.neurons_list)} neurons, {len(self.weight)} connections)"
