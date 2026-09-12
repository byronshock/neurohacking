"""What every container of neurons shares: an input row, an output row, epochs and the schedule.

A network has `across` x `rows` addressable positions (`get_neuron_at(place,
row)`, row 0 at the top), a bottom row that receives the complement-coded and
permuted input pattern, a top row that is read as the output, and the epoch
machinery: reset, present an input at a time, run the schedule to the next
input's time (AUTHORITY.md §4.2). The hex mesh and the Cartesian lattice both
build on this; the learning code works on either.
"""

from __future__ import annotations

from typing import Iterable

from .constants import INTERVAL
from .dopamine import learn
from .exploration import gaussians
from .inputs import CODES, DEFAULT_CODE, Code, complement_code
from .neuron import Neuron
from .clock import before, slack
from .propagation import Schedule, Wave


class Network:
    """Mixin with the input/epoch machinery. Subclasses provide across (the count per row), rows, get_neuron_at, neurons and _rng."""

    across: int
    rows: int
    weight_range: tuple[float, float]

    def _init_network(self, across: int, weight_range: tuple[float, float]) -> None:
        low, high = weight_range
        if not low < high:
            raise ValueError(f"weight range must run from low to high, got {weight_range}")
        self.weight_range = (float(low), float(high))
        self.waves: list[Wave] = []  # Waves of the most recent propagation
        self.input_pattern: list[bool] | None = None  # one bit per place along the bottom row
        self.input_bits: list[bool] | None = None  # the raw bits before complement coding
        self.input_coded: list[bool] | None = None  # the complement-coded bits before permutation
        self.permutation: list[int] = list(range(across))  # place i along the bottom row shows coded bit permutation[i]
        self.epoch = 0  # how many inputs have been presented
        self.time = 0.0  # the clock, nominal milliseconds: the time of the last input
        self.interval = INTERVAL  # default spacing of inputs when no time is given
        self.input_time: float | None = None  # when the pending input arrives
        self.horizon = 0.0  # the time the schedule has run to: the next input may not come before it
        self.schedule = Schedule()  # signals in flight, across epochs
        self.dopamine = None  # a dopamine.Dopamine when the dopamine rule runs (set by whoever builds the run)
        self.readout = "top"  # what is read as the output: the top row, or "input" (the inputs are the outputs)
        self.read_window: float | None = None  # None: an output is on if it fired this epoch; else if it fired within
        # this many ms before the horizon (the read at the end of the epoch)
        self.ecc: str | None = None  # name of the error-correcting code applied before complement coding, if any
        self.input_data: list[bool] | None = None  # the raw data bits when ecc is on

    def all_neurons(self) -> Iterable[Neuron]:
        """Every neuron, in a stable order. Subclasses override if `neurons` is not a list."""
        return self.neurons

    def get_neuron_at(self, place: int, row: int) -> Neuron | None:  # pragma: no cover - overridden
        raise NotImplementedError

    def clip_weight(self, weight: float) -> float:
        """Keep a weight inside the network's weight_range."""
        low, high = self.weight_range
        return max(low, min(high, weight))

    # --- input ------------------------------------------------------------

    def input_row(self) -> list[Neuron]:
        """The bottom row of neurons, left to right: the network's input."""
        return [self.get_neuron_at(place, self.rows - 1) for place in range(self.across)]

    def output_row(self) -> list[Neuron]:
        """The network's output, left to right: the top row, or the input row when the inputs are the outputs."""
        if self.readout == "input":
            return self.input_row()
        return [self.get_neuron_at(place, 0) for place in range(self.across)]

    def output_fired(self) -> list[bool]:
        """Whether each output neuron is on: fired this epoch, or, with a read window, within it before the horizon."""
        if self.read_window is None:
            return [neuron.has_fired for neuron in self.output_row()]
        since = self.horizon - self.read_window
        return [neuron.fired_at is not None and neuron.fired_at + slack(neuron.fired_at) >= since for neuron in self.output_row()]

    def input_width(self) -> int:
        """How many neurons the input covers: one bit of the (coded, permuted) pattern each."""
        return self.across

    def next_time(self) -> float:
        """When the next input arrives if no time is given: the interval after the last one (the first at 0)."""
        return self.time + self.interval if self.epoch else 0.0

    def set_input(self, pattern, time: float | None = None) -> None:
        """Store the input pattern, one boolean per input neuron, and the time it arrives (default: next_time())."""
        pattern = [bool(b) for b in pattern]
        if len(pattern) != self.input_width():
            raise ValueError(f"input pattern has {len(pattern)} bits but the input covers {self.input_width()} neurons")
        time = self.next_time() if time is None else float(time)
        if self.epoch and before(time, self.horizon):
            raise ValueError(f"input time {time} is before the schedule has already run to, {self.horizon}")
        self.input_pattern = pattern
        self.input_time = time

    @property
    def code(self) -> Code | None:
        return CODES[self.ecc] if self.ecc else None

    def raw_bit_count(self) -> int:
        """How many raw bits an input takes: half the count across, or the code's data bits when a code is on."""
        width = self.input_width()
        if width % 2:
            raise ValueError(f"complement coding needs an even number of input neurons, got {width}")
        if self.code:
            if width // 2 != self.code.code_bits:
                raise ValueError(f"{self.code.name} needs {2 * self.code.code_bits} input neurons, got {width}")
            return self.code.data_bits
        return width // 2

    def use_ecc(self, code: str | bool | None = DEFAULT_CODE) -> None:
        """Encode raw data bits with a named code before complement coding (True means the default, Hamming (7, 4))."""
        if code is True:
            code = DEFAULT_CODE
        if code is False:
            code = None
        if code is not None and code not in CODES:
            raise ValueError(f"unknown code {code!r}; choose from {', '.join(CODES)}")
        self.ecc = code
        self.raw_bit_count()  # validates the count across

    def set_input_bits(self, bits, time: float | None = None) -> None:
        """Set the input from raw bits: (ecc-encode them,) complement-code them, then permute.

        Without ecc the raw bits number half the count across. With ecc they are
        the 4 data bits, encoded to 7 before complement coding fills 14 places.
        Place i along the bottom row receives coded bit permutation[i].
        """
        bits = [bool(b) for b in bits]
        wanted = self.raw_bit_count()
        if len(bits) != wanted:
            raise ValueError(f"expected {wanted} input bits for {self.input_width()} input neurons, got {len(bits)}")
        word = self.code.encode(bits) if self.code else bits
        coded = complement_code(word)
        self.set_input([coded[i] for i in self.permutation], time)
        self.input_data = bits if self.code else None
        self.input_bits = word  # the bits that were complement-coded: the codeword with ecc, the raw bits without
        self.input_coded = coded

    def new_random_input(self, time: float | None = None) -> list[bool]:
        """Draw fresh raw bits from the network's seeded stream and set them as the input.

        Because the stream is the same one used to build the network, a seed
        reproduces the whole sequence of inputs, not just the first.
        """
        bits = [self._rng.random() < 0.5 for _ in range(self.raw_bit_count())]
        self.set_input_bits(bits, time)
        return bits

    def input_neurons(self) -> list[Neuron]:
        """The bottom-row neurons whose input bit is 1 (empty if no pattern is set)."""
        if self.input_pattern is None:
            return []
        return [neuron for neuron, bit in zip(self.input_row(), self.input_pattern) if bit]

    def fire_input(self, until: float | None = None) -> list[Wave]:
        """Present the input: schedule the stimulus at its time and run the schedule to the horizon.

        The horizon is `until`, by default the interval after the input: the
        next input's time. Signals due at or after it wait for the next epoch.
        Returns this epoch's waves.
        """
        if self.input_pattern is None:
            raise ValueError("no input pattern set; call set_input() first")
        self.time = self.input_time if self.input_time is not None else self.next_time()
        self.epoch += 1
        for neuron in self.input_neurons():
            self.schedule.stimulus(neuron, self.time)
        self.horizon = self.time + self.interval if until is None else float(until)
        return self.schedule.run(self.horizon, self.waves, self._on_wave)

    def _on_wave(self, wave: Wave) -> None:
        """After a wave has fired: the refires learn, when the dopamine rule runs."""
        if self.dopamine is not None:
            learn(self.dopamine, wave, self.weight_range)

    def total_spikes(self) -> int:
        return sum(neuron.spikes for neuron in self.all_neurons())

    def output_times(self) -> list[float | None]:
        """When each output neuron fired in the last cascade (the cascade's time), or None if it did not."""
        return [neuron.fired_at if neuron.has_fired else None for neuron in self.output_row()]

    # --- running ----------------------------------------------------------

    def propagate(self, fire=(), inputs=None, now: float | None = None, until: float | None = None) -> list[Wave]:
        """Run a cascade from the given stimulus at `now` (default: the clock) on the network's schedule.

        Anything already in flight runs with it, up to `until` (default: one
        interval after `now`; activity can sustain itself, so a bound is
        needed). The waves are appended to this epoch's and returned.
        """
        now = self.time if now is None else now
        for neuron in fire:
            self.schedule.stimulus(neuron, now)
        for neuron, amount in (inputs or {}).items():
            self.schedule.external(neuron, amount, now)
        self.horizon = now + self.interval if until is None else float(until)
        return self.schedule.run(self.horizon, self.waves, self._on_wave)

    def reset(self, discharge: bool = False) -> None:
        """Start a new epoch: clear every neuron's fired-this-epoch state and this epoch's waves.

        Potentials are kept (there is no leak); with `discharge=True` every
        potential is zeroed instead. Signals in flight stay scheduled.
        """
        for neuron in self.all_neurons():
            neuron.reset(discharge)
        self.waves = []

    def perturb(self, sigma: float, rng, now: float | None = None) -> None:
        """Exploration: add Gaussian noise of standard deviation `sigma` to every potential, floored.

        Each neuron remembers its draw as `noise` (the reinforce rule's
        eligibility). The draws come from `exploration.gaussians`, shared
        with the array engine. `now` is accepted for symmetry and unused:
        nothing about a potential depends on the clock.
        """
        neurons = self.all_neurons() if isinstance(self.all_neurons(), list) else list(self.all_neurons())
        for neuron, draw in zip(neurons, gaussians(rng, len(neurons), sigma)):
            neuron.noise = draw
            neuron.potential = max(neuron.minimum_potential, neuron.potential + draw)

    def fired_neurons(self) -> list[Neuron]:
        """Return the neurons that have fired since the last reset."""
        return [n for n in self.all_neurons() if n.has_fired]
