"""What every container of neurons shares: an input row, an output row, epochs and propagation.

A network has `across` x `rows` addressable positions (`get_neuron_at(place,
row)`, row 0 at the top), a bottom row that receives the complement-coded and
permuted input pattern, a top row that is read as the output, and the epoch
machinery: reset, present an input, propagate wave by wave. The hex mesh and
the Cartesian lattice both build on this; the learning code works on either.
"""

from __future__ import annotations

from typing import Iterable

from .constants import INTERVAL
from .exploration import gaussians
from .inputs import CODES, DEFAULT_CODE, Code, complement_code
from .neuron import Neuron
from .propagation import Wave, propagate


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
        """The top row of neurons, left to right: the network's output."""
        return [self.get_neuron_at(place, 0) for place in range(self.across)]

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
        if self.epoch and time < self.time:
            raise ValueError(f"input time {time} is before the clock, which stands at {self.time}")
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

    def fire_input(self) -> list[Wave]:
        """Advance the clock to the input's time, force the input neurons to fire, and propagate wave by wave."""
        if self.input_pattern is None:
            raise ValueError("no input pattern set; call set_input() first")
        self.time = self.input_time if self.input_time is not None else self.next_time()
        self.epoch += 1
        return self.propagate(fire=self.input_neurons(), now=self.time)

    def output_times(self) -> list[float | None]:
        """When each output neuron fired in the last cascade (the cascade's time), or None if it did not."""
        return [neuron.fired_at if neuron.has_fired else None for neuron in self.output_row()]

    # --- running ----------------------------------------------------------

    def propagate(self, fire=(), inputs=None, now: float | None = None) -> list[Wave]:
        """Run one cascade from the given stimulus (see propagation.propagate) and keep its waves."""
        self.waves = propagate(fire=fire, inputs=inputs, now=now)
        return self.waves

    def reset(self, discharge: bool = False) -> None:
        """Start a new cascade: clear every neuron's fired state and the potential of those that fired.

        Unfired neurons keep their potential, which leaks lazily as the clock
        moves on (see Neuron). With `discharge=True` every potential is zeroed,
        the old epoch-by-epoch behaviour.
        """
        for neuron in self.all_neurons():
            neuron.reset(discharge)
        self.waves = []

    def perturb(self, sigma: float, rng, now: float | None = None) -> None:
        """Exploration: add Gaussian noise of standard deviation `sigma` to every potential, floored.

        The potential is first leaked to `now` (default: the pending input's
        time), so the noise sits on top of what survived the gap and decays
        like everything else from there. Each neuron remembers its draw as
        `noise` (the learning rule's eligibility). The draws come from
        `exploration.gaussians`, shared with the array engine.
        """
        now = self.input_time if now is None else now
        neurons = self.all_neurons() if isinstance(self.all_neurons(), list) else list(self.all_neurons())
        for neuron, draw in zip(neurons, gaussians(rng, len(neurons), sigma)):
            if now is not None:
                neuron.leak(now)
            neuron.noise = draw
            neuron.potential = max(neuron.minimum_potential, neuron.potential + draw)

    def fired_neurons(self) -> list[Neuron]:
        """Return the neurons that have fired since the last reset."""
        return [n for n in self.all_neurons() if n.has_fired]
