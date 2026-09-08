"""Input patterns for the network.

The network's input is its bottom row of neurons. A pattern is one boolean
per column; the neurons whose bit is 1 are forced to fire in wave 0.

Patterns are complement-coded: the raw bits are followed by their negations,
so 12 raw bits become 24 bits and exactly half of the input row fires no
matter what the raw bits are.
"""

from __future__ import annotations

import random
from typing import Sequence


def random_bits(count: int, seed: int | None = None) -> list[bool]:
    """`count` independent fair coin flips, reproducible with `seed`."""
    rng = random.Random(seed)
    return [rng.random() < 0.5 for _ in range(count)]


def complement_code(bits: Sequence[bool]) -> list[bool]:
    """The bits followed by their complements: [b0, b1, ...] -> [b0, b1, ..., not b0, not b1, ...]."""
    bits = [bool(b) for b in bits]
    return bits + [not b for b in bits]


def parse_bits(text: str) -> list[bool]:
    """Turn a string such as "101100" into bits. Spaces are ignored."""
    cleaned = text.replace(" ", "")
    if not cleaned or any(ch not in "01" for ch in cleaned):
        raise ValueError(f"input must be a string of 0s and 1s, got {text!r}")
    return [ch == "1" for ch in cleaned]


def format_bits(bits: Sequence[bool]) -> str:
    return "".join("1" if b else "0" for b in bits)


# --- error-correcting codes -----------------------------------------------------------

from dataclasses import dataclass


@dataclass(frozen=True)
class Code:
    """A systematic linear block code: k data bits followed by one parity bit per cover.

    Each cover lists the data-bit positions its parity bit sums (mod 2). The
    syndrome of a word is the tuple of failed checks; if every bit position has
    a distinct nonzero syndrome the code corrects single errors.
    """

    name: str
    covers: tuple[tuple[int, ...], ...]
    data_bits: int

    @property
    def code_bits(self) -> int:
        return self.data_bits + len(self.covers)

    def encode(self, data: Sequence[bool]) -> list[bool]:
        data = [bool(b) for b in data]
        if len(data) != self.data_bits:
            raise ValueError(f"{self.name} takes {self.data_bits} data bits, got {len(data)}")
        return data + [sum(data[i] for i in cover) % 2 == 1 for cover in self.covers]

    def syndrome(self, word: Sequence[bool]) -> tuple[int, ...]:
        """Which parity checks fail: all zeros means a valid codeword."""
        word = [bool(b) for b in word]
        if len(word) != self.code_bits:
            raise ValueError(f"{self.name} has {self.code_bits} bits, got {len(word)}")
        return tuple(
            int((sum(word[i] for i in cover) + word[self.data_bits + k]) % 2 == 1)
            for k, cover in enumerate(self.covers)
        )

    def _syndrome_of_position(self, position: int) -> tuple[int, ...]:
        """The syndrome a single flip at `position` produces."""
        probe = [False] * self.code_bits
        probe[position] = True
        return self.syndrome(probe)

    @property
    def corrects_single_errors(self) -> bool:
        patterns = [self._syndrome_of_position(i) for i in range(self.code_bits)]
        return len(set(patterns)) == self.code_bits and all(any(p) for p in patterns)

    def correct(self, word: Sequence[bool]) -> tuple[list[bool], int | None]:
        """Fix a single flipped bit if the code can. Returns (word, position fixed or None)."""
        word = [bool(b) for b in word]
        syndrome = self.syndrome(word)
        if not any(syndrome):
            return word, None
        if not self.corrects_single_errors:
            return word, None  # detected, but the position is ambiguous
        for i in range(self.code_bits):
            if self._syndrome_of_position(i) == syndrome:
                word[i] = not word[i]
                return word, i
        return word, None  # more than one error: the syndrome matches no single flip

    def decode(self, word: Sequence[bool]) -> list[bool]:
        """The data bits of a word, after single-error correction where the code allows it."""
        word, _ = self.correct(word)
        return word[: self.data_bits]


# Hamming (7, 4): three parity bits, each covering three data bits, giving every
# position a distinct syndrome, so any single flipped bit is located and fixed.
HAMMING74 = Code("hamming74", ((0, 1, 3), (0, 2, 3), (1, 2, 3)), 4)
# (6, 4): two parity bits; minimum distance 2, so single flips are detected only.
PARITY64 = Code("parity64", ((0, 1, 2), (1, 2, 3)), 4)

CODES = {c.name: c for c in (HAMMING74, PARITY64)}
DEFAULT_CODE = HAMMING74.name

# backwards-compatible names for the (6, 4) code
ECC_DATA_BITS, ECC_CODE_BITS, ECC_PARITIES = PARITY64.data_bits, PARITY64.code_bits, PARITY64.covers


def ecc_encode(data: Sequence[bool]) -> list[bool]:
    return PARITY64.encode(data)


def ecc_syndrome(word: Sequence[bool]) -> tuple[int, ...]:
    return PARITY64.syndrome(word)


def ecc_decode(word: Sequence[bool]) -> list[bool]:
    return [bool(b) for b in word][:ECC_DATA_BITS]
