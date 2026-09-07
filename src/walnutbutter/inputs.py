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
