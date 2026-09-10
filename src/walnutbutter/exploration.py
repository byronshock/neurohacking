"""Exploration noise, drawn the same way by both engines.

Each epoch every neuron's potential is nudged by a Gaussian draw. Both
engines take their draws from the same `random.Random` stream of uniforms
and apply the Box-Muller transform to pairs, so a seed gives the same
noise whichever engine runs the network: the object engine does the
arithmetic per pair in Python, the array engine on the whole vector at
once. (`random.gauss` was not used because it caches a spare draw across
calls, which no vector version could reproduce.)
"""

from __future__ import annotations

import math

TWO_PI = 2.0 * math.pi


def uniforms(rng, count: int) -> list[float]:
    """`count` uniforms from the stream, rounded up to an even number so pairs are whole."""
    return [rng.random() for _ in range(count + (count & 1))]


def gaussians(rng, count: int, sigma: float) -> list[float]:
    """`count` independent N(0, sigma) draws from the uniform stream, in pairs (Box-Muller)."""
    draws = uniforms(rng, count)
    out = []
    for i in range(0, len(draws), 2):
        angle = TWO_PI * draws[i]
        radius = sigma * math.sqrt(-2.0 * math.log(1.0 - draws[i + 1]))
        out.append(math.cos(angle) * radius)
        out.append(math.sin(angle) * radius)
    return out[:count]
