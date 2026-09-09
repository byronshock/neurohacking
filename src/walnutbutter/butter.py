"""Walnut butter: the substance the neurons are made of.

Butter is spread over the plane in smears. Each smear is a shape with a
density, and placing the butter packs neurons on a hexagonal lattice inside
each shape at the spacing that density implies. Thick butter means many
neurons close together; thin butter means a few far apart; bare plane means
none. Butter that is spread near other butter connects: a neuron projects to
every neuron within its reach, so density alone decides how richly wired a
region is, and a gap in the spread is a gap in the network.

Butterspace has one scale, the unit distance. Density is measured in
neurons per unit cell, where a cell is the hexagon a neuron owns in a
lattice at unit spacing (area sqrt(3)/2 square units), so unit density
means unit spacing: neighbours are exactly one unit distance apart, their
neighbours two. A density of 4 packs four neurons into every cell, half a
unit apart. Every distance in the substance, a reach included, is compared
with this one unit, whatever the density: butter four times as thick has
four times the neurons within the same reach.

The default network is one rectangular smear at unit density: an 8 x 10
lattice at unit spacing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

ROW_SPACING = math.sqrt(3) / 2
CELL_AREA = math.sqrt(3) / 2  # square units per neuron in a hex lattice at unit spacing: the unit cell
UNIT_DENSITY = 1.0  # one neuron per unit cell: the lattice at unit spacing


def spacing_for(density: float) -> float:
    """Hex-lattice spacing, in unit distances, that gives `density` neurons per unit cell: 1 / sqrt(density)."""
    if density <= 0:
        raise ValueError(f"density must be positive, got {density}")
    return 1.0 / math.sqrt(density)


def per_unit_area(density: float) -> float:
    """Convert a density in neurons per unit cell to neurons per square unit."""
    return density / CELL_AREA


@dataclass(frozen=True)
class Rect:
    """An axis-aligned rectangle, corners inclusive, in unit distances."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self):
        if not (self.x_min < self.x_max and self.y_min < self.y_max):
            raise ValueError(f"a rectangle needs x_min < x_max and y_min < y_max, got {self}")

    def contains(self, x: float, y: float) -> bool:
        return self.x_min - 1e-9 <= x <= self.x_max + 1e-9 and self.y_min - 1e-9 <= y <= self.y_max + 1e-9

    def bounds(self) -> tuple[float, float, float, float]:
        return self.x_min, self.y_min, self.x_max, self.y_max


@dataclass(frozen=True)
class Disc:
    """A disc of `radius` around (cx, cy), edge inclusive, in unit distances."""

    cx: float
    cy: float
    radius: float

    def __post_init__(self):
        if self.radius <= 0:
            raise ValueError(f"a disc needs a positive radius, got {self.radius}")

    def contains(self, x: float, y: float) -> bool:
        return math.dist((x, y), (self.cx, self.cy)) <= self.radius + 1e-9

    def bounds(self) -> tuple[float, float, float, float]:
        return self.cx - self.radius, self.cy - self.radius, self.cx + self.radius, self.cy + self.radius


@dataclass(frozen=True)
class Smear:
    shape: Rect | Disc
    density: float  # neurons per unit cell (1 = unit spacing)

    def positions(self) -> list[tuple[float, float]]:
        """Hex-lattice points inside the shape at this smear's spacing, packed from the bottom-left corner.

        Odd rows are shifted half a spacing right, pointy-top like the lattice.
        A rectangle sized for a columns x rows lattice at unit density gives
        exactly that lattice.
        """
        s = spacing_for(self.density)
        x0, y0, x1, y1 = self.shape.bounds()
        points = []
        r = 0
        while True:
            y = y0 + r * s * ROW_SPACING
            if y > y1 + 1e-9:
                break
            shift = 0.5 * s if r % 2 else 0.0
            c = 0
            while True:
                x = x0 + c * s + shift
                if x > x1 + 1e-9:
                    break
                if self.shape.contains(x, y):
                    points.append((x, y))
                c += 1
            r += 1
        return points


class WalnutButter:
    """A recipe of smears. `spread` adds one; `positions` is where the neurons go."""

    def __init__(self):
        self.smears: list[Smear] = []

    def spread(self, shape: Rect | Disc, density: float = UNIT_DENSITY) -> "WalnutButter":
        """Spread butter of the given density over a shape. Returns self, so calls chain."""
        self.smears.append(Smear(shape, density))
        return self

    def density_at(self, x: float, y: float) -> float:
        """Combined density at a point: smears that overlap add up."""
        return sum(m.density for m in self.smears if m.shape.contains(x, y))

    def positions(self) -> list[tuple[float, float]]:
        """Every neuron position the recipe produces, smear by smear."""
        points: list[tuple[float, float]] = []
        for smear in self.smears:
            points.extend(smear.positions())
        return points

    @staticmethod
    def rectangle(columns: int = 8, rows: int = 10) -> "WalnutButter":
        """The default recipe: one rectangle at unit density holding a columns x rows lattice."""
        width = (columns - 1) + 0.5  # odd rows are shifted half a unit
        height = (rows - 1) * ROW_SPACING
        return WalnutButter().spread(Rect(-width / 2, -height / 2, width / 2, height / 2), UNIT_DENSITY)

    def __repr__(self) -> str:
        return f"WalnutButter({len(self.smears)} smear{'s' if len(self.smears) != 1 else ''}, {len(self.positions())} neurons)"
