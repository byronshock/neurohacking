"""Draw a GridOfNeurons as a picture of pointy-top hexagons using pygame.

The geometry helpers at the top are plain arithmetic and need neither pygame
nor a display, so they are easy to test. `draw_grid` and `save` work on an
off-screen surface; only `show` opens a window.
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402  (import after the env var so pygame stays quiet)

from .grid import GridOfNeurons

SQRT3 = math.sqrt(3)

BACKGROUND = (24, 24, 28)
OUTLINE = (12, 12, 14)
UNFIRED = (70, 74, 84)
FIRED_CENTRE = (255, 224, 96)  # fired neurons shade from this at the origin...
FIRED_EDGE = (214, 84, 28)  # ...to this at the rim
ORIGIN_RING = (255, 255, 255)

# --- geometry (no pygame needed) -------------------------------------------


def axial_to_pixel(q: int, r: int, hex_radius: float) -> tuple[float, float]:
    """Centre of the pointy-top hexagon at axial (q, r), relative to cell (0, 0).

    hex_radius is the distance from a hexagon's centre to any corner.
    """
    x = hex_radius * SQRT3 * (q + r / 2)
    y = hex_radius * 1.5 * r
    return x, y


def hexagon_points(cx: float, cy: float, hex_radius: float) -> list[tuple[float, float]]:
    """The six corners of a pointy-top hexagon centred at (cx, cy)."""
    return [
        (
            cx + hex_radius * math.cos(math.radians(60 * i - 30)),
            cy + hex_radius * math.sin(math.radians(60 * i - 30)),
        )
        for i in range(6)
    ]


def fit_hex_radius(grid_size: int, width: int, height: int, margin: int = 24) -> float:
    """Largest hex radius at which a grid of `grid_size` fits inside width x height."""
    usable_w = width - 2 * margin
    usable_h = height - 2 * margin
    # Widest row (r = 0) holds 2*size + 1 hexagons, each sqrt(3)*R wide.
    by_width = usable_w / ((2 * grid_size + 1) * SQRT3)
    # Rows are 1.5*R apart; add a full R for the top and bottom corners.
    by_height = usable_h / (3 * grid_size + 2)
    return min(by_width, by_height)


def ring_distance(q: int, r: int) -> int:
    """How many rings out from the origin cell (0, 0) is (q, r)."""
    return max(abs(q), abs(r), abs(q + r))


def _lerp_colour(a, b, t: float):
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def neuron_colour(q: int, r: int, has_fired: bool, grid_size: int):
    """Grey if the neuron has not fired; otherwise a shade based on ring distance."""
    if not has_fired:
        return UNFIRED
    t = ring_distance(q, r) / grid_size if grid_size else 0.0
    return _lerp_colour(FIRED_CENTRE, FIRED_EDGE, t)


# --- drawing ------------------------------------------------------------------


def draw_grid(surface: pygame.Surface, grid: GridOfNeurons, margin: int = 24) -> None:
    """Paint the whole grid onto `surface`, scaled to fit."""
    width, height = surface.get_size()
    hex_radius = fit_hex_radius(grid.size, width, height, margin)
    centre_x, centre_y = width / 2, height / 2

    surface.fill(BACKGROUND)
    for (q, r), neuron in grid.neurons.items():
        dx, dy = axial_to_pixel(q, r, hex_radius)
        points = hexagon_points(centre_x + dx, centre_y + dy, hex_radius)
        pygame.draw.polygon(surface, neuron_colour(q, r, neuron.has_fired, grid.size), points)
        pygame.draw.polygon(surface, OUTLINE, points, width=max(1, round(hex_radius / 12)))

    # Mark the origin so the eye can find where the signal started.
    points = hexagon_points(centre_x, centre_y, hex_radius * 0.55)
    pygame.draw.polygon(surface, ORIGIN_RING, points, width=max(1, round(hex_radius / 8)))


def save(grid: GridOfNeurons, path: str, width: int = 800, height: int = 800) -> None:
    """Render the grid to an image file (PNG by extension). Needs no display."""
    surface = pygame.Surface((width, height))
    draw_grid(surface, grid)
    pygame.image.save(surface, path)


def show(grid: GridOfNeurons, width: int = 800, height: int = 800) -> None:
    """Open a window showing the grid. Close it, or press Esc or Q, to return."""
    pygame.init()
    try:
        screen = pygame.display.set_mode((width, height))
        fired = len(grid.fired_neurons())
        pygame.display.set_caption(
            f"neurohacking: size {grid.size}, {fired} of {len(grid.neurons)} neurons fired"
        )
        draw_grid(screen, grid)
        pygame.display.flip()

        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
            clock.tick(30)
    finally:
        pygame.quit()
