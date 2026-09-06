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
FIRED_CENTRE = (255, 224, 96)  # fired neurons shade from this in wave 0...
FIRED_EDGE = (214, 84, 28)  # ...to this in the last wave
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


def layout(grid: GridOfNeurons, width: int, height: int, margin: int = 24) -> tuple[float, float, float]:
    """Choose the largest hex radius at which the whole grid fits, and where to put it.

    Returns (hex_radius, offset_x, offset_y): the pixel centre of cell (q, r) is
    offset plus axial_to_pixel(q, r, hex_radius). Works for any grid shape by
    measuring its bounding box with a radius of 1 and scaling to fit.
    """
    centres = [axial_to_pixel(q, r, 1.0) for q, r in grid.neurons]
    xs = [x for x, _ in centres]
    ys = [y for _, y in centres]
    # Add the half-width (sqrt3/2) and the corner height (1) of the outermost cells.
    extent_w = (max(xs) - min(xs)) + SQRT3
    extent_h = (max(ys) - min(ys)) + 2.0
    hex_radius = min((width - 2 * margin) / extent_w, (height - 2 * margin) / extent_h)
    # Centre the bounding box in the window.
    offset_x = width / 2 - hex_radius * (max(xs) + min(xs)) / 2
    offset_y = height / 2 - hex_radius * (max(ys) + min(ys)) / 2
    return hex_radius, offset_x, offset_y


def _lerp_colour(a, b, t: float):
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def neuron_colour(fired_in_wave: int | None, last_wave: int):
    """Grey if the neuron has not fired; otherwise a shade based on the wave it fired in."""
    if fired_in_wave is None:
        return UNFIRED
    t = fired_in_wave / last_wave if last_wave else 0.0
    return _lerp_colour(FIRED_CENTRE, FIRED_EDGE, t)


# --- drawing ------------------------------------------------------------------


def draw_grid(surface: pygame.Surface, grid: GridOfNeurons, margin: int = 24) -> None:
    """Paint the whole grid onto `surface`, scaled to fit."""
    width, height = surface.get_size()
    hex_radius, offset_x, offset_y = layout(grid, width, height, margin)

    waves = [n.fired_in_wave for n in grid.neurons.values() if n.fired_in_wave is not None]
    last_wave = max(waves) if waves else 0

    surface.fill(BACKGROUND)
    for (q, r), neuron in grid.neurons.items():
        dx, dy = axial_to_pixel(q, r, hex_radius)
        points = hexagon_points(offset_x + dx, offset_y + dy, hex_radius)
        pygame.draw.polygon(surface, neuron_colour(neuron.fired_in_wave, last_wave), points)
        pygame.draw.polygon(surface, OUTLINE, points, width=max(1, round(hex_radius / 12)))

    # Mark the origin so the eye can find where the signal started.
    points = hexagon_points(offset_x, offset_y, hex_radius * 0.55)
    pygame.draw.polygon(surface, ORIGIN_RING, points, width=max(1, round(hex_radius / 8)))


def save(grid: GridOfNeurons, path: str, width: int = 800, height: int = 600) -> None:
    """Render the grid to an image file (PNG by extension). Needs no display."""
    surface = pygame.Surface((width, height))
    draw_grid(surface, grid)
    pygame.image.save(surface, path)


def show(grid: GridOfNeurons, width: int = 800, height: int = 600) -> None:
    """Open a window showing the grid. Close it, or press Esc or Q, to return."""
    pygame.init()
    try:
        screen = pygame.display.set_mode((width, height))
        fired = len(grid.fired_neurons())
        pygame.display.set_caption(
            f"neurohacking: {grid.columns}x{grid.rows}, {fired} of {len(grid.neurons)} neurons fired"
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
