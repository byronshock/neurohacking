"""Draw a GridOfNeurons as a picture of pointy-top hexagons using pygame.

The geometry helpers at the top are plain arithmetic and need neither pygame
nor a display, so they are easy to test. `draw_grid` and `save` work on an
off-screen surface; only `show` opens a window.
"""

from __future__ import annotations

import math
import os
import sys
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402  (import after the env var so pygame stays quiet)

from .cartesian import CartesianNodes
from .columns import HexColumns
from .grid import GridOfNeurons
from .learning import Teacher
from .monitor import run_epoch

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


DISC_FILL = 0.95  # a neuron's disc radius as a fraction of its cell's inscribed radius: a small gap between discs


def draw_grid(surface: pygame.Surface, grid: GridOfNeurons, margin: int = 24) -> None:
    """Paint the whole grid onto `surface`, scaled to fit.

    Each neuron is a disc centred on its hexagonal cell. The disc radius is just
    under the cell's inscribed radius (sqrt(3)/2 of the hex radius), so
    neighbouring discs never touch or overlap.
    """
    width, height = surface.get_size()
    hex_radius, offset_x, offset_y = layout(grid, width, height, margin)

    waves = [n.fired_in_wave for n in grid.neurons.values() if n.fired_in_wave is not None]
    last_wave = max(waves) if waves else 0

    disc = hex_radius * SQRT3 / 2 * DISC_FILL
    outline = max(1, round(hex_radius / 12))
    surface.fill(BACKGROUND)
    for (q, r), neuron in grid.neurons.items():
        dx, dy = axial_to_pixel(q, r, hex_radius)
        centre = (offset_x + dx, offset_y + dy)
        pygame.draw.circle(surface, neuron_colour(neuron.fired_in_wave, last_wave), centre, disc)
        pygame.draw.circle(surface, OUTLINE, centre, disc, width=outline)

    # Ring the stimulus: the neurons that fired in wave 0, or the input neurons
    # that will be forced when the mesh is fired, or failing both the origin.
    stimulus = [n for n in grid.neurons.values() if n.fired_in_wave == 0] or grid.input_neurons()
    if not stimulus and grid.get_origin_neuron() is not None:
        stimulus = [grid.get_origin_neuron()]
    for neuron in stimulus:
        dx, dy = axial_to_pixel(*neuron.position, hex_radius)
        pygame.draw.circle(surface, ORIGIN_RING, (offset_x + dx, offset_y + dy), hex_radius * 0.55, width=max(1, round(hex_radius / 8)))


# --- Cartesian populations ------------------------------------------------------


def node_layout(nodes: CartesianNodes, width: int, height: int, margin: int = 24):
    """Map unit distances onto pixels so the region and every neuron fit, preserving aspect ratio.

    Returns (to_pixel, node_radius, box) where box is the pygame.Rect the
    placement region occupies on screen and node_radius is a quarter of a unit
    distance in pixels (so neurons within a unit of each other nearly touch).
    """
    (rx0, rx1), (ry0, ry1) = nodes.region
    (ex0, ex1), (ey0, ey1) = nodes.extent()
    x_min, x_max = min(rx0, ex0) - 0.5, max(rx1, ex1) + 0.5  # half a unit of breathing room
    y_min, y_max = min(ry0, ey0) - 0.5, max(ry1, ey1) + 0.5
    scale = min((width - 2 * margin) / (x_max - x_min), (height - 2 * margin) / (y_max - y_min))
    span_w, span_h = scale * (x_max - x_min), scale * (y_max - y_min)
    left, top = (width - span_w) / 2, (height - span_h) / 2

    def to_pixel(x: float, y: float) -> tuple[float, float]:
        # y grows upward in the plane but downward on the screen
        return left + (x - x_min) * scale, top + (y_max - y) * scale

    radius = max(2.0, scale * 0.25)
    bx, by = to_pixel(rx0, ry1)
    box = pygame.Rect(round(bx), round(by), round((rx1 - rx0) * scale), round((ry1 - ry0) * scale))
    return to_pixel, radius, box


def draw_nodes(surface: pygame.Surface, nodes: CartesianNodes, margin: int = 24) -> None:
    """Paint every neuron as a disc at its (x, y) position, coloured by wave like the grid."""
    width, height = surface.get_size()
    to_pixel, radius, box = node_layout(nodes, width, height, margin)
    waves = [n.fired_in_wave for n in nodes if n.fired_in_wave is not None]
    last_wave = max(waves) if waves else 0
    surface.fill(BACKGROUND)
    pygame.draw.rect(surface, UNFIRED, box, width=1)  # the random placement region, in unit distances
    for neuron in nodes:
        px, py = to_pixel(*neuron.position)
        pygame.draw.circle(surface, neuron_colour(neuron.fired_in_wave, last_wave), (px, py), radius)
        pygame.draw.circle(surface, OUTLINE, (px, py), radius, width=1)
        if neuron.fired_in_wave == 0:
            pygame.draw.circle(surface, ORIGIN_RING, (px, py), radius * 0.55, width=max(1, round(radius / 6)))


def save_nodes(nodes: CartesianNodes, path: str, width: int = 800, height: int = 600) -> None:
    surface = pygame.Surface((width, height))
    draw_nodes(surface, nodes)
    pygame.image.save(surface, path)


def show_nodes(nodes: CartesianNodes, width: int = 800, height: int = 600, fps: int = 30) -> None:
    """Open a window on a Cartesian population until Esc, Q or the window is closed."""
    pygame.init()
    try:
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(
            f"walnutbutter: {len(nodes)} nodes in a {nodes.width:g} x {nodes.height:g} unit region   [Esc] quit"
        )
        draw_nodes(screen, nodes)
        pygame.display.flip()
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
            clock.tick(fps)
    finally:
        pygame.quit()


# --- the hex grid -----------------------------------------------------------------


def draw_columns(surface: pygame.Surface, columns: HexColumns, margin: int = 24) -> None:
    """Paint a stack of hexagonal columns layer by layer, side by side, bottom layer (the input) on the left."""
    width, height = surface.get_size()
    layers = columns.layers
    xs = [n.position[0] for n in columns.all_neurons()]
    ys = [n.position[1] for n in columns.all_neurons()]
    lw, lh = (max(xs) - min(xs)) + 1.0, (max(ys) - min(ys)) + 1.0  # one layer's footprint plus half a unit all round
    gap = 0.5
    span_w, span_h = layers * lw + (layers - 1) * gap, lh
    scale = min((width - 2 * margin) / span_w, (height - 2 * margin) / span_h)
    left, top = (width - scale * span_w) / 2, (height - scale * span_h) / 2
    radius = max(2.0, scale * 0.2)
    waves = [n.fired_in_wave for n in columns.all_neurons() if n.fired_in_wave is not None]
    last_wave = max(waves) if waves else 0
    surface.fill(BACKGROUND)
    for layer in range(layers):
        x_off = left + layer * (lw + gap) * scale
        pygame.draw.rect(surface, UNFIRED, pygame.Rect(round(x_off), round(top), round(lw * scale), round(lh * scale)), width=1)
        for neuron in columns.layer(layer):
            x, y, _ = neuron.position
            px = x_off + (x - min(xs) + 0.5) * scale
            py = top + (max(ys) - y + 0.5) * scale
            pygame.draw.circle(surface, neuron_colour(neuron.fired_in_wave, last_wave), (px, py), radius)
            pygame.draw.circle(surface, OUTLINE, (px, py), radius, width=1)
            if neuron.fired_in_wave == 0:
                pygame.draw.circle(surface, ORIGIN_RING, (px, py), radius * 0.55, width=max(1, round(radius / 6)))


def as_mesh(network):
    """The object mesh to draw: an array network is synced back into its mesh first."""
    if getattr(network, "engine", "objects") == "arrays":
        network.sync_to_mesh()
        return network.mesh
    return network


def draw(surface: pygame.Surface, network, margin: int = 24) -> None:
    """Paint whichever container this is: discs on hex cells for the grid, discs at positions for nodes."""
    network = as_mesh(network)
    if isinstance(network, HexColumns):
        draw_columns(surface, network, margin)
    elif isinstance(network, CartesianNodes):
        draw_nodes(surface, network, margin)
    else:
        draw_grid(surface, network, margin)


def save(grid, path: str, width: int = 800, height: int = 600) -> None:
    """Render the network to an image file (PNG by extension). Needs no display."""
    surface = pygame.Surface((width, height))
    draw(surface, grid)
    pygame.image.save(surface, path)


def format_elapsed(seconds: float) -> str:
    """Seconds as h:mm:ss."""
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def caption(grid: GridOfNeurons, teacher: Teacher | None = None) -> str:
    grid = as_mesh(grid)
    fired = len(grid.fired_neurons())
    state = f"{fired} of {len(grid.neurons)} fired in {len(grid.waves)} waves" if fired else "unfired"
    omega = f" omega {grid.omega:g}" if getattr(grid, "omega", 0) else ""
    if isinstance(grid, CartesianNodes) and getattr(grid, "reach", None) is not None:
        omega = f" lattice, reach {grid.reach:g}"
    if isinstance(grid, HexColumns) and grid.layers > 1:
        omega = f"x{grid.layers} layers" + omega
    epoch = f" epoch {grid.epoch}:" if grid.epoch else ":"
    learning = f"   {teacher.status()}" if teacher else ""
    return f"walnutbutter {grid.across}x{grid.rows}{omega}{epoch} {state}{learning}   [Space] new input  [Esc] quit"


def caption_fast(grid: GridOfNeurons, epochs_per_second: float, fps: int, teacher: Teacher | None = None) -> str:
    return caption(grid, teacher).replace(
        "[Space] new input", f"free-running at {epochs_per_second:,.0f} epochs/s, monitored at {fps} Hz"
    )


def handle_event(
    event: pygame.event.Event, grid: GridOfNeurons, teacher: Teacher | None = None
) -> tuple[bool, bool]:
    """Apply one event to the grid. Returns (keep_running, needs_redraw)."""
    if event.type == pygame.QUIT:
        return False, False
    if event.type != pygame.KEYDOWN:
        return True, False
    if event.key in (pygame.K_ESCAPE, pygame.K_q):
        return False, False
    if event.key == pygame.K_SPACE:
        if teacher:
            teacher.epoch()  # new input with exploration noise, then reinforce
        else:
            run_epoch(grid)  # clear every neuron, draw a new random input, propagate
        return True, True
    return True, False


def show(
    grid: GridOfNeurons,
    width: int = 800,
    height: int = 600,
    fast: bool = False,
    fps: int = 30,
    teacher: Teacher | None = None,
    report_seconds: float | None = 30.0,
    log=None,
    on_report=None,
) -> None:
    """Open a window on the grid and let the keyboard drive it.

    Space resets every neuron and runs a new epoch with a fresh random input.
    Esc or Q closes the window. With a `teacher`, every epoch is followed by a
    teaching step and the title bar reports the running accuracy. While
    free-running with a teacher, a progress line goes to `log` (default:
    stderr) every `report_seconds`, and `on_report`, if given, is called
    right after it (used to write checkpoints); None disables both.

    With `fast` the window becomes a monitor: the system runs epoch after
    epoch as fast as it can, silently, and the window samples its state `fps`
    times a second. Every sample is a completed epoch. Returns when the window
    is closed; the grid keeps the state of its last epoch.
    """
    pygame.init()
    from .neuron import Neuron  # local import: only needed to silence the free run

    was_verbose = Neuron.verbose
    try:
        screen = pygame.display.set_mode((width, height))
        clock = pygame.time.Clock()
        frame_time = 1.0 / fps
        next_frame = time.perf_counter() + frame_time
        epochs_per_second = 0.0
        needs_redraw = True
        running = True
        started = time.perf_counter()
        last_report = started
        log = log or (lambda line: print(line, file=sys.stderr, flush=True))
        if fast:
            Neuron.verbose = False
        while running:
            if needs_redraw:
                draw(screen, grid)
                pygame.display.set_caption(
                    caption_fast(grid, epochs_per_second, fps, teacher) if fast else caption(grid, teacher)
                )
                pygame.display.flip()
                needs_redraw = False
            for event in pygame.event.get():
                running, changed = handle_event(event, grid, teacher)
                needs_redraw = needs_redraw or changed
                if not running:
                    break
            if not running:
                break
            if fast:
                # Let the system run until the monitor's next sample is due.
                count = 0
                while time.perf_counter() < next_frame:
                    if teacher:
                        teacher.epoch(verbose=False)
                    else:
                        run_epoch(grid, verbose=False)
                    count += 1
                epochs_per_second = 0.8 * epochs_per_second + 0.2 * count * fps if epochs_per_second else count * fps
                now = time.perf_counter()
                if teacher and report_seconds is not None and now - last_report >= report_seconds:
                    last_report = now
                    teacher.record(now - started, epochs_per_second)
                    log(f"[{format_elapsed(now - started)}] epoch {grid.epoch:,}: {teacher.status()}, {epochs_per_second:,.0f} epochs/s")
                    if on_report:
                        on_report()
                next_frame += frame_time
                if time.perf_counter() > next_frame:  # drawing took longer than a frame; don't try to catch up
                    next_frame = time.perf_counter() + frame_time
                needs_redraw = True
            else:
                clock.tick(fps)
    finally:
        Neuron.verbose = was_verbose
        pygame.quit()
