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
HOT = (255, 48, 32)  # a neuron that fired at the end of the epoch...
COOL = (40, 96, 255)  # ...through to one that last fired an epoch or more ago
TRACE_BACKGROUND = (16, 16, 20)
TRACE_TICK = (255, 200, 80)
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


def neuron_colour(fired_at: float | None, now: float, span: float):
    """Grey if the neuron has never fired; otherwise hot (red) for a spike at `now` cooling to blue over `span` ms."""
    if fired_at is None:
        return UNFIRED
    age = max(0.0, now - fired_at)
    return _lerp_colour(HOT, COOL, min(1.0, age / span) if span > 0 else 1.0)


def heat(network) -> tuple[float, float]:
    """`now` and `span` for colouring: the end of the epoch, cooling over one epoch's length."""
    return network.horizon, network.interval or 1.0


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

    now, span = heat(grid)

    disc = hex_radius * SQRT3 / 2 * DISC_FILL
    outline = max(1, round(hex_radius / 12))
    surface.fill(BACKGROUND)
    for (q, r), neuron in grid.neurons.items():
        dx, dy = axial_to_pixel(q, r, hex_radius)
        centre = (offset_x + dx, offset_y + dy)
        pygame.draw.circle(surface, neuron_colour(neuron.fired_at, now, span), centre, disc)
        pygame.draw.circle(surface, OUTLINE, centre, disc, width=outline)

    # Ring the stimulus: the neurons that fired in wave 0, or the input neurons
    # that will be forced when the mesh is fired, or failing both the origin.
    stimulus = [n for n in grid.neurons.values() if n.forced] or grid.input_neurons()
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
    now, span = heat(nodes)
    surface.fill(BACKGROUND)
    pygame.draw.rect(surface, UNFIRED, box, width=1)  # the random placement region, in unit distances
    for neuron in nodes:
        px, py = to_pixel(*neuron.position)
        pygame.draw.circle(surface, neuron_colour(neuron.fired_at, now, span), (px, py), radius)
        pygame.draw.circle(surface, OUTLINE, (px, py), radius, width=1)
        if neuron.forced:
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
    now, span = heat(columns)
    surface.fill(BACKGROUND)
    for layer in range(layers):
        x_off = left + layer * (lw + gap) * scale
        pygame.draw.rect(surface, UNFIRED, pygame.Rect(round(x_off), round(top), round(lw * scale), round(lh * scale)), width=1)
        for neuron in columns.layer(layer):
            x, y, _ = neuron.position
            px = x_off + (x - min(xs) + 0.5) * scale
            py = top + (max(ys) - y + 0.5) * scale
            pygame.draw.circle(surface, neuron_colour(neuron.fired_at, now, span), (px, py), radius)
            pygame.draw.circle(surface, OUTLINE, (px, py), radius, width=1)
            if neuron.forced:
                pygame.draw.circle(surface, ORIGIN_RING, (px, py), radius * 0.55, width=max(1, round(radius / 6)))


def as_mesh(network):
    """The object mesh to draw: an array network is synced back into its mesh first."""
    if getattr(network, "engine", "objects") == "arrays":
        network.sync_to_mesh()
        return network.mesh
    return network


def draw(surface: pygame.Surface, network, margin: int = 24, trace: bool = False) -> None:
    """Paint whichever container this is: discs on hex cells for the grid, discs at positions for nodes.

    With `trace`, the bottom quarter of the surface shows the current epoch's
    trace: a raster of every spike, time across, neuron down.
    """
    network = as_mesh(network)
    target = surface
    if trace:
        width, height = surface.get_size()
        split = round(height * 0.72)
        target = surface.subsurface(pygame.Rect(0, 0, width, split))
        draw_trace(surface.subsurface(pygame.Rect(0, split, width, height - split)), network)
    if isinstance(network, HexColumns):
        draw_columns(target, network, margin)
    elif isinstance(network, CartesianNodes):
        draw_nodes(target, network, margin)
    else:
        draw_grid(target, network, margin)


def draw_trace(surface: pygame.Surface, network, margin: int = 12) -> None:
    """The current epoch as a raster: each wave a column at its time, a tick per neuron that fired, forced ones ringed white."""
    width, height = surface.get_size()
    surface.fill(TRACE_BACKGROUND)
    neurons = list(network.all_neurons())
    index = {neuron: i for i, neuron in enumerate(neurons)}
    start, end = network.time, network.horizon
    span = end - start if end > start else 1.0
    left, right = margin, width - margin
    top, bottom = margin, height - margin
    pygame.draw.line(surface, UNFIRED, (left, bottom), (right, bottom), 1)
    for tick in range(int(span) + 1):  # a mark per millisecond
        x = left + (right - left) * tick / span
        pygame.draw.line(surface, UNFIRED, (x, bottom), (x, bottom + 3), 1)
    row_height = (bottom - top) / max(1, len(neurons))
    for wave in network.waves:
        x = left + (right - left) * (wave.time - start) / span
        for neuron in wave.fired:
            y = top + row_height * index[neuron]
            colour = ORIGIN_RING if neuron.forced and wave.time == start else TRACE_TICK
            pygame.draw.line(surface, colour, (x, y), (x, y + max(1.0, row_height - 1)), 2)


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
    epoch = f" epoch {grid.epoch} at {grid.time:g} ms:" if grid.epoch else ":"
    dopamine = getattr(grid, "dopamine", None)
    learning = f"   {teacher.status()}" if teacher else f"   {dopamine.status()}" if dopamine else ""
    return f"walnutbutter {grid.across}x{grid.rows}{omega}{epoch} {state}{learning}   [Space] new input  [Esc] quit"


def caption_fast(
    grid: GridOfNeurons, epochs_per_second: float, fps: int, teacher: Teacher | None = None, paused: bool = False
) -> str:
    if paused:
        return caption(grid, teacher).replace("[Space] new input", "paused at the end of this epoch, its trace below  [Space] resume")
    return caption(grid, teacher).replace(
        "[Space] new input", f"free-running at {epochs_per_second:,.0f} epochs/s, monitored at {fps} Hz  [Space] pause"
    )


def handle_event(
    event: pygame.event.Event, grid: GridOfNeurons, teacher: Teacher | None = None, state: dict | None = None
) -> tuple[bool, bool]:
    """Apply one event to the grid. Returns (keep_running, needs_redraw).

    Stepping (no `state`): Space runs one epoch. Free-running (`state` with
    a "paused" key): Space pauses at the end of the current epoch, showing
    its trace, and resumes.
    """
    if event.type == pygame.QUIT:
        return False, False
    if event.type != pygame.KEYDOWN:
        return True, False
    if event.key in (pygame.K_ESCAPE, pygame.K_q):
        return False, False
    if event.key == pygame.K_SPACE:
        if state is not None:
            state["paused"] = not state["paused"]
            return True, True
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
    noise: float | None = None,
    rng=None,
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
        state = {"paused": False} if fast else None
        if fast:
            Neuron.verbose = False
        while running:
            if needs_redraw:
                paused = bool(state and state["paused"])
                draw(screen, grid, trace=paused)
                pygame.display.set_caption(
                    caption_fast(grid, epochs_per_second, fps, teacher, paused) if fast else caption(grid, teacher)
                )
                pygame.display.flip()
                needs_redraw = False
            for event in pygame.event.get():
                running, changed = handle_event(event, grid, teacher, state)
                needs_redraw = needs_redraw or changed
                if not running:
                    break
            if not running:
                break
            if fast and state["paused"]:
                clock.tick(fps)  # hold at the end of the epoch, its trace on show
                continue
            if fast:
                # Let the system run until the monitor's next sample is due.
                count = 0
                while time.perf_counter() < next_frame:
                    if teacher:
                        teacher.epoch(verbose=False)
                    else:
                        run_epoch(grid, verbose=False, noise=noise or 0.0, rng=rng)
                    count += 1
                epochs_per_second = 0.8 * epochs_per_second + 0.2 * count * fps if epochs_per_second else count * fps
                now = time.perf_counter()
                if report_seconds is not None and now - last_report >= report_seconds:
                    last_report = now
                    if teacher:
                        teacher.record(now - started, epochs_per_second)
                        status = teacher.status()
                    else:
                        dopamine = getattr(grid, "dopamine", None)
                        status = f"{grid.total_spikes():,} spikes" + (f", {dopamine.status()}" if dopamine else "")
                    log(f"[{format_elapsed(now - started)}] epoch {grid.epoch:,}: {status}, {epochs_per_second:,.0f} epochs/s")
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
