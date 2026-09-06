import math

import pygame
import pytest

from neurohacking.cli import cli_main
from neurohacking.grid import DIRECTIONS, GridOfNeurons
from neurohacking import visualizer as viz


def test_origin_maps_to_zero():
    assert viz.axial_to_pixel(0, 0, 10) == (0, 0)


def test_all_six_neighbours_are_equally_spaced():
    spacing = 10 * viz.SQRT3  # centre-to-centre distance for pointy-top hexagons
    for dq, dr in DIRECTIONS:
        x, y = viz.axial_to_pixel(dq, dr, 10)
        assert math.hypot(x, y) == pytest.approx(spacing)


def test_hexagon_corners_lie_on_the_radius():
    points = viz.hexagon_points(50, 50, 20)
    assert len(points) == 6
    for x, y in points:
        assert math.hypot(x - 50, y - 50) == pytest.approx(20)


@pytest.mark.parametrize("columns, rows", [(9, 5), (3, 11), (1, 1), (24, 20)])
def test_layout_keeps_every_whole_hexagon_inside_the_margin(columns, rows):
    width, height, margin = 640, 480, 10
    grid = GridOfNeurons(columns=columns, rows=rows)
    radius, ox, oy = viz.layout(grid, width, height, margin)
    assert radius > 0
    xs, ys = [], []
    for q, r in grid.neurons:
        dx, dy = viz.axial_to_pixel(q, r, radius)
        for x, y in viz.hexagon_points(ox + dx, oy + dy, radius):
            assert margin - 1e-6 <= x <= width - margin + 1e-6
            assert margin - 1e-6 <= y <= height - margin + 1e-6
            xs.append(x)
            ys.append(y)
    # The grid fills the window in at least one direction and is centred in both.
    assert min(xs) == pytest.approx(margin) or min(ys) == pytest.approx(margin)
    assert (min(xs) + max(xs)) / 2 == pytest.approx(width / 2)
    assert (min(ys) + max(ys)) / 2 == pytest.approx(height / 2)


def test_default_grid_fills_the_default_window_in_both_directions():
    radius, _, _ = viz.layout(GridOfNeurons(columns=24, rows=20), 800, 600)
    grid_w = radius * viz.SQRT3 * 24.5  # 24 columns plus the half-cell shift of odd rows
    grid_h = radius * (1.5 * 19 + 2)
    assert grid_w == pytest.approx(752)  # 800 minus two 24px margins
    assert grid_h / (600 - 48) > 0.95


def test_unfired_is_grey_and_fired_shades_by_wave():
    assert viz.neuron_colour(None, 3) == viz.UNFIRED
    assert viz.neuron_colour(0, 3) == viz.FIRED_CENTRE
    assert viz.neuron_colour(3, 3) == viz.FIRED_EDGE
    assert viz.neuron_colour(0, 0) == viz.FIRED_CENTRE  # a single wave must not divide by zero


def test_draw_grid_paints_fired_and_unfired_neurons(capsys):
    grid = GridOfNeurons(columns=5, rows=5)
    surface = pygame.Surface((300, 300))

    viz.draw_grid(surface, grid)  # nothing fired yet
    radius, ox, oy = viz.layout(grid, 300, 300)
    dx, dy = viz.axial_to_pixel(1, 0, radius)
    neighbour_pixel = (round(ox + dx), round(oy + dy))
    assert surface.get_at(neighbour_pixel)[:3] == viz.UNFIRED
    assert surface.get_at((2, 2))[:3] == viz.BACKGROUND

    grid.activate_origin()
    viz.draw_grid(surface, grid)
    assert surface.get_at(neighbour_pixel)[:3] != viz.UNFIRED


def test_save_writes_an_image_file(tmp_path, capsys):
    grid = GridOfNeurons(columns=5, rows=5)
    grid.activate_origin()
    out = tmp_path / "grid.png"
    viz.save(grid, str(out), width=200, height=200)
    assert out.exists() and out.stat().st_size > 0
    assert pygame.image.load(str(out)).get_size() == (200, 200)


def test_cli_save_option(tmp_path, capsys):
    out = tmp_path / "cli.png"
    assert cli_main(["--columns", "3", "--rows", "3", "--save", str(out)]) == 0
    assert out.exists()
    assert "Saved" in capsys.readouterr().err


def test_cli_window_option_sets_image_size(tmp_path, capsys):
    out = tmp_path / "wide.png"
    assert cli_main(["--columns", "3", "--rows", "3", "--save", str(out), "--window", "320", "200"]) == 0
    assert pygame.image.load(str(out)).get_size() == (320, 200)


# --- the interactive window ---------------------------------------------------


def key(k):
    return pygame.event.Event(pygame.KEYDOWN, key=k)


def test_space_fires_origin_and_r_resets(capsys):
    grid = GridOfNeurons(columns=3, rows=3)
    assert viz.handle_event(key(pygame.K_SPACE), grid) == (True, True)
    assert len(grid.fired_neurons()) == 9
    assert viz.handle_event(key(pygame.K_SPACE), grid) == (True, False)  # already fired: nothing to do
    assert viz.handle_event(key(pygame.K_r), grid) == (True, True)
    assert grid.fired_neurons() == [] and grid.waves == []


def test_quit_keys_and_window_close_stop_the_loop():
    grid = GridOfNeurons(columns=3, rows=3)
    assert viz.handle_event(key(pygame.K_ESCAPE), grid) == (False, False)
    assert viz.handle_event(key(pygame.K_q), grid) == (False, False)
    assert viz.handle_event(pygame.event.Event(pygame.QUIT), grid) == (False, False)
    assert viz.handle_event(key(pygame.K_x), grid) == (True, False)  # unknown key ignored
    assert grid.fired_neurons() == []


def test_caption_reports_state(capsys):
    grid = GridOfNeurons(columns=3, rows=3)
    assert viz.caption(grid).startswith("neurohacking 3x3: unfired")
    grid.activate_origin()
    assert "9 of 9 fired in 4 waves" in viz.caption(grid)  # waves 0-2 fire; wave 3 delivers to already-fired cells


def test_show_opens_on_the_unfired_mesh_and_returns_on_quit(monkeypatch, capsys):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    grid = GridOfNeurons(columns=3, rows=3)
    seen = []
    scripted = [[key(pygame.K_SPACE)], [pygame.event.Event(pygame.QUIT)]]

    def fake_get():
        seen.append(viz.caption(grid))
        return scripted.pop(0) if scripted else [pygame.event.Event(pygame.QUIT)]

    monkeypatch.setattr(pygame.event, "get", fake_get)
    viz.show(grid, 200, 150)
    assert seen[0].startswith("neurohacking 3x3: unfired")  # first frame drawn before any key
    assert len(grid.fired_neurons()) == 9  # Space fired it; state survives closing


def test_cli_show_opens_before_firing(monkeypatch, capsys):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    fired_when_shown = []

    def fake_show(grid, width, height):
        fired_when_shown.append(len(grid.fired_neurons()))

    monkeypatch.setattr(viz, "show", fake_show)
    assert cli_main(["--columns", "3", "--rows", "3", "--show"]) == 0
    assert fired_when_shown == [0]
    assert "0 of 9 neurons fired in 0 waves" in capsys.readouterr().err
