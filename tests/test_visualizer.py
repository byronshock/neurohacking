import math

import pygame
import pytest

from neurohacking.cli import cli_main
from neurohacking.grid import GridOfNeurons
from neurohacking import visualizer as viz


def test_origin_maps_to_zero():
    assert viz.axial_to_pixel(0, 0, 10) == (0, 0)


def test_all_six_neighbours_are_equally_spaced():
    grid = GridOfNeurons(size=1)
    spacing = 10 * viz.SQRT3  # centre-to-centre distance for pointy-top hexagons
    for dq, dr in grid.directions:
        x, y = viz.axial_to_pixel(dq, dr, 10)
        assert math.hypot(x, y) == pytest.approx(spacing)


def test_hexagon_corners_lie_on_the_radius():
    points = viz.hexagon_points(50, 50, 20)
    assert len(points) == 6
    for x, y in points:
        assert math.hypot(x - 50, y - 50) == pytest.approx(20)


def test_fitted_grid_stays_inside_the_surface():
    size, width, height = 4, 640, 480
    radius = viz.fit_hex_radius(size, width, height, margin=10)
    for q, r in GridOfNeurons(size=size).neurons:
        dx, dy = viz.axial_to_pixel(q, r, radius)
        for x, y in viz.hexagon_points(width / 2 + dx, height / 2 + dy, radius):
            assert 10 <= x <= width - 10 + 1e-6
            assert 10 <= y <= height - 10 + 1e-6


def test_unfired_is_grey_and_fired_shades_by_wave():
    assert viz.neuron_colour(None, 3) == viz.UNFIRED
    assert viz.neuron_colour(0, 3) == viz.FIRED_CENTRE
    assert viz.neuron_colour(3, 3) == viz.FIRED_EDGE
    assert viz.neuron_colour(0, 0) == viz.FIRED_CENTRE  # a single wave must not divide by zero


def test_draw_grid_paints_fired_and_unfired_neurons(capsys):
    grid = GridOfNeurons(size=2)
    surface = pygame.Surface((300, 300))

    viz.draw_grid(surface, grid)  # nothing fired yet
    radius = viz.fit_hex_radius(2, 300, 300)
    dx, dy = viz.axial_to_pixel(1, 0, radius)
    neighbour_pixel = (round(150 + dx), round(150 + dy))
    assert surface.get_at(neighbour_pixel)[:3] == viz.UNFIRED
    assert surface.get_at((2, 2))[:3] == viz.BACKGROUND

    grid.activate_origin()
    viz.draw_grid(surface, grid)
    assert surface.get_at(neighbour_pixel)[:3] != viz.UNFIRED


def test_save_writes_an_image_file(tmp_path, capsys):
    grid = GridOfNeurons(size=2)
    grid.activate_origin()
    out = tmp_path / "grid.png"
    viz.save(grid, str(out), width=200, height=200)
    assert out.exists() and out.stat().st_size > 0
    assert pygame.image.load(str(out)).get_size() == (200, 200)


def test_cli_save_option(tmp_path, capsys):
    out = tmp_path / "cli.png"
    assert cli_main(["--size", "1", "--save", str(out)]) == 0
    assert out.exists()
    assert "Saved" in capsys.readouterr().err
