import math

import pytest

from walnutbutter.cartesian import UNIT_SQUARE, CartesianNodes
from walnutbutter.neuron import Neuron
from walnutbutter.propagation import propagate


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(Neuron, "verbose", False)


def test_default_bounds_are_the_unit_square():
    nodes = CartesianNodes()
    assert nodes.bounds == UNIT_SQUARE == ((-1.0, 1.0), (-1.0, 1.0))
    assert len(nodes) == 0


def test_neurons_without_coordinates_are_placed_at_random_inside_the_bounds():
    nodes = CartesianNodes(count=500, seed=1)
    xs = [x for x, _ in nodes.positions()]
    ys = [y for _, y in nodes.positions()]
    assert len(nodes) == 500
    assert all(-1 <= x <= 1 and -1 <= y <= 1 for x, y in nodes.positions())
    assert min(xs) < -0.9 and max(xs) > 0.9 and min(ys) < -0.9 and max(ys) > 0.9  # spread across the box
    assert len(set(nodes.positions())) == 500  # all distinct
    assert abs(sum(xs) / 500) < 0.1 and abs(sum(ys) / 500) < 0.1  # roughly centred


def test_explicit_coordinates_are_kept_exactly():
    nodes = CartesianNodes()
    n = nodes.add(0.25, -0.5, name="corner")
    assert n.position == (0.25, -0.5) and n.name == "corner"
    assert nodes[0] is n


def test_one_coordinate_given_and_the_other_random():
    nodes = CartesianNodes(seed=2)
    a = nodes.add(x=0.0)
    b = nodes.add(y=1.0)
    assert a.position[0] == 0.0 and -1 <= a.position[1] <= 1
    assert b.position[1] == 1.0 and -1 <= b.position[0] <= 1


def test_bounds_are_inclusive_and_enforced():
    nodes = CartesianNodes()
    nodes.add(-1.0, 1.0)
    nodes.add(1.0, -1.0)
    with pytest.raises(ValueError):
        nodes.add(1.0001, 0.0)
    with pytest.raises(ValueError):
        nodes.add(0.0, -2.0)
    assert nodes.contains(0.5, 0.5) and not nodes.contains(0.5, 1.5)


def test_custom_bounds():
    nodes = CartesianNodes(count=50, bounds=((0, 10), (-5, 5)), seed=3)
    assert all(0 <= x <= 10 and -5 <= y <= 5 for x, y in nodes.positions())
    with pytest.raises(ValueError):
        CartesianNodes(bounds=((1, 0), (0, 1)))
    with pytest.raises(ValueError):
        CartesianNodes(count=-1)


def test_same_seed_gives_same_positions():
    a = CartesianNodes(count=20, seed=7)
    b = CartesianNodes(count=20, seed=7)
    c = CartesianNodes(count=20, seed=8)
    assert a.positions() == b.positions()
    assert a.positions() != c.positions()


def test_default_names_and_thresholds():
    nodes = CartesianNodes(count=3, threshold=0.5)
    assert [n.name for n in nodes] == ["Node_0", "Node_1", "Node_2"]
    assert all(n.threshold == 0.5 for n in nodes)
    assert nodes.add(threshold=2.0).threshold == 2.0


def test_distance_nearest_and_within():
    nodes = CartesianNodes()
    origin = nodes.add(0.0, 0.0)
    near = nodes.add(0.1, 0.0)
    far = nodes.add(0.9, 0.9)
    assert CartesianNodes.distance(origin, near) == pytest.approx(0.1)
    assert nodes.nearest(0.0, 0.0) == [origin]
    assert nodes.nearest(0.0, 0.0, count=2, exclude=origin) == [near, far]
    assert nodes.within(0.0, 0.0, radius=0.5) == [origin, near]
    assert nodes.within(0.0, 0.0, radius=0.5, exclude=origin) == [near]


def test_neurons_can_be_connected_and_propagated_like_any_others():
    nodes = CartesianNodes(count=3, seed=1)
    a, b, c = nodes
    a.connect(b, 1, weight=1.0)
    b.connect(c, 2, weight=1.0)
    waves = propagate(fire=[a])
    assert [n.fired_in_wave for n in nodes] == [0, 1, 2]
    assert nodes.fired_neurons() == [a, b, c]
    nodes.reset()
    assert nodes.fired_neurons() == []


def test_repr_and_iteration():
    nodes = CartesianNodes(count=2, seed=1)
    assert repr(nodes) == "CartesianNodes(2 neurons in [-1, 1] x [-1, 1])"
    assert list(nodes) == nodes.neurons


def test_cli_nodes_headless_lists_positions(capsys):
    from walnutbutter.cli import cli_main
    assert cli_main(["--headless", "--nodes", "5", "--seed", "1"]) == 0
    captured = capsys.readouterr()
    assert captured.out.count("Node_") == 5
    assert "5 nodes placed at random" in captured.err and "seed 1" in captured.err


def test_cli_nodes_defaults_to_64(capsys):
    from walnutbutter.cli import cli_main
    assert cli_main(["--headless", "--nodes", "--seed", "1"]) == 0
    captured = capsys.readouterr()
    assert captured.out.count("Node_") == 64 and "64 nodes placed" in captured.err


def test_cli_nodes_rejects_zero(capsys):
    from walnutbutter.cli import cli_main
    assert cli_main(["--headless", "--nodes", "0"]) == 2
    assert "at least 1" in capsys.readouterr().err


def test_cli_nodes_saves_a_picture_and_opens_the_window(tmp_path, monkeypatch, capsys):
    import pygame
    from walnutbutter import visualizer as viz
    from walnutbutter.cli import cli_main
    out = tmp_path / "nodes.png"
    assert cli_main(["--headless", "--nodes", "20", "--seed", "1", "--save", str(out)]) == 0
    assert pygame.image.load(str(out)).get_size() == (800, 600)
    shown = []
    monkeypatch.setattr(viz, "show_nodes", lambda nodes, w, h: shown.append((len(nodes), w, h)))
    assert cli_main(["--nodes", "20", "--seed", "1", "--window", "400", "300"]) == 0
    assert shown == [(20, 400, 300)]


def test_draw_nodes_places_discs_at_their_positions():
    import pygame
    from walnutbutter import visualizer as viz
    nodes = CartesianNodes()
    corner = nodes.add(-1.0, 1.0)  # top-left of the box
    nodes.add(0.0, 0.0)
    nodes.add(1.0, -1.0)  # bottom-right
    surface = pygame.Surface((400, 300))
    viz.draw_nodes(surface, nodes)
    to_pixel, radius, box = viz.node_layout(nodes, 400, 300)
    assert to_pixel(0.0, 0.0) == (200, 150)  # the box is centred
    assert box.width == box.height == 300 - 48  # a square box keeps its shape: limited by the height
    assert box.center == (200, 150)
    cx, cy = to_pixel(-1.0, 1.0)
    assert cx < 200 and cy < 150  # x left, y up on screen
    assert surface.get_at((round(cx), round(cy)))[:3] == viz.UNFIRED
    assert surface.get_at((2, 2))[:3] == viz.BACKGROUND
    corner.fire()
    viz.draw_nodes(surface, nodes)
    assert surface.get_at((round(cx), round(cy)))[:3] != viz.UNFIRED


def test_show_nodes_returns_on_quit(monkeypatch):
    import pygame
    from walnutbutter import visualizer as viz
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    nodes = CartesianNodes(count=10, seed=1)
    monkeypatch.setattr(pygame.event, "get", lambda: [pygame.event.Event(pygame.QUIT)])
    viz.show_nodes(nodes, 200, 150)
