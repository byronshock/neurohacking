import math

import pytest

from walnutbutter.cartesian import CartesianNodes
from walnutbutter.neuron import Neuron
from walnutbutter.propagation import propagate


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(Neuron, "verbose", False)


def test_default_is_an_8x10_hexagonal_lattice_at_unit_spacing():
    from walnutbutter.cartesian import ROW_SPACING
    nodes = CartesianNodes()
    assert nodes.layout == "hex" and (nodes.columns, nodes.rows) == (8, 10) and len(nodes) == 80
    a, b = nodes.node_at(0, 0), nodes.node_at(1, 0)
    assert CartesianNodes.distance(a, b) == pytest.approx(1.0)  # horizontal neighbours one unit apart
    up = nodes.node_at(0, 1)
    assert up.position[0] - a.position[0] == pytest.approx(0.5)  # odd rows shifted half a unit
    assert up.position[1] - a.position[1] == pytest.approx(ROW_SPACING)
    assert CartesianNodes.distance(a, up) == pytest.approx(1.0)  # diagonal neighbours too
    xs = [x for x, _ in nodes.positions()]
    ys = [y for _, y in nodes.positions()]
    assert (min(xs) + max(xs)) / 2 == pytest.approx(0.0) and (min(ys) + max(ys)) / 2 == pytest.approx(0.0)
    assert nodes.node_at(8, 0) is None and nodes.node_at(0, 10) is None


def test_lattice_interior_neurons_have_exactly_six_neighbours_within_a_unit():
    nodes = CartesianNodes(columns=6, rows=6)
    counts = {}
    for r in range(6):
        for c in range(6):
            counts[(c, r)] = len(nodes.neighbours(nodes.node_at(c, r)))
    interior = [counts[(c, r)] for r in range(1, 5) for c in range(1, 5) if not (r % 2 == 0 and c == 1) and not (r % 2 == 1 and c == 4)]
    assert counts[(2, 2)] == 6 and counts[(3, 3)] == 6
    assert all(2 <= n <= 6 for n in counts.values())
    assert sum(1 for n in counts.values() if n == 6) >= 12  # the interior is fully connected at radius 1
    # nothing else is within a unit: the next ring starts at sqrt(3)
    centre = nodes.node_at(2, 2)
    assert len(nodes.neighbours(centre, radius=1.7)) == 6 and len(nodes.neighbours(centre, radius=1.8)) > 6


def test_lattice_region_covers_the_footprint_and_names_follow_column_row():
    nodes = CartesianNodes(columns=4, rows=3)
    assert all(nodes.in_region(x, y) for x, y in nodes.positions())
    assert nodes.node_at(3, 2).name == "Node_3_2" and nodes[0] is nodes.node_at(0, 0)
    with pytest.raises(ValueError):
        CartesianNodes(columns=0, rows=3)
    with pytest.raises(ValueError):
        CartesianNodes(count=5)  # count belongs to the random layout
    with pytest.raises(ValueError):
        CartesianNodes(layout="spiral")


def test_random_layout_default_region_is_eight_by_ten_units():
    nodes = CartesianNodes(layout="random", count=0)
    assert (nodes.width, nodes.height) == (8.0, 10.0)
    assert nodes.region == ((-4.0, 4.0), (-5.0, 5.0))
    assert len(nodes) == 0 and nodes.node_at(0, 0) is None


def test_random_neurons_fill_the_region_at_about_one_per_unit_area():
    nodes = CartesianNodes(layout="random", count=800, seed=1)
    xs = [x for x, _ in nodes.positions()]
    ys = [y for _, y in nodes.positions()]
    assert all(nodes.in_region(x, y) for x, y in nodes.positions())
    assert min(xs) < -3.8 and max(xs) > 3.8 and min(ys) < -4.8 and max(ys) > 4.8
    assert abs(sum(xs) / 800) < 0.3 and abs(sum(ys) / 800) < 0.3
    assert len(set(nodes.positions())) == 800


def test_neurons_can_be_several_units_apart_and_outside_the_region():
    nodes = CartesianNodes(layout="random", count=0)
    a = nodes.add(0.0, 0.0)
    b = nodes.add(6.0, 8.0)  # ten units away, and outside the 8 x 10 region
    far = nodes.add(30.0, -30.0)
    assert CartesianNodes.distance(a, b) == pytest.approx(10.0)
    assert not nodes.in_region(*b.position) and not nodes.in_region(*far.position)
    (x0, x1), (y0, y1) = nodes.extent()
    assert (x0, x1, y0, y1) == (0.0, 30.0, -30.0, 8.0)


def test_explicit_coordinates_and_partial_coordinates():
    nodes = CartesianNodes(layout="random", count=0, seed=2)
    n = nodes.add(0.25, -0.5, name="corner")
    assert n.position == (0.25, -0.5) and n.name == "corner" and nodes[0] is n
    a = nodes.add(x=0.0)
    b = nodes.add(y=1.0)
    assert a.position[0] == 0.0 and -5 <= a.position[1] <= 5
    assert b.position[1] == 1.0 and -4 <= b.position[0] <= 4


def test_custom_region_and_validation():
    nodes = CartesianNodes(layout="random", count=50, width=20, height=2, seed=3)
    assert all(-10 <= x <= 10 and -1 <= y <= 1 for x, y in nodes.positions())
    with pytest.raises(ValueError):
        CartesianNodes(layout="random", width=0, height=1)
    with pytest.raises(ValueError):
        CartesianNodes(layout="random", count=-1)


def test_same_seed_gives_same_positions():
    a, b, c = (CartesianNodes(layout="random", count=20, seed=s) for s in (7, 7, 8))
    assert a.positions() == b.positions() and a.positions() != c.positions()


def test_names_thresholds_and_floor():
    nodes = CartesianNodes(layout="random", count=3, threshold=0.5, minimum_potential=-0.3)
    assert [n.name for n in nodes] == ["Node_0", "Node_1", "Node_2"]
    assert all(n.threshold == 0.5 and n.minimum_potential == -0.3 for n in nodes)
    assert nodes.add(threshold=2.0).threshold == 2.0


def test_within_and_neighbours_use_unit_distances():
    nodes = CartesianNodes(layout="random", count=0)
    origin = nodes.add(0.0, 0.0)
    close = nodes.add(0.8, 0.0)  # inside one unit
    edge = nodes.add(0.0, 1.0)  # exactly one unit
    far = nodes.add(3.0, 3.0)
    assert nodes.neighbours(origin) == [close, edge]  # default radius is one unit
    assert nodes.neighbours(origin, radius=5.0) == [close, edge, far]
    assert nodes.within(0.0, 0.0, radius=0.5) == [origin]
    assert nodes.nearest(0.0, 0.0, count=2, exclude=origin) == [close, edge]


def test_neurons_can_be_connected_and_propagated_like_any_others():
    nodes = CartesianNodes(layout="random", count=3, seed=1)
    a, b, c = nodes
    a.connect(b, 1, weight=1.0)
    b.connect(c, 2, weight=1.0)
    propagate(fire=[a])
    assert [n.fired_in_wave for n in nodes] == [0, 1, 2]
    nodes.reset()
    assert nodes.fired_neurons() == [] and all(n.potential == 0 for n in nodes)


def test_repr_and_iteration():
    nodes = CartesianNodes(layout="random", count=2, seed=1)
    assert repr(nodes) == "CartesianNodes(2 neurons at random in a 8 x 10 unit region)"
    assert repr(CartesianNodes()) == "CartesianNodes(8x10 hexagonal lattice, 80 neurons at unit spacing)"
    assert list(nodes) == nodes.neurons


# --- command line and drawing ------------------------------------------------------


def test_cli_nodes_headless_lists_positions(capsys):
    from walnutbutter.cli import cli_main
    assert cli_main(["--headless", "--nodes", "5", "--seed", "1"]) == 0
    captured = capsys.readouterr()
    assert captured.out.count("Node_") == 5
    assert "5 neurons at random in a 8 x 10 unit region" in captured.err and "seed 1" in captured.err


def test_cli_bare_nodes_is_a_lattice_sized_by_columns_and_rows(capsys):
    from walnutbutter.cli import cli_main
    assert cli_main(["--headless", "--nodes", "--columns", "6", "--rows", "4"]) == 0
    captured = capsys.readouterr()
    assert captured.out.count("Node_") == 24 and "6x4 hexagonal lattice, 24 neurons at unit spacing" in captured.err
    assert cli_main(["--headless", "--nodes"]) == 0
    assert "8x10 hexagonal lattice, 80 neurons" in capsys.readouterr().err


def test_cli_nodes_rejects_negative(capsys):
    from walnutbutter.cli import cli_main
    assert cli_main(["--headless", "--nodes", "-3"]) == 2
    assert "cannot be negative" in capsys.readouterr().err


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


def test_draw_nodes_scales_unit_distances_and_includes_outliers():
    import pygame
    from walnutbutter import visualizer as viz
    nodes = CartesianNodes(layout="random", count=0, width=8, height=8)
    a = nodes.add(0.0, 0.0)
    nodes.add(1.0, 0.0)
    surface = pygame.Surface((400, 400))
    to_pixel, radius, box = viz.node_layout(nodes, 400, 400)
    ax, ay = to_pixel(0.0, 0.0)
    bx, by = to_pixel(1.0, 0.0)
    unit_px = bx - ax
    assert ay == by and unit_px > 0
    assert radius == pytest.approx(unit_px * 0.25)  # a neuron is a quarter of a unit across
    assert box.width == pytest.approx(8 * unit_px, abs=1) and box.height == pytest.approx(8 * unit_px, abs=1)
    viz.draw_nodes(surface, nodes)
    assert surface.get_at((round(ax), round(ay)))[:3] == viz.UNFIRED
    a.fire()
    viz.draw_nodes(surface, nodes)
    assert surface.get_at((round(ax), round(ay)))[:3] != viz.UNFIRED
    # an outlier far outside the region pulls the scale in, but stays on screen
    nodes.add(20.0, 0.0)
    to_pixel2, _, _ = viz.node_layout(nodes, 400, 400)
    px, _ = to_pixel2(20.0, 0.0)
    assert 0 <= px <= 400 and to_pixel2(1.0, 0.0)[0] - to_pixel2(0.0, 0.0)[0] < unit_px


def test_show_nodes_returns_on_quit(monkeypatch):
    import pygame
    from walnutbutter import visualizer as viz
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    nodes = CartesianNodes(columns=4, rows=3)
    monkeypatch.setattr(pygame.event, "get", lambda: [pygame.event.Event(pygame.QUIT)])
    viz.show_nodes(nodes, 200, 150)
