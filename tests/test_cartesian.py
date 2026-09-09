import math

import pytest

from walnutbutter.cartesian import CartesianNodes, connection_probability, gaussian_density
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
    assert "wired:" in captured.err


def test_cli_bare_nodes_builds_a_learning_lattice(capsys):
    from walnutbutter.cli import cli_main
    assert cli_main(["--headless", "--nodes", "--columns", "6", "--rows", "4", "--seed", "1", "-q", "--epochs", "5", "--no-save"]) == 0
    err = capsys.readouterr().err
    assert "6x4 hexagonal lattice, 24 neurons at unit spacing" in err and "every pair within 2 units" in err
    assert "learning reversed" in err and "after 5 epochs" in err


def test_cli_nodes_rejects_negative(capsys):
    from walnutbutter.cli import cli_main
    assert cli_main(["--headless", "--nodes", "-3"]) == 2
    assert "cannot be negative" in capsys.readouterr().err


def test_cli_nodes_saves_a_picture_and_opens_the_window(tmp_path, monkeypatch, capsys):
    pygame = pytest.importorskip('pygame')
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
    pygame = pytest.importorskip('pygame')
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
    pygame = pytest.importorskip('pygame')
    from walnutbutter import visualizer as viz
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    nodes = CartesianNodes(columns=4, rows=3)
    monkeypatch.setattr(pygame.event, "get", lambda: [pygame.event.Event(pygame.QUIT)])
    viz.show_nodes(nodes, 200, 150)



# --- wiring by distance -------------------------------------------------------


def test_connection_probability_follows_the_standard_gaussian():
    assert gaussian_density(0.0, sigma=1.0) == pytest.approx(1 / (2 * math.pi))
    assert connection_probability(0.0) == 0.0  # the same position: never
    assert connection_probability(1e-6) == pytest.approx(1.0)  # but arbitrarily close: almost certain
    assert connection_probability(1.0, sigma=1.0) == pytest.approx(math.exp(-0.5))  # 0.607
    assert connection_probability(2.0, sigma=1.0) == pytest.approx(math.exp(-2.0))  # 0.135
    assert connection_probability(3.0, sigma=1.0) == pytest.approx(math.exp(-4.5))  # 0.011
    assert connection_probability(1.0, sigma=1.0, scale=0.5) == pytest.approx(0.5 * math.exp(-0.5))
    assert connection_probability(1.5) == pytest.approx(math.exp(-0.5))  # the default sigma is 1.5
    assert connection_probability(0.01, scale=3.0) == 1.0  # never above one
    assert connection_probability(0.0, scale=3.0) == 0.0  # zero distance stays zero whatever the scale
    assert connection_probability(50.0, sigma=1.0) == 0.0  # the density has underflowed: exactly zero, never connected


def test_connection_probability_sigma_sets_the_reach():
    assert connection_probability(2.0, sigma=2.0) == pytest.approx(math.exp(-0.5))  # two units at sigma 2 = one unit at sigma 1
    assert connection_probability(1.0, sigma=0.5) == pytest.approx(math.exp(-2.0))  # a tight Gaussian barely reaches a neighbour
    assert connection_probability(3.0, sigma=3.0) == pytest.approx(math.exp(-0.5))
    assert gaussian_density(0.0, sigma=2.0) == pytest.approx(1 / (8 * math.pi))
    assert connection_probability(0.0, sigma=2.0) == 0.0  # zero distance is still zero
    with pytest.raises(ValueError):
        connection_probability(1.0, sigma=0.0)


def test_connect_by_distance_sigma_changes_the_degree():
    tight = CartesianNodes(seed=4)
    wide = CartesianNodes(seed=4)
    default = CartesianNodes(seed=4)
    n_tight = tight.connect_by_distance(sigma=0.5)
    n_default = default.connect_by_distance()
    n_wide = wide.connect_by_distance(sigma=2.0)
    assert n_tight < n_default < n_wide
    assert wide.receptive_field_sigma == 2.0 and default.receptive_field_sigma == 1.5
    far = [c for c in wide.connections.values() if CartesianNodes.distance(c.source, c.target) > 3.5]
    assert far  # at sigma 2 connections reach several units
    with pytest.raises(ValueError):
        CartesianNodes(seed=4).connect_by_distance(sigma=-1)


def test_neurons_at_the_same_position_never_connect():
    nodes = CartesianNodes(layout="random", count=0, seed=1)
    twins = [nodes.add(0.0, 0.0) for _ in range(6)]  # six neurons on one point
    apart = nodes.add(0.5, 0.0)
    for _ in range(50):
        nodes.connect_by_distance()
    assert all(c.target is apart or c.source is apart for c in nodes.connections.values())
    assert not any(c.source in twins and c.target in twins for c in nodes.connections.values())
    assert any(c.source is apart for c in nodes.connections.values())  # the offset neuron connects freely


def test_connect_by_distance_makes_independent_one_way_connections():
    nodes = CartesianNodes(seed=1)
    made = nodes.connect_by_distance()
    assert made == len(nodes.connections) > 0
    assert sorted(nodes.connections) == list(range(1, made + 1))
    assert all(c.source is not c.target for c in nodes.connections.values())
    assert all(c.weight == 1.0 for c in nodes.connections.values())
    forward = {(c.source, c.target) for c in nodes.connections.values()}
    assert len(forward) == made  # each ordered pair at most once
    one_way = sum(1 for (a, b) in forward if (b, a) not in forward)
    assert one_way > 0  # beyond the neighbours the reverse direction is a separate draw, so some pairs go one way only
    assert all(CartesianNodes.distance(c.source, c.target) > 2 + 1e-6 for c in nodes.connections.values() if c.kind != "local")


def test_connect_by_distance_degree_matches_the_gaussian_on_the_lattice():
    nodes = CartesianNodes(columns=20, rows=20, seed=2)  # a big lattice so the interior dominates
    nodes.connect_by_distance(sigma=1.0, neighbour_radius=1.0)
    interior = [nodes.node_at(c, r) for r in range(5, 15) for c in range(5, 15)]
    degree = sum(len(n.outgoing) for n in interior) / len(interior)
    # expected out-degree: the six guaranteed neighbours plus exp(-d^2/2) over every farther lattice point
    centre = nodes.node_at(10, 10)
    expected = sum(
        1.0 if CartesianNodes.distance(centre, n) <= 1 + 1e-6 else connection_probability(CartesianNodes.distance(centre, n), sigma=1.0)
        for n in nodes if n is not centre
    )
    assert degree == pytest.approx(expected, rel=0.1)
    assert 8 < expected < 10  # six certain neighbours plus a few from the next rings
    near = [c for c in nodes.connections.values() if CartesianNodes.distance(c.source, c.target) < 1.01]
    far = [c for c in nodes.connections.values() if CartesianNodes.distance(c.source, c.target) > 2.9]
    assert len(near) > 10 * len(far)  # near neighbours dominate, distant ones are rare but present


def test_connect_by_distance_is_seeded_and_scalable():
    a, b = CartesianNodes(seed=5), CartesianNodes(seed=5)
    a.connect_by_distance(); b.connect_by_distance()
    assert [(c.source.name, c.target.name) for c in a.connections.values()] == [
        (c.source.name, c.target.name) for c in b.connections.values()
    ]
    sparse = CartesianNodes(seed=5)
    sparse.connect_by_distance(scale=0.25)
    assert len(sparse.connections_of_kind("gaussian")) < len(a.connections_of_kind("gaussian")) / 2
    assert len(sparse.connections_of_kind("local")) == len(a.connections_of_kind("local"))  # neighbours are certain regardless
    none = CartesianNodes(seed=5)
    none.connect_by_distance(scale=0.0)
    assert none.connections_of_kind("gaussian") == [] and len(none.connections) == len(a.connections_of_kind("local"))
    with pytest.raises(ValueError):
        CartesianNodes(seed=5).connect_by_distance(scale=-1)


def test_connect_by_distance_random_weights_and_propagation():
    nodes = CartesianNodes(seed=3)
    nodes.connect_by_distance(weight=None)
    weights = [c.weight for c in nodes.connections.values()]
    assert min(weights) < 0 < max(weights) and all(-1 <= w <= 1 for w in weights)
    fixed = CartesianNodes(seed=3)
    fixed.connect_by_distance(weight=1.0)
    propagate(fire=[fixed.node_at(4, 5)])
    assert len(fixed.fired_neurons()) >= 0.95 * len(fixed)  # weight 1 everywhere lights the lattice, bar the odd neuron with no incoming connection
    assert fixed.mean_out_degree() == pytest.approx(len(fixed.connections) / len(fixed))



def test_cli_nodes_reports_the_reach_wiring(capsys):
    from walnutbutter.cli import cli_main
    base = ["--headless", "--nodes", "--seed", "1", "--columns", "6", "--rows", "4", "-q", "--epochs", "2", "--no-save"]
    assert cli_main(base) == 0
    assert "every pair within 2 units" in capsys.readouterr().err  # the default reach
    assert cli_main(base + ["--reach", "1"]) == 0
    assert "every pair within 1 units" in capsys.readouterr().err
    assert cli_main(base + ["--reach", "-1"]) == 2



# --- the lattice as a network: input row, output row, epochs, learning ---------------


def test_lattice_rows_are_addressed_from_the_top_like_the_grid():
    nodes = CartesianNodes(columns=4, rows=3)
    top = [nodes.get_neuron_at(c, 0) for c in range(4)]
    bottom = nodes.input_row()
    assert all(n.position[1] > 0 for n in top) and all(n.position[1] < 0 for n in bottom)  # top row is up, input row is down
    assert nodes.get_neuron_at(1, 2) is nodes.node_at(1, 0)
    assert nodes.get_neuron_at(4, 0) is None


def test_lattice_runs_epochs_with_complement_coded_permuted_input():
    from walnutbutter.monitor import run_epoch
    nodes = CartesianNodes(seed=3)
    nodes.connect_by_distance(weight=1.0)
    assert sorted(nodes.permutation) == list(range(8)) and nodes.permutation != list(range(8))
    run_epoch(nodes, verbose=False)
    assert nodes.epoch == 1 and sum(nodes.input_pattern) == 4
    assert nodes.waves[0].fired == nodes.input_neurons()
    assert all(n.fired_in_wave == 0 for n in nodes.input_neurons())
    assert len(nodes.fired_neurons()) >= 0.95 * len(nodes)  # weight 1 everywhere lights the lattice, bar any neuron with no incoming connection
    nodes.reset()
    assert nodes.fired_neurons() == []
    plain = CartesianNodes(seed=3, permute=False)
    assert plain.permutation == list(range(8))


def test_random_layout_has_no_rows_to_address():
    nodes = CartesianNodes(layout="random", count=10, seed=1)
    assert nodes.get_neuron_at(0, 0) is None


def test_teacher_learns_on_the_lattice():
    import statistics
    from walnutbutter.learning import Teacher, accuracy
    nodes = CartesianNodes(columns=8, rows=4, seed=1)
    nodes.connect_by_distance(sigma=1.0, weight=None)  # sparse enough that all-off is reachable quickly
    teacher = Teacher(nodes, target="all-off", lr=0.1, seed=1)
    rewards = [teacher.epoch(verbose=False) for _ in range(1500)]
    # the sparse lattice starts high on all-off; learning should still take it to near perfection
    assert statistics.mean(rewards[-200:]) > 0.98 > statistics.mean(rewards[:100])
    assert 0 <= accuracy(nodes) <= 1 and "to date over 1,500 epochs" in teacher.status()



# --- three tiers of wiring ---------------------------------------------------------


def test_neighbours_and_their_neighbours_are_always_connected_both_ways():
    nodes = CartesianNodes(seed=9)
    nodes.connect_by_distance(sigma=0.5)  # a tight Gaussian adds almost nothing beyond the guaranteed cells
    for neuron in nodes:
        for other in nodes.neighbours(neuron, radius=2.0):  # everything within two units: both rings
            assert neuron.connection_to(other) is not None and neuron.connection_to(other).kind == "local"
            assert other.connection_to(neuron) is not None
    local = nodes.connections_of_kind("local")
    assert len(local) == sum(len(nodes.neighbours(n, radius=2.0)) for n in nodes)
    assert all(CartesianNodes.distance(c.source, c.target) <= 2 + 1e-6 for c in local)
    centre = nodes.node_at(4, 5)
    mine = [c for c in centre.outgoing if c.kind == "local"]
    assert len(mine) == 18
    distances = sorted(round(CartesianNodes.distance(centre, c.target), 3) for c in mine)
    assert distances == [1.0] * 6 + [round(math.sqrt(3), 3)] * 6 + [2.0] * 6  # the hex grid's two rings exactly


def test_epsilon_and_neighbour_radius_are_tunable():
    nodes = CartesianNodes(seed=9)
    nodes.connect_by_distance(sigma=0.5, neighbour_radius=1.0)  # only the six immediate neighbours
    centre = nodes.node_at(4, 5)
    assert len([c for c in centre.outgoing if c.kind == "local"]) == 6
    ring = CartesianNodes(seed=9)
    ring.connect_by_distance(sigma=0.5, neighbour_radius=math.sqrt(3), epsilon=0.01)  # the six corners of the second ring too
    assert len([c for c in ring.node_at(4, 5).outgoing if c.kind == "local"]) == 12
    tight = CartesianNodes(seed=9)
    tight.connect_by_distance(sigma=0.5, neighbour_radius=0.0, epsilon=0.0)  # no guaranteed neighbours at all
    assert tight.connections_of_kind("local") == []
    with pytest.raises(ValueError):
        CartesianNodes(seed=9).connect_by_distance(neighbour_radius=-1)


def test_no_small_world_shortcuts_on_the_lattice():
    nodes = CartesianNodes(seed=9)
    nodes.connect_by_distance(sigma=1.0)
    kinds = {c.kind for c in nodes.connections.values()}
    assert kinds <= {"local", "gaussian"}
    assert nodes.connections_of_kind("small_world") == []
    with pytest.raises(TypeError):
        CartesianNodes(seed=9).connect_by_distance(omega=0.2)  # not a lattice parameter



def test_cli_random_scatter_is_shown_not_learned(capsys):
    from walnutbutter.cli import cli_main
    assert cli_main(["--headless", "--nodes", "12", "--seed", "1"]) == 0
    err = capsys.readouterr().err
    assert "12 neurons at random" in err and "every pair within 2 units" in err and "no rows" in err


# --- lattice checkpoints and the reproducible command line ---------------------


def test_lattice_checkpoint_round_trips(tmp_path):
    from walnutbutter.learning import Teacher
    from walnutbutter.persistence import checkpoint, restore
    nodes = CartesianNodes(seed=3)
    nodes.connect_by_distance(sigma=1.5, weight=None)
    teacher = Teacher(nodes, seed=3)
    for _ in range(30):
        teacher.epoch(verbose=False)
    path = tmp_path / "lattice.json"
    data = checkpoint(nodes, path, teacher)
    assert data["container"] == "lattice" and data["receptive_field_sigma"] == 1.5
    restored, _ = restore(path)
    assert isinstance(restored, CartesianNodes) and restored.epoch == 30
    assert restored.positions() == nodes.positions() and restored.permutation == nodes.permutation
    assert [(c.source.name, c.target.name, c.kind, c.weight) for c in restored.connections.values()] == [
        (c.source.name, c.target.name, c.kind, c.weight) for c in nodes.connections.values()
    ]
    assert [n.threshold for n in restored] == [n.threshold for n in nodes]
    from walnutbutter.monitor import run_epoch
    run_epoch(nodes, bits=[True, False, True, True], verbose=False)
    run_epoch(restored, bits=[True, False, True, True], verbose=False)
    assert [n.has_fired for n in nodes] == [n.has_fired for n in restored]


def test_cli_lattice_run_saves_a_loadable_checkpoint(tmp_path, capsys):
    from pathlib import Path
    from walnutbutter.cli import cli_main
    from walnutbutter.persistence import restore
    assert cli_main(["--headless", "--nodes", "--seed", "3", "--reach", "2", "-q", "--epochs", "20"]) == 0
    files = list(Path("runs").glob("*-seed3.json"))
    assert len(files) == 1
    restored, data = restore(files[0])
    assert isinstance(restored, CartesianNodes) and data["epoch"] == 20 and data["reach"] == 2.0 and restored.reach == 2.0
    assert cli_main(["--headless", "-q", "--epochs", "5", "--load-weights", str(files[0]), "--no-save"]) == 0
    assert "to date over 25 epochs" in capsys.readouterr().err


def test_command_line_reproduces_the_sweep_sequence_exactly(tmp_path, capsys):
    """The sweep built the network and ran every epoch through Teacher.epoch; the command line must match bit for bit."""
    from pathlib import Path
    from walnutbutter.cli import cli_main
    from walnutbutter.learning import Teacher
    from walnutbutter.persistence import read_checkpoint
    epochs = 300
    nodes = CartesianNodes(columns=8, rows=10, seed=3)
    nodes.connect_within(reach=2.0, weight=None)
    teacher = Teacher(nodes, seed=3)
    for _ in range(epochs):
        teacher.epoch(verbose=False)
    assert cli_main(["--headless", "--nodes", "--seed", "3", "--reach", "2", "-q", "--epochs", str(epochs)]) == 0
    saved = read_checkpoint(next(Path("runs").glob("*-seed3.json")))
    assert saved["weights"] == [c.weight for c in nodes.connections.values()]
    assert saved["thresholds"] == [n.threshold for n in nodes]
    assert saved["learning"]["total_reward"] == teacher.total_reward
