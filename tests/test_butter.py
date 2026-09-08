import math

import pytest

from walnutbutter.butter import ROW_SPACING, UNIT_DENSITY, Disc, Rect, WalnutButter, spacing_for
from walnutbutter.cartesian import CartesianNodes
from walnutbutter.neuron import Neuron
from walnutbutter.propagation import propagate


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(Neuron, "verbose", False)


def test_unit_density_means_unit_spacing():
    assert spacing_for(UNIT_DENSITY) == pytest.approx(1.0)
    assert spacing_for(4 * UNIT_DENSITY) == pytest.approx(0.5)  # four times as dense: half the spacing
    with pytest.raises(ValueError):
        spacing_for(0)


def test_shapes_contain_their_edges_and_validate():
    rect = Rect(-1, -2, 1, 2)
    assert rect.contains(1, 2) and rect.contains(-1, -2) and not rect.contains(1.001, 0)
    disc = Disc(0, 0, 1.5)
    assert disc.contains(1.5, 0) and not disc.contains(1.51, 0)
    with pytest.raises(ValueError):
        Rect(1, 0, 0, 1)
    with pytest.raises(ValueError):
        Disc(0, 0, 0)


def test_the_default_rectangle_reproduces_the_lattice_exactly():
    butter = WalnutButter.rectangle(8, 10)
    lattice = CartesianNodes(columns=8, rows=10)
    got = sorted((round(x, 9), round(y, 9)) for x, y in butter.positions())
    want = sorted((round(x, 9), round(y, 9)) for x, y in lattice.positions())
    assert got == want and len(got) == 80


def test_denser_butter_packs_more_neurons_into_the_same_shape():
    disc = Disc(0, 0, 3)
    thin = WalnutButter().spread(disc, UNIT_DENSITY / 4)
    unit = WalnutButter().spread(disc, UNIT_DENSITY)
    thick = WalnutButter().spread(disc, 4 * UNIT_DENSITY)
    assert len(thin.positions()) < len(unit.positions()) < len(thick.positions())
    area = math.pi * 9
    assert len(unit.positions()) == pytest.approx(area * UNIT_DENSITY, rel=0.15)
    assert len(thick.positions()) == pytest.approx(area * 4 * UNIT_DENSITY, rel=0.1)
    assert all(disc.contains(x, y) for x, y in thick.positions())


def test_smears_add_and_density_at_reports_the_sum():
    butter = WalnutButter().spread(Rect(-4, -4, 4, 4), 1.0).spread(Disc(0, 0, 1), 2.0)
    assert butter.density_at(0, 0) == 3.0 and butter.density_at(3, 3) == 1.0 and butter.density_at(10, 10) == 0.0
    assert len(butter.smears) == 2
    inner = [p for p in butter.positions() if Disc(0, 0, 1).contains(*p)]
    assert len(inner) > len([p for p in WalnutButter().spread(Rect(-4, -4, 4, 4), 1.0).positions() if Disc(0, 0, 1).contains(*p)])


def test_from_butter_places_the_neurons_and_reach_wires_them():
    butter = WalnutButter().spread(Disc(-4, 0, 1.5), UNIT_DENSITY).spread(Disc(4, 0, 1.5), UNIT_DENSITY)  # two islands 8 apart
    nodes = CartesianNodes.from_butter(butter, seed=1)
    assert nodes.layout == "butter" and nodes.butter is butter and len(nodes) == len(butter.positions())
    nodes.connect_within(reach=2.0)
    assert nodes.reach == 2.0
    # each island is wired within itself, and no connection crosses the gap between them
    for c in nodes.connections.values():
        assert CartesianNodes.distance(c.source, c.target) <= 2.0 + 1e-6
        assert (c.source.position[0] < 0) == (c.target.position[0] < 0)
    left = [n for n in nodes if n.position[0] < 0]
    propagate(fire=[left[0]])
    assert all(not n.has_fired for n in nodes if n.position[0] > 0)  # a gap in the spread is a gap in the network


def test_reach_two_at_unit_density_is_exactly_the_two_hex_rings():
    nodes = CartesianNodes(seed=1)
    nodes.connect_within(reach=2.0)
    centre = nodes.node_at(4, 5)
    distances = sorted(round(CartesianNodes.distance(centre, c.target), 3) for c in centre.outgoing)
    assert distances == [1.0] * 6 + [round(math.sqrt(3), 3)] * 6 + [2.0] * 6
    assert all(c.kind == "local" for c in nodes.connections.values())
    pairs = [(c.source, c.target) for c in nodes.connections.values()]
    assert len(pairs) == len(set(pairs))
    assert all((b, a) in set(pairs) for a, b in pairs)  # near is symmetric, so every link has its reverse
    assert CartesianNodes(seed=1).connect_within(reach=1.0) == 410  # the six neighbours only
    with pytest.raises(ValueError):
        CartesianNodes(seed=1).connect_within(reach=-1)


def test_reach_wiring_is_deterministic_and_weights_are_seeded():
    a, b = CartesianNodes(seed=5), CartesianNodes(seed=5)
    a.connect_within(weight=None)
    b.connect_within(weight=None)
    assert [(c.source.name, c.target.name, c.weight) for c in a.connections.values()] == [
        (c.source.name, c.target.name, c.weight) for c in b.connections.values()
    ]
    fixed = CartesianNodes(seed=6)
    fixed.connect_within(weight=1.0)
    assert {c.weight for c in fixed.connections.values()} == {1.0}
