import pytest

from neurohacking.grid import GridOfNeurons


@pytest.fixture
def grid():
    return GridOfNeurons(size=3)


def test_origin_exists_and_has_six_neighbours(grid):
    origin = grid.get_origin_neuron()
    assert origin is grid.get_neuron(0, 0)
    assert len(origin.outgoing) == 6
    assert len(origin.incoming) == 6


def test_edge_neuron_has_fewer_neighbours(grid):
    corner = grid.get_neuron(3, 0)
    assert 0 < len(corner.outgoing) < 6


def test_get_neuron_outside_grid_returns_none(grid):
    assert grid.get_neuron(50, 50) is None


def test_every_neighbour_pair_is_connected_both_ways(grid):
    for neuron in grid.neurons.values():
        for connection in neuron.outgoing:
            assert connection in connection.target.incoming
            assert connection.target.connection_to(neuron) is not None


def test_connection_registry_ids_run_from_one_without_gaps(grid):
    assert sorted(grid.connections) == list(range(1, len(grid.connections) + 1))
    for connection_id, connection in grid.connections.items():
        assert connection.id == connection_id
        assert grid.get_connection(connection_id) is connection


@pytest.mark.parametrize("size, expected", [(1, 24), (2, 84), (3, 180)])
def test_connection_count_matches_hexagon_formula(size, expected):
    # A hexagon of radius n has 9n^2 + 3n neighbouring pairs, each connected both ways.
    assert len(GridOfNeurons(size=size).connections) == expected


def test_grid_connections_default_to_weight_one(grid):
    assert all(connection.weight == 1.0 for connection in grid.connections.values())


def test_every_connection_joins_adjacent_neurons_exactly_once(grid):
    seen = set()
    for connection in grid.connections.values():
        (q1, r1), (q2, r2) = connection.source.position, connection.target.position
        assert (q2 - q1, r2 - r1) in grid.directions
        pair = (connection.source, connection.target)  # ordered: direction matters
        assert pair not in seen, "same direction connected twice"
        seen.add(pair)


def test_connection_between_returns_the_registered_object(grid):
    origin, neighbour = grid.get_neuron(0, 0), grid.get_neuron(1, 0)
    connection = grid.connection_between(origin, neighbour)
    assert connection is grid.get_connection(connection.id)
    assert grid.connection_between(origin, grid.get_neuron(3, 0)) is None
    reverse = grid.connection_between(neighbour, origin)
    assert reverse is not connection and reverse.id != connection.id


def test_deactivating_outgoing_origin_connections_isolates_it(grid, capsys):
    for connection in grid.get_origin_neuron().outgoing:
        connection.is_active = False
    grid.activate_origin()
    assert grid.fired_neurons() == [grid.get_origin_neuron()]


def test_deactivating_incoming_origin_connections_does_not_stop_it_sending(grid, capsys):
    for connection in grid.get_origin_neuron().incoming:
        connection.is_active = False
    grid.activate_origin()
    assert len(grid.fired_neurons()) == len(grid.neurons)


def test_neuron_names_and_positions_match_coordinates(grid):
    neuron = grid.get_neuron(1, -2)
    assert neuron.name == "Neuron_1_-2"
    assert neuron.position == (1, -2)


def test_activate_origin_reaches_every_neuron_exactly_once(grid, capsys):
    grid.activate_origin()
    assert len(grid.fired_neurons()) == len(grid.neurons)
    out = capsys.readouterr().out
    assert out.count("fired") == len(grid.neurons)


def test_reset_clears_all_fired_flags(grid, capsys):
    grid.activate_origin()
    grid.reset()
    assert grid.fired_neurons() == []


def test_default_size_is_five():
    assert GridOfNeurons().size == 5


@pytest.mark.parametrize("size, expected", [(0, 1), (1, 7), (2, 19), (3, 37), (10, 331)])
def test_grid_is_a_hexagon_with_expected_cell_count(size, expected):
    # A hexagon of radius n has 3n^2 + 3n + 1 cells.
    grid = GridOfNeurons(size=size)
    assert len(grid.neurons) == expected
    for q, r in grid.neurons:
        assert max(abs(q), abs(r), abs(q + r)) <= size


def test_every_interior_neuron_has_six_neighbours():
    grid = GridOfNeurons(size=3)
    for (q, r), neuron in grid.neurons.items():
        if max(abs(q), abs(r), abs(q + r)) < 3:
            assert len(neuron.outgoing) == 6


def test_grid_applies_weight_and_threshold_to_everything():
    grid = GridOfNeurons(size=2, weight=0.3, threshold=0.6)
    assert all(c.weight == 0.3 for c in grid.connections.values())
    assert all(n.threshold == 0.6 for n in grid.neurons.values())


def test_weights_below_threshold_stop_the_signal_at_the_origin(capsys):
    grid = GridOfNeurons(size=3, weight=0.5, threshold=1.0)
    grid.activate_origin()
    assert grid.fired_neurons() == [grid.get_origin_neuron()]
    ring_one = grid.get_neuron(1, 0)
    assert ring_one.potential == 0.5  # it heard the origin, but only once


def test_low_threshold_lets_the_signal_cross_the_grid(capsys):
    grid = GridOfNeurons(size=3, weight=0.5, threshold=0.5)
    grid.activate_origin()
    assert len(grid.fired_neurons()) == len(grid.neurons)
