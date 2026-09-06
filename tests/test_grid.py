import pytest

from neurohacking.grid import GridOfNeurons


@pytest.fixture
def grid():
    return GridOfNeurons(size=3)


def test_origin_exists_and_has_six_neighbours(grid):
    origin = grid.get_origin_neuron()
    assert origin is grid.get_neuron(0, 0)
    assert len(origin.connections) == 6


def test_edge_neuron_has_fewer_neighbours(grid):
    corner = grid.get_neuron(3, 0)
    assert 0 < len(corner.connections) < 6


def test_get_neuron_outside_grid_returns_none(grid):
    assert grid.get_neuron(50, 50) is None


def test_connections_are_shared_by_both_ends(grid):
    for neuron in grid.neurons.values():
        for connection in neuron.connections:
            assert connection in connection.other(neuron).connections


def test_connection_registry_ids_run_from_one_without_gaps(grid):
    assert sorted(grid.connections) == list(range(1, len(grid.connections) + 1))
    for connection_id, connection in grid.connections.items():
        assert connection.id == connection_id
        assert grid.get_connection(connection_id) is connection


@pytest.mark.parametrize("size, expected", [(1, 12), (2, 42), (3, 90)])
def test_connection_count_matches_hexagon_formula(size, expected):
    # A hexagon of radius n has 9n^2 + 3n neighbouring pairs.
    assert len(GridOfNeurons(size=size).connections) == expected


def test_every_connection_joins_adjacent_neurons_exactly_once(grid):
    seen = set()
    for connection in grid.connections.values():
        (q1, r1), (q2, r2) = connection.first.position, connection.second.position
        assert (q2 - q1, r2 - r1) in grid.directions
        pair = frozenset([connection.first, connection.second])
        assert pair not in seen, "pair connected twice"
        seen.add(pair)


def test_connection_between_returns_the_registered_object(grid):
    origin, neighbour = grid.get_neuron(0, 0), grid.get_neuron(1, 0)
    connection = grid.connection_between(origin, neighbour)
    assert connection is grid.get_connection(connection.id)
    assert grid.connection_between(origin, grid.get_neuron(3, 0)) is None


def test_deactivating_origin_connections_isolates_it(grid, capsys):
    for connection in grid.get_origin_neuron().connections:
        connection.is_active = False
    grid.activate_origin()
    assert grid.fired_neurons() == [grid.get_origin_neuron()]


def test_neuron_names_and_positions_match_coordinates(grid):
    neuron = grid.get_neuron(1, -2)
    assert neuron.name == "Neuron_1_-2"
    assert neuron.position == (1, -2)


def test_activate_origin_reaches_every_neuron_exactly_once(grid, capsys):
    grid.activate_origin()
    assert len(grid.fired_neurons()) == len(grid.neurons)
    out = capsys.readouterr().out
    assert out.count("received a signal") == len(grid.neurons)


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
            assert len(neuron.connections) == 6
