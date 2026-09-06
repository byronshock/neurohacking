import pytest

from error_rate_monitor.grid import GridOfNeurons


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


def test_connections_are_symmetric(grid):
    for neuron in grid.neurons.values():
        for target, _ in neuron.connections:
            assert any(back is neuron for back, _ in target.connections)


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
