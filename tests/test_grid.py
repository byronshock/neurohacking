import pytest

from neurohacking.grid import DIRECTIONS, GridOfNeurons, axial_to_offset, offset_to_axial


@pytest.fixture
def grid():
    return GridOfNeurons(columns=7, rows=5)


def test_grid_has_one_neuron_per_cell(grid):
    assert len(grid.neurons) == 7 * 5
    assert (grid.columns, grid.rows) == (7, 5)


@pytest.mark.parametrize("column, row", [(0, 0), (3, -2), (-4, 5), (2, 7), (-1, -1)])
def test_offset_and_axial_convert_both_ways(column, row):
    assert axial_to_offset(*offset_to_axial(column, row)) == (column, row)


def test_odd_rows_shift_half_a_cell_right():
    # In pointy-top "odd-r" layout the x position is proportional to q + r/2.
    for row in range(-3, 4):
        q, r = offset_to_axial(0, row)
        assert q + r / 2 == pytest.approx(0.5 if row % 2 else 0.0)


def test_origin_is_the_centre_cell(grid):
    origin = grid.get_origin_neuron()
    assert origin is grid.get_neuron(0, 0)
    assert origin is grid.get_neuron_at(3, 2)
    assert len(origin.outgoing) == 6 and len(origin.incoming) == 6


def test_every_cell_is_reachable_by_column_and_row(grid):
    for row in range(5):
        for column in range(7):
            assert grid.get_neuron_at(column, row) is not None
    assert grid.get_neuron_at(7, 0) is None
    assert grid.get_neuron_at(0, 5) is None
    assert grid.get_neuron(50, 50) is None


def test_interior_cells_have_six_neighbours_and_edges_fewer(grid):
    for row in range(5):
        for column in range(7):
            count = len(grid.get_neuron_at(column, row).outgoing)
            if 0 < column < 6 and 0 < row < 4:
                assert count == 6
            else:
                assert 2 <= count < 6


def test_neuron_names_and_positions_match_coordinates(grid):
    neuron = grid.get_neuron(1, -2)
    assert neuron.name == "Neuron_1_-2"
    assert neuron.position == (1, -2)


def test_single_cell_grid_has_no_connections():
    grid = GridOfNeurons(columns=1, rows=1)
    assert len(grid.neurons) == 1 and grid.connections == {}
    assert grid.get_origin_neuron() is not None


@pytest.mark.parametrize("columns, rows", [(0, 3), (3, 0), (-1, 2)])
def test_grid_rejects_empty_dimensions(columns, rows):
    with pytest.raises(ValueError):
        GridOfNeurons(columns=columns, rows=rows)


# --- connections --------------------------------------------------------------


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


def test_connection_count_equals_total_neighbour_count(grid):
    expected = sum(len(grid.get_neighbors(n)) for n in grid.neurons.values())
    assert len(grid.connections) == expected
    # 7x5 odd-r rectangle: 5 rows x 6 horizontal pairs = 30, plus 4 row
    # boundaries x 13 diagonal pairs = 52; 82 pairs, each connected both ways.
    assert expected == 164


def test_grid_connections_default_to_weight_one(grid):
    assert all(connection.weight == 1.0 for connection in grid.connections.values())


def test_every_connection_joins_adjacent_neurons_exactly_once(grid):
    seen = set()
    for connection in grid.connections.values():
        (q1, r1), (q2, r2) = connection.source.position, connection.target.position
        assert (q2 - q1, r2 - r1) in DIRECTIONS
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


# --- propagation on the grid --------------------------------------------------


def test_activate_origin_reaches_every_neuron_exactly_once(grid, capsys):
    grid.activate_origin()
    assert len(grid.fired_neurons()) == len(grid.neurons)
    assert capsys.readouterr().out.count("fired") == len(grid.neurons)


def test_reset_clears_all_fired_flags(grid, capsys):
    grid.activate_origin()
    grid.reset()
    assert grid.fired_neurons() == []


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


def test_grid_applies_weight_and_threshold_to_everything():
    grid = GridOfNeurons(columns=3, rows=3, weight=0.3, threshold=0.6)
    assert all(c.weight == 0.3 for c in grid.connections.values())
    assert all(n.threshold == 0.6 for n in grid.neurons.values())


def test_weights_below_threshold_stop_the_signal_at_the_origin(capsys):
    grid = GridOfNeurons(columns=7, rows=5, weight=0.5, threshold=1.0)
    grid.activate_origin()
    assert grid.fired_neurons() == [grid.get_origin_neuron()]
    assert grid.get_neuron(1, 0).potential == 0.5  # it heard the origin, but only once


def test_low_threshold_lets_the_signal_cross_the_grid(capsys):
    grid = GridOfNeurons(columns=7, rows=5, weight=0.5, threshold=0.5)
    grid.activate_origin()
    assert len(grid.fired_neurons()) == len(grid.neurons)


# --- random weights -----------------------------------------------------------


def test_randomize_weights_draws_each_connection_between_minus_one_and_one(grid):
    grid.randomize_weights(seed=1)
    weights = [c.weight for c in grid.connections.values()]
    assert all(-1.0 <= w <= 1.0 for w in weights)
    assert len(set(weights)) > len(weights) // 2  # they really are individual draws
    assert min(weights) < 0 < max(weights)


def test_random_weights_differ_per_direction(grid):
    grid.randomize_weights(seed=2)
    origin, right = grid.get_neuron(0, 0), grid.get_neuron(1, 0)
    assert grid.connection_between(origin, right).weight != grid.connection_between(right, origin).weight


def test_same_seed_gives_same_weights_and_different_seeds_differ():
    a = GridOfNeurons(columns=5, rows=5, weight=None, seed=7)
    b = GridOfNeurons(columns=5, rows=5, weight=None, seed=7)
    c = GridOfNeurons(columns=5, rows=5, weight=None, seed=8)
    weights = lambda g: [x.weight for x in g.connections.values()]
    assert weights(a) == weights(b)
    assert weights(a) != weights(c)


def test_weight_none_in_constructor_randomizes():
    grid = GridOfNeurons(columns=5, rows=5, weight=None, seed=3)
    assert grid.weight is None and grid.seed == 3
    assert len({c.weight for c in grid.connections.values()}) > 1


def test_randomize_weights_respects_custom_range(grid):
    grid.randomize_weights(low=0.2, high=0.3, seed=4)
    assert all(0.2 <= c.weight <= 0.3 for c in grid.connections.values())


def test_fixed_weight_is_still_the_library_default(grid):
    assert grid.weight == 1.0
    assert all(c.weight == 1.0 for c in grid.connections.values())


def test_default_threshold_is_a_quarter():
    grid = GridOfNeurons(columns=3, rows=3)
    assert grid.threshold == 0.25
    assert all(n.threshold == 0.25 for n in grid.neurons.values())



# --- omega: small-world shortcuts ---------------------------------------------


def hex_neighbours(a, b):
    (q1, r1), (q2, r2) = a.position, b.position
    return (q2 - q1, r2 - r1) in DIRECTIONS


def test_omega_zero_adds_no_shortcuts(grid):
    assert grid.omega == 0.0
    assert grid.small_world_connections() == []
    assert len(grid.local_connections()) == len(grid.connections)


@pytest.mark.parametrize("omega", [0.1, 0.25, 0.5])
def test_omega_is_the_fraction_of_all_connections(omega):
    grid = GridOfNeurons(columns=10, rows=8, omega=omega, seed=1)
    local = len(grid.local_connections())
    shortcuts = len(grid.small_world_connections())
    assert local == len(GridOfNeurons(columns=10, rows=8).connections)  # the mesh itself is untouched
    assert shortcuts == round(omega * local / (1 - omega))
    assert shortcuts / len(grid.connections) == pytest.approx(omega, abs=0.01)


def test_shortcuts_go_to_non_neighbours_without_duplicates():
    grid = GridOfNeurons(columns=10, rows=8, omega=0.3, seed=2)
    seen = set()
    for c in grid.small_world_connections():
        assert c.kind == "small_world"
        assert c.source is not c.target
        assert not hex_neighbours(c.source, c.target)
        assert (c.source, c.target) not in seen
        seen.add((c.source, c.target))
        assert c in c.source.outgoing and c in c.target.incoming


def test_shortcut_ids_continue_after_local_ones():
    grid = GridOfNeurons(columns=6, rows=6, omega=0.2, seed=3)
    local_ids = [c.id for c in grid.local_connections()]
    shortcut_ids = [c.id for c in grid.small_world_connections()]
    assert shortcut_ids == list(range(max(local_ids) + 1, len(grid.connections) + 1))


def test_same_seed_gives_same_shortcuts_and_weights():
    a = GridOfNeurons(columns=8, rows=6, weight=None, omega=0.2, seed=9)
    b = GridOfNeurons(columns=8, rows=6, weight=None, omega=0.2, seed=9)
    pairs = lambda g: [(c.source.name, c.target.name, c.weight) for c in g.connections.values()]
    assert pairs(a) == pairs(b)
    assert pairs(a) != pairs(GridOfNeurons(columns=8, rows=6, weight=None, omega=0.2, seed=10))


def test_shortcuts_get_random_weights_too():
    grid = GridOfNeurons(columns=8, rows=6, weight=None, omega=0.2, seed=4)
    weights = {c.weight for c in grid.small_world_connections()}
    assert len(weights) > 1 and all(-1 <= w <= 1 for w in weights)


def test_shortcuts_never_lengthen_the_epoch_and_usually_shorten_it(capsys):
    plain = GridOfNeurons(columns=24, rows=20, weight=1.0)
    plain.activate_origin()
    shortcut = GridOfNeurons(columns=24, rows=20, weight=1.0, omega=0.1, seed=5)
    shortcut.activate_origin()
    assert len(shortcut.fired_neurons()) == len(shortcut.neurons)
    assert len(shortcut.waves) < len(plain.waves)


@pytest.mark.parametrize("omega", [-0.1, 1.0, 1.5])
def test_grid_rejects_omega_out_of_range(omega):
    with pytest.raises(ValueError):
        GridOfNeurons(columns=3, rows=3, omega=omega)


@pytest.mark.parametrize("columns, rows", [(1, 1), (2, 2), (3, 3)])
def test_tiny_meshes_with_omega_do_not_hang(columns, rows):
    grid = GridOfNeurons(columns=columns, rows=rows, omega=0.5, seed=6)
    for c in grid.small_world_connections():
        assert not hex_neighbours(c.source, c.target)


def test_repr_marks_small_world_connections():
    grid = GridOfNeurons(columns=6, rows=6, omega=0.2, seed=7)
    assert repr(grid.small_world_connections()[0]).endswith(", small_world)")
    assert repr(grid.local_connections()[0]).endswith(", active)")
