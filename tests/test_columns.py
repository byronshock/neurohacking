"""Hexagonal columns in R3: one layer is the grid exactly; more layers wire by horizontal distance."""

import json
import math

import pytest

from walnutbutter.cli import cli_main
from walnutbutter.columns import GUARANTEED, LAYER_SPACING, ROW_SPACING, SPACING, HexColumns
from walnutbutter.grid import GridOfNeurons
from walnutbutter.learning import Teacher, output_row
from walnutbutter.monitor import run_epoch
from walnutbutter.neuron import Neuron
from walnutbutter.persistence import checkpoint, restore


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(Neuron, "verbose", False)


def by_name(net):
    """(source name, target name) -> connection, with grid neurons renamed to the columns' scheme."""
    return {(c.source.name, c.target.name): c for c in net.connections.values()}


def test_one_layer_is_the_hex_grid_exactly():
    grid = GridOfNeurons(seed=1, weight=None)
    stack = HexColumns(seed=1)
    assert len(stack.connections) == len(grid.connections) == 1395
    # same neurons in the same order, so connection ids, kinds, weights, shortcuts and the permutation all match
    grid_neurons, stack_neurons = list(grid.all_neurons()), list(stack.all_neurons())
    index = {n: i for i, n in enumerate(grid_neurons)}
    stack_index = {n: i for i, n in enumerate(stack_neurons)}
    for i in range(1, len(grid.connections) + 1):
        g, s = grid.connections[i], stack.connections[i]
        assert (index[g.source], index[g.target]) == (stack_index[s.source], stack_index[s.target])
        assert g.weight == s.weight
        assert (g.kind == "small_world") == (s.kind == "small_world")
    assert stack.permutation == grid.permutation
    assert len(stack.small_world_connections()) == len(grid.small_world_connections()) == 279
    assert [n.name for n in stack.input_row()] == [f"Column_{p}_9_L0" for p in range(8)]
    assert [stack_index[n] for n in stack.output_row()] == [index[n] for n in output_row(grid)]


def test_one_layer_runs_like_the_grid_epoch_for_epoch():
    grid, stack = GridOfNeurons(seed=3, weight=None), HexColumns(seed=3)
    a, b = Teacher(grid, seed=1), Teacher(stack, seed=1)
    for _ in range(30):
        assert a.epoch(verbose=False) == b.epoch(verbose=False)
        assert [n.fired_in_wave for n in grid.all_neurons()] == [n.fired_in_wave for n in stack.all_neurons()]


def test_cells_are_half_a_unit_apart_and_the_guaranteed_radius_is_one_unit():
    stack = HexColumns(seed=1, omega=0)
    centre = stack.get_neuron_at(4, 5)
    distances = sorted(round(HexColumns.distance(centre, c.target), 3) for c in centre.outgoing)
    assert distances == [0.5] * 6 + [round(math.sqrt(3) / 2, 3)] * 6 + [1.0] * 6
    assert SPACING == 0.5 and ROW_SPACING == pytest.approx(math.sqrt(3) / 4) and GUARANTEED == 1.0
    assert all(stack.is_guaranteed(centre, c.target) for c in centre.outgoing)
    far = stack.get_neuron_at(0, 0)
    assert not stack.is_guaranteed(centre, far) and centre.connection_to(far) is None
    assert not stack.is_guaranteed(centre, centre)  # the same position never connects


def test_more_layers_wire_by_horizontal_distance_at_any_height():
    stack = HexColumns(across=6, rows=5, layers=3, seed=2, omega=0)
    assert len(stack) == 90 and stack.layers == 3
    middle = stack.get_neuron_at(3, 2, layer=1)
    kinds = {}
    for c in middle.outgoing:
        kinds[c.kind] = kinds.get(c.kind, 0) + 1
    assert len(middle.outgoing) == 19 * 3 - 1  # its column and the eighteen around it, in every layer, minus itself
    assert kinds == {"local": 6, "local2": 12, "column": 2, "local-up": 12, "local2-up": 24}
    above, below = stack.get_neuron_at(3, 2, layer=2), stack.get_neuron_at(3, 2, layer=0)
    assert HexColumns.horizontal_distance(middle, above) == 0.0 and stack.is_guaranteed(middle, above)
    assert stack.column(3, 2) == [below, middle, above]
    assert above.position[2] == pytest.approx(2 * LAYER_SPACING) and below.position[2] == 0.0
    assert LAYER_SPACING == SPACING / 4  # a column of four layers is one cell tall
    # every connection obeys the rule, both directions
    for c in stack.connections.values():
        assert stack.is_guaranteed(c.source, c.target) and c.target.connection_to(c.source) is not None
    assert all(len(n.outgoing) <= 56 for n in stack.all_neurons())


def test_shortcuts_only_join_pairs_the_rule_leaves_out():
    stack = HexColumns(across=6, rows=5, layers=2, seed=4, omega=0.1)
    shortcuts = stack.small_world_connections()
    assert shortcuts and all(not stack.is_guaranteed(c.source, c.target) for c in shortcuts)
    assert round(len(shortcuts) / len(stack.connections), 1) == 0.1
    assert HexColumns(across=6, rows=5, layers=2, seed=4, omega=0.1).small_world_connections()[0].target.name == shortcuts[0].target.name


def test_the_bottom_layer_is_the_input_and_the_top_layer_the_output():
    stack = HexColumns(across=6, rows=5, layers=3, seed=1, omega=0)
    assert stack.input_row() == stack.layer(0) and stack.output_row() == stack.layer(2)
    assert stack.input_width() == 30 and len(stack.permutation) == 30 and sorted(stack.permutation) == list(range(30))
    assert stack.raw_bit_count() == 15
    bits = stack.new_random_input()
    assert len(bits) == 15 and len(stack.input_pattern) == 30 and sum(stack.input_pattern) == 15
    run_epoch(stack, verbose=False)
    assert all(n.forced for n, bit in zip(stack.layer(0), stack.input_pattern) if bit)
    assert len(output_row(stack)) == 30
    one = HexColumns(seed=1)
    assert one.input_width() == 8 and len(one.input_row()) == 8 and one.input_row()[0].name == "Column_0_9_L0"


def test_a_code_needs_fourteen_input_neurons_however_they_are_arranged():
    stack = HexColumns(across=7, rows=2, layers=2, seed=1)
    stack.use_ecc()
    assert stack.raw_bit_count() == 4 and len(stack.new_random_input()) == 4 and sum(stack.input_pattern) == 7
    with pytest.raises(ValueError):
        HexColumns(across=6, rows=2, layers=2, seed=1).use_ecc()


def test_learning_runs_on_a_stack_and_both_engines_agree():
    np = pytest.importorskip("numpy")
    from walnutbutter.arrays import ArrayNetwork

    a = HexColumns(across=5, rows=4, layers=3, seed=6)
    b = ArrayNetwork(HexColumns(across=5, rows=4, layers=3, seed=6))
    ta, tb = Teacher(a, seed=2, homeostasis=0.01), Teacher(b, seed=2, homeostasis=0.01)
    for _ in range(40):
        assert ta.epoch(verbose=False) == tb.epoch(verbose=False)
        assert [(-1 if n.fired_in_wave is None else n.fired_in_wave) for n in a.all_neurons()] == b.fired_wave.tolist()
    assert np.allclose([a.connections[i].weight for i in range(1, len(a.connections) + 1)], b.weight, atol=1e-12)
    assert 0.0 <= ta.accuracy_to_date <= 1.0


def test_checkpoints_rebuild_a_stack_from_its_seed(tmp_path):
    stack = HexColumns(across=5, rows=4, layers=2, seed=9, omega=0.1)
    teacher = Teacher(stack, seed=1)
    for _ in range(10):
        teacher.epoch(verbose=False)
    path = tmp_path / "stack.json"
    data = checkpoint(stack, path, teacher)
    assert data["container"] == "columns" and data["layers"] == 2 and data["omega"] == 0.1
    restored, record = restore(path)
    assert isinstance(restored, HexColumns) and restored.layers == 2 and restored.epoch == 10
    assert [c.weight for c in restored.connections.values()] == [c.weight for c in stack.connections.values()]
    assert [c.target.name for c in restored.small_world_connections()] == [c.target.name for c in stack.small_world_connections()]
    assert restored.permutation == stack.permutation
    with pytest.raises(ValueError):
        HexColumns(layers=0)
    with pytest.raises(ValueError):
        HexColumns(omega=1.0)


def test_cli_layers_runs_headless_in_batches_and_on_arrays(tmp_path, capsys):
    save = tmp_path / "stack.json"
    assert cli_main(["--layers", "3", "-a", "4", "-r", "3", "--headless", "--epochs", "10", "--save-weights", str(save)]) == 0
    err = capsys.readouterr().err
    assert "HexColumns(4x3x3 layers" in err and "input 12 neurons, output 12" in err and "after 10 epochs" in err
    data = json.loads(save.read_text())
    assert data["container"] == "columns" and data["layers"] == 3 and data["epoch"] == 10
    assert cli_main(["--load-weights", str(save), "--headless", "--epochs", "5", "--no-save"]) == 0
    assert "loaded" in capsys.readouterr().err
    assert cli_main(["--layers", "2", "-a", "4", "-r", "3", "--seeds", "2", "--epochs", "10", "--no-save"]) == 0
    assert "2 layers of hexagonal columns" in capsys.readouterr().err
    pytest.importorskip("numpy")
    assert cli_main(["--layers", "2", "-a", "4", "-r", "3", "--engine", "arrays", "--headless", "--epochs", "10", "--no-save"]) == 0
    assert "engine: arrays" in capsys.readouterr().err
    assert cli_main(["--layers", "0", "--headless", "--no-save"]) == 2


def test_the_visualizer_draws_layers_side_by_side(tmp_path, monkeypatch):
    pytest.importorskip("pygame")
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from walnutbutter import visualizer as viz

    stack = HexColumns(across=4, rows=3, layers=3, seed=1)
    run_epoch(stack, verbose=False)
    path = tmp_path / "stack.png"
    viz.save(stack, str(path), 300, 120)
    assert path.exists() and path.stat().st_size > 0
    assert "fired" in viz.caption(stack)
