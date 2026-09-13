"""improved_sustain: a 10x7 hex grid wired to reach 3, the four inputs in the middle of the middle row."""

import json

import pytest

from walnutbutter.cli import apply_problem, build_parser, cli_main
from walnutbutter.grid import GridOfNeurons, hex_distance
from walnutbutter.monitor import run_epoch
from walnutbutter.neuron import Neuron
from walnutbutter.persistence import restore
from walnutbutter.problems import PROBLEMS


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(Neuron, "verbose", False)


def test_reach_three_wires_the_third_ring_and_leaves_reach_two_untouched():
    two = GridOfNeurons(across=10, rows=7, omega=0)
    three = GridOfNeurons(across=10, rows=7, omega=0, reach=3)
    assert two.reach == 2 and three.reach == 3
    assert len(three.get_origin_neuron().outgoing) == 36 and len(two.get_origin_neuron().outgoing) == 18  # 6 + 12 + 18 for an interior cell
    for connection in three.connections.values():
        d = hex_distance(connection.source.position, connection.target.position)
        assert 1 <= d <= 3 and connection.kind == {1: "local", 2: "local2", 3: "local3"}[d]
    assert three.local_connections() and all(c.kind.startswith("local") for c in three.local_connections())
    for i in range(1, len(two.connections) + 1):  # the first two rings keep their ids: reach only appends
        a, b = two.connections[i], three.connections[i]
        assert (a.source.name, a.target.name, a.kind) == (b.source.name, b.target.name, b.kind)
    with pytest.raises(ValueError):
        GridOfNeurons(across=4, rows=3, reach=2.5)
    shortcuts = GridOfNeurons(across=10, rows=7, omega=0.1, seed=3, reach=3)
    for c in shortcuts.small_world_connections():
        assert hex_distance(c.source.position, c.target.position) > 3  # shortcuts only join pairs the reach leaves out


def test_an_input_zone_replaces_the_bottom_row():
    grid = GridOfNeurons(across=10, rows=7, omega=0, reach=3)
    grid.coding = "raw"
    grid.set_input_cells([(3, 3), (4, 3), (5, 3), (6, 3)])
    row = grid.input_row()
    assert [n.name for n in row] == [grid.get_neuron_at(p, 3).name for p in (3, 4, 5, 6)]
    assert grid.input_width() == 4 and grid.raw_bit_count() == 4 and grid.permutation == [0, 1, 2, 3]
    grid.readout = "input"
    assert grid.output_row() == row
    grid.set_input_bits([True, False, False, True])
    assert grid.input_neurons() == [row[0], row[3]]
    with pytest.raises(ValueError):
        grid.set_input_cells([(20, 3)])


def test_the_problem_and_its_settings():
    p = PROBLEMS["improved_sustain"]
    assert (p.across, p.rows, p.reach, p.input_cells, p.permute, p.coding) == (10, 7, 3, ((3, 3), (4, 3), (5, 3), (6, 3)), False, "raw")
    assert (p.readout, p.read, p.target, p.critic, p.interval) == ("input", "again", "copy", "row", 20.0)
    args = build_parser().parse_args(["--problem", "improved_sustain"])
    apply_problem(args)
    assert args.across == 10 and args.rows == 7 and args.grid_reach == 3 and args.no_permute and args.input_cells == p.input_cells


def test_improved_sustain_runs_checkpoints_and_restores(tmp_path, capsys):
    save = tmp_path / "i.json"
    assert cli_main(["--headless", "--problem", "improved_sustain", "--seed", "2", "--epochs", "12", "--save-weights", str(save), "--no-trace"]) == 0
    err = capsys.readouterr().err
    assert "problem improved_sustain" in err
    data = json.loads(save.read_text())
    assert data["across"] == 10 and data["rows"] == 7 and data["grid_reach"] == 3 and data["input_cells"] == [[3, 3], [4, 3], [5, 3], [6, 3]]
    assert data["permutation"] == [0, 1, 2, 3] and data["coding"] == "raw"
    restored, _ = restore(save)
    assert restored.reach == 3 and restored.input_cells == [(3, 3), (4, 3), (5, 3), (6, 3)] and len(restored.input_row()) == 4
    assert [c.weight for c in restored.connections.values()] == data["weights"]
    assert cli_main(["--headless", "--load-weights", str(save), "--epochs", "3", "--no-save"]) == 0  # the problem comes back with its rule
    assert "problem: improved_sustain (from the checkpoint)" in capsys.readouterr().err


def test_both_engines_agree_on_improved_sustain():
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from walnutbutter.arrays import ArrayNetwork
    from walnutbutter.dopamine import Dopamine

    def make():
        grid = GridOfNeurons(across=10, rows=7, weight=None, seed=4, reach=3)
        grid.coding, grid.readout, grid.read, grid.interval = "raw", "input", "again", 20.0
        grid.set_input_cells([(3, 3), (4, 3), (5, 3), (6, 3)])
        grid.dopamine = Dopamine()
        return grid

    mesh, net = make(), ArrayNetwork(make())
    assert net.input_index.tolist() == [net.index[n] for n in net.mesh.input_row()] and len(net.input_index) == 4
    for _ in range(30):
        run_epoch(mesh, verbose=False)
        run_epoch(net, verbose=False)
        assert mesh.input_pattern == net.input_pattern and mesh.output_fired() == net.output_fired()
        assert [n.spikes for n in mesh.all_neurons()] == net.spikes.tolist()
        assert np.allclose([c.weight for c in mesh.connections.values()], net.weight, atol=1e-12)
