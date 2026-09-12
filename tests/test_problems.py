"""--problem: what the network is asked to do, whether anything outside it trains it, and how it is read."""

import json

import pytest

from walnutbutter import constants as C
from walnutbutter.cli import apply_problem, build_parser, cli_main
from walnutbutter.grid import GridOfNeurons
from walnutbutter.learning import Teacher, accuracy
from walnutbutter.monitor import run_epoch
from walnutbutter.neuron import Neuron
from walnutbutter.problems import PROBLEMS


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(Neuron, "verbose", False)


def test_the_problems_and_the_default():
    assert set(PROBLEMS) == {"reversal", "sustain_inputs"}
    assert build_parser().parse_args([]).problem == C.PROBLEM == "reversal"
    assert PROBLEMS["reversal"].trained and not PROBLEMS["sustain_inputs"].trained
    sustain = PROBLEMS["sustain_inputs"]
    assert sustain.across == 8  # 4 bits, complement-coded: the 16 inputs on 8 neurons
    assert (sustain.interval, sustain.readout, sustain.read_window, sustain.target) == (20.0, "input", C.REFRACTORY, "copy")


def test_apply_problem_settles_interval_target_readout_and_training():
    args = build_parser().parse_args(["--problem", "sustain_inputs"])
    apply_problem(args)
    assert args.interval == 20.0 and args.target == "copy" and args.readout == "input" and args.read_window == 5.0
    assert args.learn and args.homeostasis == 0 and args.unstick == 0  # scored, never trained from outside
    args = build_parser().parse_args(["--problem", "sustain_inputs", "--interval", "7"])
    apply_problem(args)
    assert args.interval == 7.0  # an explicit interval wins
    args = build_parser().parse_args([])
    apply_problem(args)
    assert args.interval == C.INTERVAL and args.readout == "top" and args.read_window is None and args.target == "reversed"
    with pytest.raises(ValueError):
        apply_problem(build_parser().parse_args(["--problem", "sustain_inputs", "--rule", "reinforce"]))


def test_the_inputs_are_the_outputs_and_the_read_is_at_the_end_of_the_epoch():
    grid = GridOfNeurons(across=4, rows=3, weight=1.0, omega=0)
    grid.readout, grid.read_window, grid.interval = "input", 5.0, 20.0
    assert grid.output_row() == grid.input_row()
    run_epoch(grid, bits=[True, False], verbose=False)
    fired_at = [n.fired_at for n in grid.input_row()]
    since = grid.horizon - 5.0
    assert grid.output_fired() == [t is not None and t >= since - 1e-9 for t in fired_at]
    assert 0.0 <= accuracy(grid, "copy") <= 1.0
    grid.read_window = None
    assert grid.output_fired() == [n.has_fired for n in grid.input_row()]  # the plain read: fired this epoch


def test_both_engines_read_the_same_outputs():
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from walnutbutter.arrays import ArrayNetwork

    def make():
        grid = GridOfNeurons(across=8, rows=4, weight=None, seed=4)
        grid.readout, grid.read_window, grid.interval = "input", 5.0, 20.0
        return grid

    mesh, net = make(), ArrayNetwork(make())
    assert net.output_index.tolist() == [net.index[n] for n in net.mesh.input_row()]
    a, b = Teacher(mesh, seed=1, target="copy", homeostasis=0, unstick=0), Teacher(net, seed=1, target="copy", homeostasis=0, unstick=0)
    for _ in range(40):
        assert a.epoch(verbose=False) == b.epoch(verbose=False)
        assert mesh.output_fired() == net.output_fired()


def test_sustain_inputs_is_scored_traced_and_checkpointed(tmp_path, capsys):
    save, trace = tmp_path / "s.json", tmp_path / "s.csv"
    assert cli_main(["--headless", "--problem", "sustain_inputs", "--seed", "3", "--epochs", "12", "-r", "4",
                     "--save-weights", str(save)]) == 0
    err = capsys.readouterr().err
    assert "problem sustain_inputs" in err and "Epochs 20 ms apart" in err and "scored on the input row against copy" in err
    assert "scoring copy" in err and "dopamine" in err and f"tracing every epoch to {trace}" in err
    data = json.loads(save.read_text())
    assert data["across"] == 8 and data["epoch"] == 12 and data["problem"] == "sustain_inputs" and data["interval"] == 20.0
    assert data["learning"]["target"] == "copy" and data["learning"]["homeostasis"] == 0 and data["learning"]["rule"] == "dopamine"
    lines = trace.read_text().splitlines()
    assert lines[0] == "epoch,time_ms,dopamine,expected,score" and len(lines) == 13
    epoch, time_ms, dopamine, expected, score = lines[-1].split(",")
    assert epoch == "12" and float(time_ms) == 220.0 and float(dopamine) >= 0 and float(expected) >= 0 and 0 <= float(score) <= 1
    assert cli_main(["--headless", "--load-weights", str(save), "--epochs", "3", "--no-save"]) == 0
    err = capsys.readouterr().err
    assert "problem: sustain_inputs (from the checkpoint)" in err and "scoring copy" in err and "tracing" not in err
    assert cli_main(["--headless", "--problem", "sustain_inputs", "--seeds", "2", "--seed", "1", "--epochs", "3", "--no-save"]) == 0
    assert cli_main(["--headless", "--problem", "sustain_inputs", "--rule", "reinforce"]) == 2
    assert "needs a trained problem" in capsys.readouterr().err


def test_reversal_is_unchanged(tmp_path, capsys):
    save = tmp_path / "r.json"
    assert cli_main(["--headless", "--seed", "3", "--epochs", "3", "-r", "4", "--save-weights", str(save), "--no-trace"]) == 0
    data = json.loads(save.read_text())
    assert data["problem"] == "reversal" and "learning" in data and data["across"] == 8 and data["interval"] == 10.0
    assert not save.with_suffix(".csv").exists()
