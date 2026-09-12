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
    assert sustain.across == 4 and sustain.coding == "raw"  # the 16 four-bit inputs as they are on 4 neurons
    assert (sustain.interval, sustain.readout, sustain.read, sustain.target, sustain.critic) == (20.0, "input", "again", "copy", "row")


def test_apply_problem_settles_interval_target_readout_and_training():
    args = build_parser().parse_args(["--problem", "sustain_inputs"])
    apply_problem(args)
    assert args.interval == 20.0 and args.target == "copy" and args.readout == "input" and args.read == "again"
    assert args.critic == "row" and args.coding == "raw" and args.across == 4
    assert args.learn and args.homeostasis == 0 and args.unstick == 0  # scored, never trained from outside
    args = build_parser().parse_args(["--problem", "sustain_inputs", "--interval", "7"])
    apply_problem(args)
    assert args.interval == 7.0  # an explicit interval wins
    args = build_parser().parse_args([])
    apply_problem(args)
    assert args.interval == C.INTERVAL and args.readout == "top" and args.read == "fired" and args.target == "reversed"
    assert args.coding == "complement"
    with pytest.raises(ValueError):
        apply_problem(build_parser().parse_args(["--problem", "sustain_inputs", "--rule", "reinforce"]))


def test_the_inputs_are_the_outputs_and_on_means_spiked_again():
    grid = GridOfNeurons(across=4, rows=3, weight=1.0, omega=0)
    grid.readout, grid.read, grid.interval = "input", "again", 20.0
    assert grid.output_row() == grid.input_row()
    run_epoch(grid, bits=[True, False], verbose=False)
    fired_at = [n.fired_at for n in grid.input_row()]
    assert grid.output_fired() == [t is not None and t > grid.time for t in fired_at]  # strictly after the forced moment
    forced = [n for n in grid.input_row() if n.forced]
    assert forced and all(n.spikes >= 1 for n in forced)
    for n in forced:  # a forced neuron that only fired the once, at the input's moment, is not on
        if n.fired_at == grid.time:
            assert not grid.output_fired()[grid.input_row().index(n)]
    assert 0.0 <= accuracy(grid, "copy") <= 1.0
    grid.read, grid.read_window = "window", 5.0
    since = grid.horizon - 5.0
    assert grid.output_fired() == [t is not None and t >= since - 1e-9 for t in fired_at]  # the window read, still available
    grid.read = "fired"
    assert grid.output_fired() == [n.has_fired for n in grid.input_row()]  # the plain read: fired this epoch


def test_raw_coding_lays_the_bits_down_as_they_are():
    from walnutbutter.persistence import checkpoint, restore
    grid = GridOfNeurons(across=4, rows=3, weight=1.0, omega=0, seed=2, permute=False)
    grid.coding = "raw"
    assert grid.raw_bit_count() == 4
    grid.set_input_bits([True, False, False, True])
    assert grid.input_pattern == [True, False, False, True] and grid.input_coded == grid.input_bits == [True, False, False, True]
    grid.set_input_bits([False, False, False, False])  # nothing forced is a legal input now
    assert grid.input_neurons() == [] and sum(grid.input_pattern) == 0
    seen = {tuple(grid.new_random_input()) for _ in range(300)}
    assert len(seen) == 16 and all(len(bits) == 4 for bits in seen)  # all sixteen four-bit patterns
    odd = GridOfNeurons(across=3, rows=2, omega=0)
    odd.coding = "raw"
    assert odd.raw_bit_count() == 3  # raw coding has no evenness requirement
    grid.coding = "complement"
    assert grid.raw_bit_count() == 2
    grid.coding = "raw"
    grid.use_ecc(False)
    with pytest.raises(ValueError):
        grid.use_ecc()  # a code needs complement coding
    grid.ecc = None
    import tempfile, os
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "raw.json")
        assert checkpoint(grid, path)["coding"] == "raw"
        assert restore(path)[0].coding == "raw"


def test_both_engines_agree_on_raw_inputs():
    pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from walnutbutter.arrays import ArrayNetwork

    def make():
        grid = GridOfNeurons(across=4, rows=6, weight=None, seed=9)
        grid.coding, grid.readout, grid.read, grid.interval = "raw", "input", "again", 20.0
        return grid

    mesh, net = make(), ArrayNetwork(make())
    assert net.coding == "raw"
    for _ in range(40):
        run_epoch(mesh, verbose=False)
        run_epoch(net, verbose=False)
        assert mesh.input_pattern == net.input_pattern and len(mesh.input_pattern) == 4
        assert mesh.output_fired() == net.output_fired()
        assert accuracy(mesh, "copy") == accuracy(net, "copy")


def test_the_sustained_critic_scores_only_the_forced_neurons():
    from walnutbutter.learning import CRITICS, sustained
    grid = GridOfNeurons(across=4, rows=3, weight=1.0, omega=0)
    grid.readout = "input"
    grid.set_input_bits([True, False])  # pattern 1 0 0 1 after complement coding, before the permutation
    row = grid.input_row()
    forced = [n for n, bit in zip(row, grid.input_pattern) if bit]
    unforced = [n for n, bit in zip(row, grid.input_pattern) if not bit]
    assert len(forced) == len(unforced) == 2
    for n in row:
        n.has_fired = False
    assert sustained(grid) == 0.0
    forced[0].has_fired = True
    assert sustained(grid) == 0.5  # one of the two forced neurons sustained
    unforced[0].has_fired = unforced[1].has_fired = True
    assert sustained(grid) == 0.5  # the unforced neurons are not scored, on or off
    forced[1].has_fired = True
    assert sustained(grid) == 1.0
    assert CRITICS["sustained"] is sustained and "sustained" in CRITICS
    grid.set_input_bits([False, False])  # still two forced: complement coding always forces half the row
    assert sustained(grid) in (0.0, 0.5, 1.0)


def test_both_engines_read_the_same_outputs():
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from walnutbutter.arrays import ArrayNetwork

    def make():
        grid = GridOfNeurons(across=8, rows=4, weight=None, seed=4)
        grid.readout, grid.read, grid.interval = "input", "again", 20.0
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
    assert "-> coded" not in err
    assert "problem sustain_inputs" in err and "Epochs 20 ms apart" in err and "by the row critic, on meaning spiked again after the input" in err
    assert "shows raw bit" in err
    assert "scoring copy" in err and "dopamine" in err and f"tracing every epoch to {trace}" in err
    data = json.loads(save.read_text())
    assert data["across"] == 4 and data["epoch"] == 12 and data["problem"] == "sustain_inputs" and data["interval"] == 20.0
    assert data["coding"] == "raw" and data["learning"]["target"] == "copy" and data["learning"]["critic"] == "row"
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
