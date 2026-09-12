"""--problem: what the network is asked to do, and whether anything outside it trains it."""

import json

from walnutbutter import constants as C
from walnutbutter.cli import build_parser, cli_main
from walnutbutter.problems import PROBLEMS


def test_the_problems_and_the_default():
    assert set(PROBLEMS) == {"reversal", "sustain_inputs"}
    assert build_parser().parse_args([]).problem == C.PROBLEM == "reversal"
    assert PROBLEMS["reversal"].trained and not PROBLEMS["sustain_inputs"].trained
    assert PROBLEMS["sustain_inputs"].across == 8  # 4 bits, complement-coded: the 16 inputs on 8 neurons


def test_sustain_inputs_runs_eight_across_without_a_teacher(tmp_path, capsys):
    save = tmp_path / "s.json"
    assert cli_main(["--headless", "--problem", "sustain_inputs", "--seed", "3", "--epochs", "5", "-r", "4",
                     "--save-weights", str(save)]) == 0
    err = capsys.readouterr().err
    assert "problem sustain_inputs" in err and "runs untrained" in err
    assert "learning" not in err  # no Teacher status lines
    data = json.loads(save.read_text())
    assert data["across"] == 8 and data["epoch"] == 5 and data["problem"] == "sustain_inputs"
    assert "learning" not in data  # nothing external trained it


def test_a_checkpoint_carries_its_problem_and_seeds_need_a_trained_problem(tmp_path, capsys):
    save = tmp_path / "s.json"
    assert cli_main(["--headless", "--problem", "sustain_inputs", "--seed", "3", "--epochs", "2", "-r", "4",
                     "--save-weights", str(save)]) == 0
    capsys.readouterr()
    assert cli_main(["--headless", "--load-weights", str(save), "--epochs", "2", "--no-save"]) == 0
    err = capsys.readouterr().err
    assert "problem: sustain_inputs (from the checkpoint)" in err and "learning" not in err
    assert cli_main(["--headless", "--problem", "sustain_inputs", "--seeds", "2", "--epochs", "3", "--no-save"]) == 2
    assert "no external training yet" in capsys.readouterr().err


def test_reversal_is_unchanged(tmp_path, capsys):
    save = tmp_path / "r.json"
    assert cli_main(["--headless", "--seed", "3", "--epochs", "3", "-r", "4", "--save-weights", str(save)]) == 0
    data = json.loads(save.read_text())
    assert data["problem"] == "reversal" and "learning" in data and data["across"] == 8
