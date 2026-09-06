import pytest

from neurohacking import main
from neurohacking.cli import cli_main
from neurohacking.grid import GridOfNeurons
from neurohacking.monitor import run_epoch
from neurohacking.neuron import Neuron


def test_main_fires_the_bottom_row_and_returns_the_grid(capsys):
    grid = main(columns=4, rows=3, weight=1.0, seed=1)
    out = capsys.readouterr().out
    assert len(grid.fired_neurons()) == 12
    assert grid.waves[0].fired == grid.input_neurons()
    assert all(n.position[1] == max(r for _, r in grid.neurons) for n in grid.waves[0].fired)
    assert "input " in out and "-> bottom row " in out


def test_main_complement_codes_the_input():
    grid = main(columns=6, rows=3, weight=1.0, input_bits=[True, False, True])
    assert grid.input_pattern == [True, False, True, False, True, False]
    assert len(grid.waves[0].fired) == 3


def test_run_epoch_requires_even_columns_and_the_right_bit_count(capsys):
    with pytest.raises(ValueError):
        run_epoch(GridOfNeurons(columns=5, rows=3))
    with pytest.raises(ValueError):
        run_epoch(GridOfNeurons(columns=6, rows=3), bits=[True, False])


def test_run_epoch_resets_the_mesh_and_presents_a_new_input(capsys):
    grid = main(columns=8, rows=4, weight=1.0, seed=3)
    first_bits, first_fired = grid.input_bits, [n.fired_in_wave for n in grid.neurons.values()]
    assert grid.epoch == 1
    seen = {tuple(first_bits)}
    for _ in range(6):
        run_epoch(grid)
        seen.add(tuple(grid.input_bits))
        assert all(n.potential >= 0 or n.has_fired for n in grid.neurons.values())
    assert grid.epoch == 7
    assert len(seen) > 1  # the input actually changes between epochs
    # a second grid with the same seed replays the same sequence of inputs
    other = main(columns=8, rows=4, weight=1.0, seed=3)
    for _ in range(6):
        run_epoch(other)
    assert other.input_bits == grid.input_bits


def test_run_epoch_clears_every_neuron_before_firing(capsys):
    grid = main(columns=6, rows=3, weight=1.0, seed=1)
    fired_before = {n.name: n.fired_in_wave for n in grid.neurons.values()}
    run_epoch(grid, bits=[False, True, False])  # a specific, different input
    assert grid.waves[0].fired == grid.input_neurons()
    assert grid.input_pattern == [False, True, False, True, False, True]
    assert any(fired_before[n.name] != n.fired_in_wave for n in grid.neurons.values())
    assert "epoch 2: input 010 -> bottom row 010101" in capsys.readouterr().out


def test_run_epoch_keeps_weights_and_shortcuts(capsys):
    grid = main(columns=8, rows=4, seed=2, omega=0.2)
    weights = [c.weight for c in grid.connections.values()]
    shortcuts = len(grid.small_world_connections())
    run_epoch(grid)
    assert [c.weight for c in grid.connections.values()] == weights
    assert len(grid.small_world_connections()) == shortcuts


def test_main_random_input_is_reproducible_by_seed(capsys):
    a = main(columns=8, rows=4, seed=4)
    b = main(columns=8, rows=4, seed=4)
    assert a.input_pattern == b.input_pattern
    assert [n.has_fired for n in a.neurons.values()] == [n.has_fired for n in b.neurons.values()]
    assert a.input_pattern != main(columns=8, rows=4, seed=5).input_pattern


def test_cli_runs_and_returns_zero(capsys):
    assert cli_main(["--weight", "1"]) == 0
    captured = capsys.readouterr()
    assert captured.out.count("fired in wave 0.") == 12  # half of the 24-column bottom row
    assert "480 of 480 neurons fired" in captured.err


def test_cli_input_option_sets_the_pattern(capsys):
    assert cli_main(["--columns", "6", "--rows", "3", "--weight", "1", "--input", "110"]) == 0
    out = capsys.readouterr().out
    assert "epoch 1: input 110 -> bottom row 110001" in out
    assert out.count("fired in wave 0.") == 3


@pytest.mark.parametrize("bad", [["--input", "10"], ["--input", "1x1"], ["--columns", "5", "--rows", "3"]])
def test_cli_rejects_bad_input(bad, capsys):
    assert cli_main(["--columns", "6", "--rows", "3"] + bad if "--columns" not in bad else bad) == 2
    assert "error:" in capsys.readouterr().err


def test_cli_defaults_to_random_weights_and_reports_the_seed(capsys):
    assert cli_main(["--columns", "6", "--rows", "6"]) == 0
    err = capsys.readouterr().err
    assert "seed " in err and "of 36 neurons fired" in err


def test_cli_seed_makes_runs_repeatable(capsys):
    cli_main(["--columns", "10", "--rows", "8", "--seed", "11"])
    first = capsys.readouterr()
    cli_main(["--columns", "10", "--rows", "8", "--seed", "11"])
    second = capsys.readouterr()
    assert first.out == second.out and first.err == second.err
    assert "seed 11" in first.err


def test_cli_columns_and_rows_options(capsys):
    assert cli_main(["--columns", "4", "--rows", "3", "--weight", "1"]) == 0
    assert capsys.readouterr().out.count("fired") == 12


def test_cli_rejects_unknown_arguments():
    with pytest.raises(SystemExit) as exc:
        cli_main(["0.5"])
    assert exc.value.code == 2


def test_cli_weight_and_threshold_options(capsys):
    args = ["--columns", "6", "--rows", "5", "--weight", "0.2", "--threshold", "1", "--input", "101"]
    assert cli_main(args) == 0
    assert capsys.readouterr().out.count("fired") == 3  # only the input neurons: 0.4 max input < 1


def test_main_passes_weight_and_threshold_through(capsys):
    grid = main(columns=4, rows=3, weight=0.25, threshold=0.25, seed=1)
    assert len(grid.fired_neurons()) == 12


def test_cli_omega_option_reports_shortcuts(capsys):
    assert cli_main(["--columns", "6", "--rows", "6", "--weight", "1", "--omega", "0.2", "--seed", "1"]) == 0
    err = capsys.readouterr().err
    assert "omega 0.2:" in err and "small-world connections" in err and "seed 1" in err


def test_cli_rejects_omega_out_of_range(capsys):
    assert cli_main(["--columns", "4", "--rows", "3", "--omega", "1"]) == 2
    assert "omega" in capsys.readouterr().err


def test_main_passes_omega_through(capsys):
    grid = main(columns=6, rows=6, weight=1.0, omega=0.25, seed=2)
    assert grid.omega == 0.25 and len(grid.small_world_connections()) > 0


def test_run_epoch_verbose_false_prints_nothing_about_the_input(capsys, monkeypatch):
    monkeypatch.setattr(Neuron, "verbose", False)
    grid = main(columns=6, rows=3, weight=1.0, seed=1)
    capsys.readouterr()
    run_epoch(grid, verbose=False)
    assert capsys.readouterr().out == ""


def test_cli_quiet_suppresses_neuron_lines_but_keeps_the_epoch_line(capsys):
    assert cli_main(["--columns", "6", "--rows", "3", "--weight", "1", "--quiet"]) == 0
    out = capsys.readouterr().out
    assert "fired in wave" not in out and "epoch 1: input" in out
    assert Neuron.verbose is True  # restored once the command finishes
    cli_main(["--columns", "6", "--rows", "3", "--weight", "1"])
    assert "fired in wave" in capsys.readouterr().out


def test_cli_learn_runs_epochs_and_reports_accuracy(capsys):
    args = ["--columns", "8", "--rows", "4", "--seed", "1", "--quiet", "--learn", "--target", "all-off", "--epochs", "2000"]
    assert cli_main(args) == 0
    err = capsys.readouterr().err
    assert "learning all-off (perturb, lr 0.03): accuracy" in err
    assert "after 2000 epochs:" in err
    final = float(err.rsplit("accuracy ", 1)[1].split("%")[0])
    assert final > 85


def test_cli_epochs_without_learn_just_runs_them(capsys):
    assert cli_main(["--columns", "8", "--rows", "4", "--seed", "1", "--quiet", "--epochs", "5"]) == 0
    out = capsys.readouterr().out
    assert out.count("epoch ") == 5 and "epoch 5:" in out


def test_cli_rejects_unknown_target():
    with pytest.raises(SystemExit):
        cli_main(["--learn", "--target", "sideways"])


def test_cli_eligibility_and_sigma_options(capsys):
    args = ["--columns", "8", "--rows", "4", "--seed", "1", "-q", "--learn", "--eligibility", "hebb",
            "--sigma", "0.3", "--lr", "0.02", "--epochs", "20"]
    assert cli_main(args) == 0
    assert "learning reversed (hebb, lr 0.02)" in capsys.readouterr().err
