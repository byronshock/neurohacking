import pytest

from neurohacking import main
from neurohacking.cli import cli_main


def test_main_fires_the_whole_grid_and_returns_it(capsys):
    grid = main(columns=3, rows=3, weight=1.0)
    assert len(grid.fired_neurons()) == 9
    assert "Origin neuron Neuron_0_0 has 6 connections" in capsys.readouterr().out


def test_cli_runs_and_returns_zero(capsys):
    assert cli_main(["--weight", "1"]) == 0
    captured = capsys.readouterr()
    assert "Neuron_0_0 fired in wave 0." in captured.out
    assert "480 of 480 neurons fired" in captured.err  # default 24 x 20


def test_cli_defaults_to_random_weights_and_reports_the_seed(capsys):
    assert cli_main(["--columns", "5", "--rows", "5"]) == 0
    err = capsys.readouterr().err
    assert "seed " in err
    assert "of 25 neurons fired" in err


def test_cli_seed_makes_runs_repeatable(capsys):
    cli_main(["--columns", "9", "--rows", "7", "--threshold", "0.3", "--seed", "11"])
    first = capsys.readouterr()
    cli_main(["--columns", "9", "--rows", "7", "--threshold", "0.3", "--seed", "11"])
    second = capsys.readouterr()
    assert first.out == second.out and first.err == second.err
    assert "seed 11" in first.err


def test_cli_fixed_weight_does_not_mention_randomness(capsys):
    cli_main(["--columns", "3", "--rows", "3", "--weight", "0.5"])
    assert "random" not in capsys.readouterr().err


def test_cli_columns_and_rows_options(capsys):
    assert cli_main(["--columns", "4", "--rows", "3", "--weight", "1"]) == 0
    assert capsys.readouterr().out.count("fired") == 12


def test_cli_rejects_unknown_arguments():
    with pytest.raises(SystemExit) as exc:
        cli_main(["0.5"])
    assert exc.value.code == 2


def test_cli_weight_and_threshold_options(capsys):
    assert cli_main(["--columns", "5", "--rows", "5", "--weight", "0.2"]) == 0  # 0.2 < default threshold 0.25
    assert capsys.readouterr().out.count("fired") == 1  # only the origin


def test_main_passes_weight_and_threshold_through(capsys):
    grid = main(columns=3, rows=3, weight=0.25, threshold=0.25)
    assert len(grid.fired_neurons()) == 9


def test_main_random_weights_are_reproducible_by_seed(capsys):
    a = main(columns=5, rows=5, seed=5)
    b = main(columns=5, rows=5, seed=5)
    assert [n.has_fired for n in a.neurons.values()] == [n.has_fired for n in b.neurons.values()]
    assert [c.weight for c in a.connections.values()] == [c.weight for c in b.connections.values()]


def test_cli_omega_option_reports_shortcuts(capsys):
    assert cli_main(["--columns", "6", "--rows", "6", "--weight", "1", "--omega", "0.2", "--seed", "1"]) == 0
    err = capsys.readouterr().err
    assert "omega 0.2:" in err and "small-world connections" in err and "seed 1" in err


def test_cli_rejects_omega_out_of_range(capsys):
    assert cli_main(["--columns", "3", "--rows", "3", "--omega", "1"]) == 2
    assert "omega" in capsys.readouterr().err


def test_main_passes_omega_through(capsys):
    grid = main(columns=6, rows=6, weight=1.0, omega=0.25, seed=2)
    assert grid.omega == 0.25 and len(grid.small_world_connections()) > 0
