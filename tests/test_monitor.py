import pytest

from neurohacking import main
from neurohacking.cli import cli_main


def test_main_fires_the_whole_grid_and_returns_it(capsys):
    grid = main(columns=3, rows=3)
    assert len(grid.fired_neurons()) == 9
    assert "Origin neuron Neuron_0_0 has 6 connections" in capsys.readouterr().out


def test_cli_runs_and_returns_zero(capsys):
    assert cli_main([]) == 0
    captured = capsys.readouterr()
    assert "Neuron_0_0 fired in wave 0." in captured.out
    assert "480 of 480 neurons fired" in captured.err  # default 24 x 20


def test_cli_columns_and_rows_options(capsys):
    assert cli_main(["--columns", "4", "--rows", "3"]) == 0
    assert capsys.readouterr().out.count("fired") == 12


def test_cli_rejects_unknown_arguments():
    with pytest.raises(SystemExit) as exc:
        cli_main(["0.5"])
    assert exc.value.code == 2


def test_cli_weight_and_threshold_options(capsys):
    assert cli_main(["--columns", "5", "--rows", "5", "--weight", "0.5", "--threshold", "1"]) == 0
    assert capsys.readouterr().out.count("fired") == 1  # only the origin


def test_main_passes_weight_and_threshold_through(capsys):
    grid = main(columns=3, rows=3, weight=0.25, threshold=0.25)
    assert len(grid.fired_neurons()) == 9
