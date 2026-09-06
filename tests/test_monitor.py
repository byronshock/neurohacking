import pytest

from neurohacking import main
from neurohacking.cli import cli_main


def test_main_fires_the_whole_grid_and_returns_it(capsys):
    grid = main(grid_size=2)
    assert len(grid.fired_neurons()) == len(grid.neurons)
    assert "Origin neuron Neuron_0_0 has 6 connections" in capsys.readouterr().out


def test_cli_runs_and_returns_zero(capsys):
    assert cli_main([]) == 0
    assert "Neuron_0_0 fired in wave 0." in capsys.readouterr().out


def test_cli_size_option_controls_grid(capsys):
    assert cli_main(["--size", "1"]) == 0
    # size 1 hexagon: origin plus its six neighbours
    assert capsys.readouterr().out.count("fired") == 7


def test_cli_rejects_unknown_arguments():
    with pytest.raises(SystemExit) as exc:
        cli_main(["0.5"])
    assert exc.value.code == 2


def test_cli_weight_and_threshold_options(capsys):
    assert cli_main(["--size", "2", "--weight", "0.5", "--threshold", "1"]) == 0
    assert capsys.readouterr().out.count("fired") == 1  # only the origin


def test_main_passes_weight_and_threshold_through(capsys):
    grid = main(grid_size=1, weight=0.25, threshold=0.25)
    assert len(grid.fired_neurons()) == 7
