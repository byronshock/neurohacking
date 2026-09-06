import pytest

from error_rate_monitor import main
from error_rate_monitor.cli import cli_main


def test_main_fires_the_whole_grid_and_returns_it(capsys):
    grid = main(grid_size=2)
    assert len(grid.fired_neurons()) == len(grid.neurons)
    assert "Origin neuron Neuron_0_0 has 6 connections" in capsys.readouterr().out


def test_cli_runs_and_returns_zero(capsys):
    assert cli_main([]) == 0
    assert "Neuron_0_0 received a signal." in capsys.readouterr().out


def test_cli_size_option_controls_grid(capsys):
    assert cli_main(["--size", "1"]) == 0
    # size 1 with the current abs(q) + abs(r) <= size shape: origin plus 4 neighbours
    assert capsys.readouterr().out.count("received a signal") == 5


def test_cli_rejects_unknown_arguments():
    with pytest.raises(SystemExit) as exc:
        cli_main(["0.5"])
    assert exc.value.code == 2
