from error_rate_monitor import ErrorRateMonitor
from error_rate_monitor.cli import main


def test_run_fires_the_whole_grid(capsys):
    monitor = ErrorRateMonitor(grid_size=2)
    monitor.run()
    grid = monitor.grid
    assert len(grid.fired_neurons()) == len(grid.neurons)
    assert "Origin neuron Neuron_0_0 has 6 connections" in capsys.readouterr().out


def test_cli_runs_and_returns_zero(capsys):
    assert main([]) == 0
    assert "Neuron_0_0 received a signal." in capsys.readouterr().out


def test_cli_rejects_unknown_arguments(capsys):
    try:
        main(["0.5"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected argparse to exit")
