import io

import pytest

from error_rate_monitor import ErrorRateMonitor
from error_rate_monitor.cli import main


def make_monitor(rate: float, **kwargs) -> tuple[ErrorRateMonitor, io.StringIO]:
    """Build a monitor that writes to an in-memory buffer instead of the terminal."""
    buffer = io.StringIO()
    return ErrorRateMonitor(rate, interval=0.001, output=buffer, **kwargs), buffer


def test_formats_as_percentage_with_two_decimals():
    monitor, _ = make_monitor(0.0523)
    assert monitor.format_error_rate() == "5.23%"


def test_rounds_to_two_decimals():
    monitor, _ = make_monitor(0.123456)
    assert monitor.format_error_rate() == "12.35%"


def test_set_error_rate_changes_display():
    monitor, _ = make_monitor(0.1)
    monitor.set_error_rate(0.25)
    assert monitor.error_rate == 0.25
    assert monitor.format_error_rate() == "25.00%"


@pytest.mark.parametrize("bad_rate", [-0.01, 1.01, 5])
def test_rejects_rate_outside_zero_to_one(bad_rate):
    with pytest.raises(ValueError):
        ErrorRateMonitor(bad_rate)


def test_rejects_non_positive_interval():
    with pytest.raises(ValueError):
        ErrorRateMonitor(0.1, interval=0)


def test_run_prints_one_line_per_tick():
    monitor, buffer = make_monitor(0.5)
    monitor.run(max_ticks=3)
    assert buffer.getvalue().splitlines() == ["50.00%", "50.00%", "50.00%"]


def test_cli_rejects_invalid_rate(capsys):
    exit_code = main(["2.0"])
    assert exit_code == 2
    assert "error rate must be between 0 and 1" in capsys.readouterr().err
