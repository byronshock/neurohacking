# error-rate-monitor

Continuously prints an error rate to the terminal, as a percentage with two
decimals, until you press Ctrl+C.

## Setup (once)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
error-rate-monitor 0.05            # prints "5.00%" once per second
error-rate-monitor 0.05 -i 0.5     # twice per second
```

The rate is given as a fraction between 0 and 1. Stop with Ctrl+C.

## From Python

```python
from error_rate_monitor import ErrorRateMonitor

monitor = ErrorRateMonitor(0.05)
monitor.set_error_rate(0.12)   # change what is displayed
monitor.run()                  # loops until Ctrl+C
```

## Tests

```bash
pytest
```

## Layout

```
src/error_rate_monitor/
  monitor.py   ErrorRateMonitor class (the logic)
  cli.py       argument parsing and the `error-rate-monitor` command
  __main__.py  lets you run `python -m error_rate_monitor`
tests/         pytest tests
pyproject.toml project metadata, dependencies, and the command definition
```
