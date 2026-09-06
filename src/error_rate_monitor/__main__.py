"""Allows `python -m error_rate_monitor ...` as an alternative to the installed command."""

from .cli import cli_main

raise SystemExit(cli_main())
