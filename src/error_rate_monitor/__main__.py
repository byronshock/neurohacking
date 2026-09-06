"""Allows `python -m error_rate_monitor ...` as an alternative to the installed command."""

from .cli import main

raise SystemExit(main())
