"""The clock's tolerance: two moments closer than this are the same moment (AUTHORITY.md §4.1).

Times are nominal milliseconds in floating point and a hop is REFRACTORY /
REFRACTORY_HOPS, rarely representable exactly, so a chain of hops that
should land on an input's time lands a few ulps off it, and a spike that
should return the instant a neuron's refractory period ends may compute to
a hair before it. Every comparison of two moments therefore allows a slack
proportional to the clock, and an input's exact time anchors any wave it
joins, so that chains never drift off the input clock.
"""

TOLERANCE = 1e-9  # relative, per millisecond of clock time (at least 1 ms)


def slack(time: float) -> float:
    return TOLERANCE * max(1.0, abs(time))


def before(time: float, until: float) -> bool:
    """True if `time` is before `until` by more than the slack: the schedule runs events that are `before` its horizon."""
    return time < until - slack(until)
