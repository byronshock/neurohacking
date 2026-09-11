"""constants.py is the one place a default lives: the constructors, the clock, the Teacher and the CLI all read it."""

import inspect

from walnutbutter import constants as C
from walnutbutter.cartesian import CartesianNodes
from walnutbutter.cli import build_parser
from walnutbutter.columns import HexColumns
from walnutbutter.grid import GridOfNeurons
from walnutbutter.learning import Teacher, reinforce
from walnutbutter.monitor import main
from walnutbutter.neuron import Neuron


def defaults_of(function) -> dict:
    return {name: p.default for name, p in inspect.signature(function).parameters.items() if p.default is not p.empty}


def test_the_command_line_defaults_are_the_constants():
    args = build_parser().parse_args([])
    assert (args.rows, args.omega, args.reach, args.epsilon) == (C.ROWS, C.OMEGA, C.REACH, C.WEIGHT_EPSILON)
    assert (args.threshold, args.minimum_potential) == (C.THRESHOLD, C.MINIMUM_POTENTIAL)
    assert (args.interval, args.tau, args.refractory) == (C.INTERVAL, C.TAU, C.REFRACTORY)
    assert (args.target, args.critic, args.eligibility, args.late) == (C.TARGET, C.CRITIC, C.ELIGIBILITY, C.LATE)
    assert (args.lr, args.sigma, args.homeostasis, args.target_rate) == (C.LR, C.SIGMA, C.HOMEOSTASIS, C.TARGET_RATE)
    assert (args.unstick, args.unstick_target, tuple(args.threshold_range)) == (C.UNSTICK, C.UNSTICK_TARGET, C.THRESHOLD_RANGE)


def test_the_neuron_and_its_clock_read_the_constants():
    assert (Neuron.tau, Neuron.refractory) == (C.TAU, C.REFRACTORY)
    neuron = Neuron()
    assert (neuron.threshold, neuron.minimum_potential) == (C.THRESHOLD, C.MINIMUM_POTENTIAL)
    assert GridOfNeurons(across=2, rows=2, omega=0).interval == C.INTERVAL


def test_every_container_builds_the_default_network_from_the_constants():
    for build in (GridOfNeurons, HexColumns, CartesianNodes, main):
        d = defaults_of(build)
        assert (d["across"], d["rows"], d["threshold"], d["minimum_potential"], d["weight_range"]) == (
            C.ACROSS, C.ROWS, C.THRESHOLD, C.MINIMUM_POTENTIAL, C.WEIGHT_RANGE), build.__name__
    assert defaults_of(GridOfNeurons)["omega"] == defaults_of(HexColumns)["omega"] == C.OMEGA
    assert defaults_of(CartesianNodes.connect_within)["reach"] == C.REACH


def test_the_teacher_and_the_rule_read_the_constants():
    d = defaults_of(Teacher)
    assert (d["target"], d["critic"], d["eligibility"], d["late"]) == (C.TARGET, C.CRITIC, C.ELIGIBILITY, C.LATE)
    assert (d["lr"], d["sigma"], d["baseline_rate"], d["window"]) == (C.LR, C.SIGMA, C.BASELINE_RATE, C.WINDOW)
    assert (d["homeostasis"], d["target_rate"], d["threshold_range"]) == (C.HOMEOSTASIS, C.TARGET_RATE, C.THRESHOLD_RANGE)
    assert (d["unstick"], d["unstick_target"]) == (C.UNSTICK, C.UNSTICK_TARGET)
    r = defaults_of(reinforce)
    assert (r["lr"], r["sigma"], r["eligibility"], r["late"]) == (C.LR, C.SIGMA, C.ELIGIBILITY, C.LATE)
