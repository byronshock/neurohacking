"""Dopamine: produced locally by refires, consumed globally; the neurons learn as they fire (AUTHORITY.md §6)."""

import json
import math
import random

import pytest

from walnutbutter.cli import cli_main
from walnutbutter.dopamine import ORDERS, Dopamine, learn
from walnutbutter.grid import GridOfNeurons
from walnutbutter.learning import Teacher
from walnutbutter.monitor import run_epoch
from walnutbutter.neuron import Neuron
from walnutbutter.propagation import Schedule


@pytest.fixture(autouse=True)
def pinned(monkeypatch):
    monkeypatch.setattr(Neuron, "verbose", False)
    monkeypatch.setattr(Neuron, "refractory", 5.0)
    monkeypatch.setattr(Neuron, "refractory_hops", 3.0)


def test_release_is_the_gamma_density_of_the_refire_delay():
    d = Dopamine(release_alpha=2.0, release_theta=1.0)
    assert d.release_amount(0.0) == 0.0  # an instant refire releases nothing under alpha 2
    assert d.release_amount(1.0) == pytest.approx(math.exp(-1))  # the peak, (alpha - 1) * theta past the refractory period
    assert d.release_amount(2.0) == pytest.approx(2.0 * math.exp(-2))
    assert d.release_amount(-0.5) == 0.0  # a rounding hair early counts as instant
    scaled = Dopamine(release_alpha=2.0, release_theta=5.0)
    assert scaled.release_amount(5.0) == pytest.approx(5.0 * math.exp(-1) / 25.0)
    assert Dopamine(release_alpha=1.0, release_theta=5.0).release_amount(0.0) == pytest.approx(1 / 5.0)  # alpha 1: the old exponential, over theta
    assert d.delay_of(previous=0.0, now=5.0) == 0.0 and d.delay_of(0.0, 7.5) == 2.5
    with pytest.raises(ValueError):
        Dopamine(tau=0)
    with pytest.raises(ValueError):
        Dopamine(release_theta=0)
    with pytest.raises(ValueError):
        Dopamine(order="sideways")
    with pytest.raises(ValueError):
        Dopamine(lr=-1)


def test_the_pool_decays_lazily_and_the_expectation_is_an_exponential_window_from_zero():
    d = Dopamine(tau=20.0, expectation_tau=100.0)
    assert d.peek(5.0) == 0.0 and d.expected() == 0.0 and d.advantage(5.0) == 0.0 and d.updated == 0.0
    seen = []
    d.step(10.0, [1.0, 0.5], update=seen.append)
    assert d.level == 1.5 and d.total == 1.5 and d.releases == 2 and d.updates == 2 and d.updated == 10.0
    assert seen == [1.5]  # the advantage was read against the expectation before it moved, still 0
    assert d.expected() == pytest.approx((1 - math.exp(-10.0 / 100.0)) * 1.5)  # then it caught up a little toward 1.5
    assert d.peek(30.0) == pytest.approx(1.5 * math.exp(-1)) and d.updated == 10.0  # a read that changes nothing
    assert d.advantage(30.0) == pytest.approx(1.5 * math.exp(-1) - d.expected()) and d.updated == 10.0
    d.step(30.0, [], update=lambda a: None)  # an empty wave still moves the expectation toward the (decayed) value
    assert d.updated == 30.0 and d.level == pytest.approx(1.5 * math.exp(-1))
    assert d.expected() == pytest.approx((1 - math.exp(-0.1)) * 1.5 + (1 - math.exp(-0.2)) * (1.5 * math.exp(-1) - (1 - math.exp(-0.1)) * 1.5))
    assert "dopamine" in d.status() and "2 releases" in d.status()
    assert Dopamine.from_state(d.state()).state() == d.state()
    with pytest.raises(ValueError):
        Dopamine(expectation_tau=0)


def test_release_first_and_update_first_differ_in_what_the_update_sees():
    seen = {}
    first = Dopamine(tau=20.0, order="release-first")
    first.step(10.0, [1.0], update=lambda a: seen.__setitem__("first", a))
    second = Dopamine(tau=20.0, order="update-first")
    second.step(10.0, [1.0], update=lambda a: seen.__setitem__("second", a))
    assert seen["first"] == 1.0  # its own release is in the value it consumed (the expectation starts at 0)
    assert seen["second"] == 0.0  # nothing in the pool yet
    assert first.level == second.level == 1.0


def test_a_loop_that_refires_releases_and_moves_its_gated_incoming_weights():
    """a -> b -> c -> a: a refires at 5 ms the instant it recovers. Its synapse from c carried the signal; the one from x did not."""
    a, b, c, x = (Neuron(name, threshold=0.5) for name in "abcx")
    a.connect(b, 1)
    b.connect(c, 2)
    c_a = c.connect(a, 3)
    x_a = x.connect(a, 4)
    dopamine = Dopamine(tau=20.0, release_alpha=2.0, release_theta=1.0, lr=0.1)
    schedule = Schedule()
    schedule.stimulus(a, 0.0)
    waves = schedule.run(until=6.0, on_wave=lambda w: learn(dopamine, w, (-2.0, 2.0)))
    assert [w.time for w in waves] == [0.0, pytest.approx(5 / 3), pytest.approx(10 / 3), pytest.approx(5.0)]
    assert a.spikes == 2 and a.previous_fired_at == 0.0 and a.fired_at == pytest.approx(5.0)
    assert dopamine.releases == 1 and dopamine.total == 0.0 and dopamine.updates == 1  # a's refire at delay 0: eligible, releases nothing
    assert c_a.weight == 1.0 and x_a.weight == 1.0  # the eligibility (its release) was 0, so nothing moved
    assert c_a.last_signal == pytest.approx(5.0) and x_a.last_signal is None  # the gate is set up all the same
    assert b.connection_to(c).weight == 1.0  # b and c fired once each: no refire, nothing released, nothing moved
    # a refire one millisecond late is the peak: previous spike at 0, refire at 6 ms
    from walnutbutter.propagation import Wave
    late = Neuron("late", threshold=0.5)
    feed = Neuron("feed").connect(late, 5)
    late.previous_fired_at, late.fired_at, feed.last_signal = 0.0, 6.0, 6.0
    expected_before = dopamine.expected()  # the advantage is read before the expectation moves
    advantage = learn(dopamine, Wave(0, 6.0, fired=[late]), (-2.0, 2.0))
    assert advantage == pytest.approx(math.exp(-1) - expected_before) and dopamine.total == pytest.approx(math.exp(-1))
    assert dopamine.expected() > expected_before  # and then it moved a little toward what it saw
    assert feed.weight == pytest.approx(1.0 + 0.1 * advantage * math.exp(-1))  # lr * advantage * release, gated


@pytest.mark.parametrize("order", ORDERS)
def test_both_engines_learn_identically_by_dopamine(order):
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    from walnutbutter.arrays import ArrayNetwork

    mesh = GridOfNeurons(weight=None, seed=5, across=6, rows=4)
    mesh.dopamine = Dopamine(order=order)
    twin = GridOfNeurons(weight=None, seed=5, across=6, rows=4)
    twin.dopamine = Dopamine(order=order)
    net = ArrayNetwork(twin)
    ra, rb = random.Random(3), random.Random(3)
    for _ in range(100):
        run_epoch(mesh, verbose=False, noise=0.1, rng=ra)
        run_epoch(net, verbose=False, noise=0.1, rng=rb)
        assert [(-1 if n.fired_in_wave is None else n.fired_in_wave) for n in mesh.all_neurons()] == net.fired_wave.tolist()
        assert mesh.dopamine.state() == net.dopamine.state()
        assert np.allclose([c.weight for c in mesh.connections.values()], net.weight, atol=1e-12)
    assert mesh.dopamine.releases > 0 and mesh.dopamine.updates > 0 and mesh.total_spikes() == net.total_spikes()
    assert [(-np.inf if c.last_signal is None else c.last_signal) for c in mesh.connections.values()] == net.last_signal.tolist()
    net.sync_to_mesh()
    assert sorted((t, c.id) for t, c in twin.schedule.pending()) == sorted((t, c.id) for t, c in mesh.schedule.pending())


def test_the_weights_move_on_the_default_grid_and_stay_in_range():
    grid = GridOfNeurons(weight=None, seed=1)  # the proven 8x10 topology
    grid.dopamine = Dopamine()
    before = [c.weight for c in grid.connections.values()]
    rng = random.Random(1)
    for _ in range(50):
        run_epoch(grid, verbose=False, noise=0.1, rng=rng)
    after = [c.weight for c in grid.connections.values()]
    assert grid.dopamine.releases > 0 and after != before
    assert all(-1.0 <= w <= 1.0 for w in after)


def test_the_teacher_scores_under_dopamine_and_reinforces_under_the_other_rule():
    grid = GridOfNeurons(across=8, rows=4, weight=None, seed=2, omega=0)
    teacher = Teacher(grid, seed=2)
    assert teacher.rule == "dopamine" and grid.dopamine is not None and grid.dopamine.lr == teacher.lr
    teacher.epoch(verbose=False)
    assert "dopamine" in teacher.status() and "perturb, lr" not in teacher.status()
    other = GridOfNeurons(across=8, rows=4, weight=None, seed=2, omega=0)
    old = Teacher(other, seed=2, rule="reinforce")
    old.epoch(verbose=False)
    assert old.rule == "reinforce" and other.dopamine is None and "perturb, lr" in old.status()
    with pytest.raises(ValueError):
        Teacher(grid, rule="osmosis")


def test_sustain_inputs_runs_by_dopamine_and_checkpoints_it(tmp_path, capsys):
    save = tmp_path / "s.json"
    assert cli_main(["--headless", "--problem", "sustain_inputs", "--seed", "3", "--epochs", "30", "-r", "4",
                     "--save-weights", str(save)]) == 0
    err = capsys.readouterr().err
    assert "rule: dopamine, release-first" in err and "release gamma(alpha 2, theta 1 ms)" in err and "after 30 epochs" in err
    data = json.loads(save.read_text())
    assert data["dopamine"]["releases"] > 0 and data["problem"] == "sustain_inputs" and data["learning"]["rule"] == "dopamine"
    assert cli_main(["--headless", "--load-weights", str(save), "--epochs", "5", "--no-save"]) == 0
    err = capsys.readouterr().err
    assert "from the checkpoint" in err and "rule: dopamine" in err
    assert cli_main(["--headless", "-a", "8", "-r", "4", "--seed", "1", "--epochs", "5", "--rule", "reinforce", "--no-save", "-q"]) == 0
    assert "perturb, lr" in capsys.readouterr().err
    assert cli_main(["--headless", "--order", "update-first", "--dopamine-tau", "0"]) == 2
    assert cli_main(["--headless", "--seeds", "2", "--seed", "1", "-a", "8", "-r", "4", "--epochs", "5", "--no-save",
                     "--order", "update-first"]) == 0
