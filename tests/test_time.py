"""The clock: nominal milliseconds, a lazy leak, an absolute refractory period, and inputs that arrive at a time."""

import json
import math

import pytest

from walnutbutter.cli import cli_main
from walnutbutter.columns import HexColumns
from walnutbutter.grid import GridOfNeurons
from walnutbutter.learning import Teacher
from walnutbutter.monitor import run_epoch
from walnutbutter.neuron import Neuron
from walnutbutter.persistence import checkpoint, restore
from walnutbutter.propagation import propagate


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(Neuron, "verbose", False)


@pytest.fixture
def clock(monkeypatch):
    """The defaults, pinned: tau 2 ms, refractory 5 ms."""
    monkeypatch.setattr(Neuron, "tau", 2.0)
    monkeypatch.setattr(Neuron, "refractory", 5.0)


def test_the_defaults_are_five_milliseconds_each_and_inputs_ten_apart(clock):
    assert Neuron.tau == 2.0 and Neuron.refractory == 5.0
    grid = GridOfNeurons(across=4, rows=3, omega=0)
    assert grid.interval == 10.0 and grid.time == 0.0 and grid.next_time() == 0.0
    run_epoch(grid, verbose=False)
    assert grid.time == 0.0 and grid.next_time() == 10.0
    run_epoch(grid, verbose=False)
    assert grid.time == 10.0
    run_epoch(grid, verbose=False, time=12.5)
    assert grid.time == 12.5 and grid.next_time() == 22.5
    with pytest.raises(ValueError):
        run_epoch(grid, verbose=False, time=3.0)  # the clock does not run backwards


def test_the_leak_is_lazy_and_exponential(clock):
    a = Neuron("a")
    a.receive(0.2, now=0.0)
    assert a.potential == pytest.approx(0.2) and a.last_update == 0.0
    a.receive(0.0, now=2.0)  # one tau later
    assert a.potential == pytest.approx(0.2 * math.exp(-1)) and a.last_update == 2.0
    quiet = Neuron("q")
    quiet.receive(0.2, now=0.0)
    assert quiet.potential == pytest.approx(0.2)  # nothing happens to a neuron nobody talks to
    quiet.leak(20.0)
    assert quiet.potential == pytest.approx(0.2 * math.exp(-10)) and quiet.last_update == 20.0


def test_no_leak_when_tau_is_infinite(monkeypatch):
    monkeypatch.setattr(Neuron, "tau", math.inf)
    a = Neuron("a")
    a.receive(0.2, now=0.0)
    a.receive(0.0, now=1000.0)
    assert a.potential == pytest.approx(0.2) and a.last_update == 1000.0


def test_a_neuron_ignores_signals_and_stimulus_during_its_refractory_period(clock):
    a = Neuron("a", threshold=0.1)
    a.receive(1.0, now=0.0)
    assert a.can_fire(0.0)
    a.fire(wave=1, now=0.0)
    assert a.fired_at == 0.0 and a.potential == 0.0  # the spike resets the potential
    a.reset()
    assert a.refractory_at(0.0) and a.refractory_at(4.999) and not a.refractory_at(5.0)
    a.receive(1.0, now=3.0)
    assert a.potential == 0.0  # ignored
    a.receive(1.0, now=5.0)
    assert a.potential == pytest.approx(1.0)  # recovered
    # forced stimulus: refractory wins
    b = Neuron("b")
    b.fire(wave=0, now=0.0)
    b.reset()
    waves = propagate(fire=[b], now=2.0)
    assert waves[0].fired == [] and not b.has_fired
    waves = propagate(fire=[b], now=5.0)
    assert waves[0].fired == [b] and b.fired_at == 5.0


def test_inputs_closer_than_the_refractory_period_are_partly_ignored(clock):
    grid = GridOfNeurons(across=6, rows=3, weight=None, seed=1, omega=0)
    grid.interval = 3.0  # inputs every 3 ms: a neuron that fired for one input is still refractory for the next
    run_epoch(grid, bits=[True, True, True], verbose=False)
    first = {n for n in grid.all_neurons() if n.has_fired}
    run_epoch(grid, bits=[True, True, True], verbose=False)
    assert grid.time == 3.0
    second = {n for n in grid.all_neurons() if n.has_fired}
    assert not (first & second)  # nobody who fired at 0 ms could fire again at 3 ms
    run_epoch(grid, bits=[True, True, True], verbose=False, time=10.0)
    third = {n for n in grid.all_neurons() if n.has_fired}
    assert third & first  # by 10 ms the first cascade's neurons have recovered


def test_outputs_carry_the_time_of_their_cascade(clock):
    grid = GridOfNeurons(across=4, rows=2, weight=1.0, omega=0)  # everything fires
    run_epoch(grid, bits=[True, False], verbose=False, time=7.0)
    assert grid.output_times() == [7.0] * 4
    grid.reset()
    assert grid.output_times() == [None] * 4


def test_a_short_tau_is_the_old_discharge_epoch_for_epoch(monkeypatch):
    """Inputs 10 ms apart with a leak that empties a potential in between is the old epoch-by-epoch run."""
    monkeypatch.setattr(Neuron, "refractory", 5.0)
    monkeypatch.setattr(Neuron, "tau", 1e-9)  # exp(-10 / 1e-9) underflows to exactly 0
    leaky = Teacher(GridOfNeurons(weight=None, seed=4), seed=1)
    zeroed = Teacher(GridOfNeurons(weight=None, seed=4), seed=1, discharge=True)
    for _ in range(60):
        assert leaky.epoch(verbose=False) == zeroed.epoch(verbose=False)
        assert [n.fired_in_wave for n in leaky.grid.all_neurons()] == [n.fired_in_wave for n in zeroed.grid.all_neurons()]
    assert [c.weight for c in leaky.grid.connections.values()] == [c.weight for c in zeroed.grid.connections.values()]


def test_with_a_long_tau_memory_carries_between_inputs(monkeypatch):
    monkeypatch.setattr(Neuron, "tau", 5.0)  # 13% of a potential survives a 10 ms gap
    remembering = Teacher(GridOfNeurons(weight=None, seed=4), seed=1)
    forgetting = Teacher(GridOfNeurons(weight=None, seed=4), seed=1, discharge=True)
    same = 0
    for _ in range(40):
        remembering.epoch(verbose=False)
        forgetting.epoch(verbose=False)
        same += [n.fired_in_wave for n in remembering.grid.all_neurons()] == [n.fired_in_wave for n in forgetting.grid.all_neurons()]
    assert same < 40  # and it shows


def test_both_engines_keep_the_same_clock(clock):
    np = pytest.importorskip("numpy")
    from walnutbutter.arrays import ArrayNetwork

    def make():
        grid = GridOfNeurons(across=6, rows=4, weight=None, seed=8)
        grid.interval = 3.0  # refractory periods overlap inputs, so timing matters every cascade
        return grid

    mesh, net = make(), ArrayNetwork(make())
    a, b = Teacher(mesh, seed=2), Teacher(net, seed=2)
    for k in range(60):
        time = None if k % 7 else mesh.time + 12.0  # now and then a long gap: everyone recovers and leaks
        ra = a.epoch(verbose=False) if time is None else (run_epoch(mesh, verbose=False, noise=a.sigma, rng=a.rng, time=time), a.step())[1]
        rb = b.epoch(verbose=False) if time is None else (run_epoch(net, verbose=False, noise=b.sigma, rng=b.rng, time=time), b.step())[1]
        assert ra == rb and mesh.time == net.time
        assert [(-1 if n.fired_in_wave is None else n.fired_in_wave) for n in mesh.all_neurons()] == net.fired_wave.tolist()
        assert [(-math.inf if n.fired_at is None else n.fired_at) for n in mesh.all_neurons()] == net.fired_at.tolist()
    assert mesh.output_times() == net.output_times()
    for n in mesh.all_neurons():
        n.leak(mesh.time)
    assert np.allclose([n.potential for n in mesh.all_neurons()], net.potential, atol=1e-9)
    assert np.allclose([c.weight for c in mesh.connections.values()], net.weight, atol=1e-12)


def test_the_clock_survives_a_checkpoint(tmp_path, clock):
    grid = GridOfNeurons(across=6, rows=4, weight=None, seed=5)
    grid.interval = 4.0
    teacher = Teacher(grid, seed=1)
    for _ in range(9):
        teacher.epoch(verbose=False)
    path = tmp_path / "clock.json"
    data = checkpoint(grid, path, teacher)
    assert data["time"] == 32.0 and data["interval"] == 4.0 and data["tau"] == 2.0 and data["refractory"] == 5.0
    assert len(data["potentials"]) == len(data["fired_at"]) == len(data["last_update"]) == 24
    restored, _ = restore(path)
    assert restored.time == 32.0 and restored.interval == 4.0 and restored.next_time() == 36.0
    assert [n.fired_at for n in restored.all_neurons()] == [n.fired_at for n in grid.all_neurons()]
    run_epoch(grid, bits=[True, False, True], verbose=False)
    run_epoch(restored, bits=[True, False, True], verbose=False)
    assert [n.fired_in_wave for n in restored.all_neurons()] == [n.fired_in_wave for n in grid.all_neurons()]
    assert restored.time == grid.time == 36.0


def test_columns_run_on_the_same_clock(clock):
    stack = HexColumns(across=4, rows=3, layers=2, seed=1, omega=0)
    stack.interval = 2.0
    run_epoch(stack, verbose=False)
    fired = [n for n in stack.all_neurons() if n.has_fired]
    run_epoch(stack, verbose=False)
    assert stack.time == 2.0 and all(not n.has_fired for n in fired)  # every one of them is still refractory


def test_cli_clock_options_and_validation(tmp_path, capsys):
    save = tmp_path / "t.json"
    assert cli_main(["--headless", "-a", "6", "-r", "4", "--seed", "1", "--epochs", "5", "--interval", "2.5", "--tau", "3", "--refractory", "1",
                     "--save-weights", str(save)]) == 0
    data = json.loads(save.read_text())
    assert data["time"] == 10.0 and data["interval"] == 2.5 and data["tau"] == 3.0 and data["refractory"] == 1.0
    assert Neuron.tau == 2.0 and Neuron.refractory == 5.0  # restored after the command
    assert cli_main(["--headless", "--seeds", "2", "--seed", "1", "-a", "6", "-r", "4", "--epochs", "5", "--interval", "2", "--no-save"]) == 0
    capsys.readouterr()
    assert cli_main(["--headless", "--tau", "0"]) == 2
    assert cli_main(["--headless", "--interval", "-1"]) == 2
    assert "must be positive" in capsys.readouterr().err
