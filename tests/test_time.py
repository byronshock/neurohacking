"""The clock: nominal milliseconds, one hop per connection, an absolute refractory period, a schedule with a horizon."""

import json

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
    """The defaults, pinned: refractory 5 ms, three hops to it."""
    monkeypatch.setattr(Neuron, "refractory", 5.0)
    monkeypatch.setattr(Neuron, "refractory_hops", 3.0)


HOP = 5.0 / 3.0


def test_the_defaults_and_the_horizon(clock):
    assert Neuron.refractory == 5.0 and Neuron.refractory_hops == 3.0 and Neuron.hop() == pytest.approx(HOP)
    grid = GridOfNeurons(across=4, rows=3, omega=0)
    assert grid.interval == 10.0 and grid.time == 0.0 and grid.next_time() == 0.0 and grid.horizon == 0.0
    run_epoch(grid, verbose=False)
    assert grid.time == 0.0 and grid.horizon == 10.0 and grid.next_time() == 10.0
    run_epoch(grid, verbose=False)
    assert grid.time == 10.0 and grid.horizon == 20.0
    run_epoch(grid, verbose=False, time=25.0)
    assert grid.time == 25.0 and grid.horizon == 35.0 and grid.next_time() == 35.0
    with pytest.raises(ValueError):
        run_epoch(grid, verbose=False, time=30.0)  # the schedule has already run to 35


def test_a_signal_takes_one_hop_and_charge_never_leaks(clock):
    a, b = Neuron("a"), Neuron("b", threshold=1.0)
    a.connect(b, weight=0.5)
    waves = propagate(fire=[a], now=0.0)
    assert [w.time for w in waves] == [0.0, pytest.approx(HOP)] and waves[1].fired == []
    assert b.potential == 0.5 and not b.has_fired
    assert b.receive(0.0, now=1000.0) and b.potential == 0.5  # nothing happens to a quiet neuron, ever
    propagate(fire=[a], now=10.0)  # the second half arrives: the charge was kept until it fired
    assert b.has_fired and b.fired_at == pytest.approx(10.0 + HOP) and b.potential == 0.0 and b.spikes == 1


def test_a_neuron_ignores_signals_and_stimulus_during_its_refractory_period(clock):
    a = Neuron("a", threshold=0.1)
    assert a.receive(1.0, now=0.0) and a.can_fire(0.0)
    a.fire(wave=1, now=0.0)
    assert a.fired_at == 0.0 and a.previous_fired_at is None and a.potential == 0.0  # the spike resets the potential
    assert a.refractory_at(0.0) and a.refractory_at(4.999) and not a.refractory_at(5.0)
    assert not a.receive(1.0, now=3.0) and a.potential == 0.0  # ignored, not integrated
    assert a.receive(1.0, now=5.0) and a.potential == pytest.approx(1.0)  # recovered
    b = Neuron("b")
    b.fire(wave=0, now=0.0)
    waves = propagate(fire=[b], now=2.0)
    assert waves[0].fired == [] and not b.forced  # forced stimulus: refractory wins
    waves = propagate(fire=[b], now=5.0)
    assert waves[0].fired == [b] and b.fired_at == 5.0 and b.previous_fired_at == 0.0 and b.spikes == 2


def test_a_tight_loop_sustains_itself(clock):
    """a -> b -> c -> a, three hops round: a's own spike comes back the instant its refractory period ends, forever."""
    a, b, c = (Neuron(name, threshold=0.5) for name in "abc")
    a.connect(b)
    b.connect(c)
    c.connect(a)
    waves = propagate(fire=[a], now=0.0, until=20.0)
    assert [n.spikes for n in (a, b, c)] == [4, 4, 4]  # a at 0, 5, 10, 15; b and c a hop and two behind
    assert a.previous_fired_at == pytest.approx(10.0) and a.fired_at == pytest.approx(15.0)
    assert [w.time for w in waves][:4] == [0.0, pytest.approx(HOP), pytest.approx(2 * HOP), pytest.approx(5.0)]
    assert propagate(fire=[a], now=20.0, until=20.0) == []  # nothing before the bound


def test_inputs_closer_than_the_refractory_period_are_partly_ignored(clock):
    grid = GridOfNeurons(across=6, rows=3, weight=None, seed=1, omega=0)
    grid.interval = 3.0  # inputs every 3 ms: a neuron forced for one input is still refractory for the next
    run_epoch(grid, bits=[True, True, True], verbose=False)
    assert grid.waves[0].time == 0.0 and grid.waves[0].fired == grid.input_neurons()
    run_epoch(grid, bits=[True, True, True], verbose=False)
    assert grid.time == 3.0 and grid.waves[0].time == 3.0 and grid.waves[0].fired == []  # the stimulus fell on refractory neurons
    assert not any(n.forced for n in grid.input_row())
    run_epoch(grid, bits=[True, True, True], verbose=False, time=20.0)
    assert grid.waves[0].fired == grid.input_neurons()  # everyone has long recovered


def test_outputs_carry_the_time_of_their_spike(clock):
    grid = GridOfNeurons(across=4, rows=2, weight=1.0, omega=0)  # everything fires
    run_epoch(grid, bits=[True, False], verbose=False, time=7.0)
    times = grid.output_times()
    assert all(t is not None and t >= 7.0 + HOP - 1e-9 for t in times)  # the top row is one hop from the bottom row
    grid.reset()
    assert grid.output_times() == [None] * 4


def test_both_engines_keep_the_same_clock(clock):
    np = pytest.importorskip("numpy")
    from walnutbutter.arrays import ArrayNetwork

    def make():
        grid = GridOfNeurons(across=6, rows=4, weight=None, seed=8)
        grid.interval = 3.0  # refractory periods overlap inputs, so timing matters every wave
        return grid

    mesh, net = make(), ArrayNetwork(make())
    a, b = Teacher(mesh, seed=2, rule="reinforce"), Teacher(net, seed=2, rule="reinforce")
    for k in range(60):
        time = None if k % 7 else mesh.horizon + 12.0  # now and then a long gap: everyone recovers
        ra = a.epoch(verbose=False) if time is None else (run_epoch(mesh, verbose=False, noise=a.sigma, rng=a.rng, time=time), a.step())[1]
        rb = b.epoch(verbose=False) if time is None else (run_epoch(net, verbose=False, noise=b.sigma, rng=b.rng, time=time), b.step())[1]
        assert ra == rb and mesh.time == net.time and mesh.horizon == net.horizon
        assert [(-1 if n.fired_in_wave is None else n.fired_in_wave) for n in mesh.all_neurons()] == net.fired_wave.tolist()
        assert [(-np.inf if n.fired_at is None else n.fired_at) for n in mesh.all_neurons()] == net.fired_at.tolist()
        assert [n.spikes for n in mesh.all_neurons()] == net.spikes.tolist()
        assert [n.forced for n in mesh.all_neurons()] == net.forced.tolist()
    assert mesh.output_times() == net.output_times()
    assert np.allclose([n.potential for n in mesh.all_neurons()], net.potential, atol=1e-9)
    assert np.allclose([c.weight for c in mesh.connections.values()], net.weight, atol=1e-12)
    assert [(-np.inf if c.last_signal is None else c.last_signal) for c in mesh.connections.values()] == net.last_signal.tolist()


def test_the_clock_survives_a_checkpoint(tmp_path, clock):
    grid = GridOfNeurons(across=6, rows=4, weight=None, seed=5)
    grid.interval = 4.0
    teacher = Teacher(grid, seed=1)  # the dopamine rule: the pool and the synapse stamps must travel too
    for _ in range(9):
        teacher.epoch(verbose=False)
    path = tmp_path / "clock.json"
    data = checkpoint(grid, path, teacher)
    assert data["time"] == 32.0 and data["horizon"] == 36.0 and data["interval"] == 4.0
    assert data["refractory"] == 5.0 and data["refractory_hops"] == 3.0 and data["learning"]["rule"] == "dopamine"
    assert len(data["potentials"]) == len(data["fired_at"]) == len(data["previous_fired_at"]) == len(data["spikes"]) == 24
    assert len(data["last_signal"]) == len(grid.connections) == len(data["weights"])
    assert data["pending"] == [[t, c.id] for t, c in grid.schedule.pending()] and data["pending"]  # signals in flight at the horizon
    assert data["dopamine"] == grid.dopamine.state() and data["dopamine"]["releases"] > 0
    restored, _ = restore(path)
    assert restored.time == 32.0 and restored.horizon == 36.0 and restored.next_time() == 36.0
    assert [n.fired_at for n in restored.all_neurons()] == [n.fired_at for n in grid.all_neurons()]
    assert [n.spikes for n in restored.all_neurons()] == [n.spikes for n in grid.all_neurons()]
    assert [c.last_signal for c in restored.connections.values()] == [c.last_signal for c in grid.connections.values()]
    assert [(t, c.id) for t, c in restored.schedule.pending()] == [(t, c.id) for t, c in grid.schedule.pending()]
    assert restored.dopamine.state() == grid.dopamine.state()
    run_epoch(grid, bits=[True, False, True], verbose=False)
    run_epoch(restored, bits=[True, False, True], verbose=False)
    assert [n.fired_in_wave for n in restored.all_neurons()] == [n.fired_in_wave for n in grid.all_neurons()]
    assert [c.weight for c in restored.connections.values()] == [c.weight for c in grid.connections.values()]  # and learned the same
    assert restored.time == grid.time == 36.0


def test_columns_run_on_the_same_clock(clock):
    stack = HexColumns(across=4, rows=3, layers=2, seed=1, omega=0)
    stack.interval = 2.0
    bits = [True, False] * 3
    run_epoch(stack, bits=bits, verbose=False)
    assert stack.waves[0].fired == stack.input_neurons()
    run_epoch(stack, bits=bits, verbose=False)  # the same input again, 2 ms later
    assert stack.time == 2.0 and stack.waves[0].time == 2.0 and stack.waves[0].fired == []  # every forced neuron is still refractory


def test_cli_clock_options_and_validation(tmp_path, capsys):
    save = tmp_path / "t.json"
    assert cli_main(["--headless", "-a", "6", "-r", "4", "--seed", "1", "--epochs", "5", "--interval", "2.5", "--refractory", "3",
                     "--refractory-hops", "2", "--save-weights", str(save)]) == 0
    data = json.loads(save.read_text())
    assert data["time"] == 10.0 and data["interval"] == 2.5 and data["refractory"] == 3.0 and data["refractory_hops"] == 2.0
    assert Neuron.refractory == 5.0 and Neuron.refractory_hops == 3.0  # restored after the command
    assert cli_main(["--headless", "--seeds", "2", "--seed", "1", "-a", "6", "-r", "4", "--epochs", "5", "--interval", "2", "--no-save"]) == 0
    capsys.readouterr()
    assert cli_main(["--headless", "--refractory", "0"]) == 2
    assert cli_main(["--headless", "--refractory-hops", "0"]) == 2
    assert cli_main(["--headless", "--interval", "-1"]) == 2
    assert "must be positive" in capsys.readouterr().err
