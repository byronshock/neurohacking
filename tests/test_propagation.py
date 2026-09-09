import pytest

from walnutbutter.grid import GridOfNeurons
from walnutbutter.neuron import Neuron
from walnutbutter.propagation import Signal, Wave, propagate


def chain(n, weight=1.0, threshold=1.0):
    """a0 -> a1 -> ... -> a(n-1), one-way."""
    neurons = [Neuron(f"n{i}", threshold=threshold) for i in range(n)]
    for i in range(n - 1):
        neurons[i].connect(neurons[i + 1], connection_id=i + 1, weight=weight)
    return neurons


def test_forced_fire_travels_down_a_chain_one_wave_per_hop(capsys):
    neurons = chain(4)
    waves = propagate(fire=[neurons[0]])
    assert [w.number for w in waves] == [0, 1, 2, 3]  # the last neuron has nothing to send
    assert [w.fired for w in waves] == [[n] for n in neurons]
    assert [n.fired_in_wave for n in neurons] == [0, 1, 2, 3]


def test_waves_record_the_signals_delivered(capsys):
    a, b = chain(2)
    waves = propagate(fire=[a])
    assert waves[1].delivered == [a.connection_to(b)]  # the queue is the connections themselves
    (signal,) = waves[1].signals()
    assert isinstance(signal, Signal)
    assert signal.connection is a.connection_to(b)
    assert signal.target is b and signal.amount == 1.0 and signal.wave == 1


def test_two_way_pair_fires_once_each(capsys, monkeypatch):
    monkeypatch.setattr(Neuron, "verbose", True)
    a, b = Neuron("a"), Neuron("b")
    a.connect(b)
    b.connect(a)
    waves = propagate(fire=[a])
    assert (a.fired_in_wave, b.fired_in_wave) == (0, 1)
    assert capsys.readouterr().out.count("fired") == 2
    assert len(waves) == 3  # wave 2 delivers b's signal back to a, which ignores it


def test_all_signals_in_a_wave_are_delivered_before_anyone_fires(capsys):
    # Two half-strength inputs arriving in the same wave add up and fire the target in that wave.
    a, b, c = Neuron("a"), Neuron("b"), Neuron("c", threshold=1.0)
    a.connect(c, weight=0.5)
    b.connect(c, weight=0.5)
    propagate(fire=[a, b])
    assert c.fired_in_wave == 1


def test_inputs_below_threshold_do_not_fire_and_leave_potential(capsys):
    a, b = chain(2, weight=0.75)
    propagate(fire=[a])
    assert not b.has_fired and b.potential == 0.75


@pytest.mark.parametrize("order", ["excite first", "inhibit first"])
def test_inhibition_result_does_not_depend_on_order(order, capsys):
    # With recursion, c could fire on the excitatory input before the inhibitory
    # one arrived. With wave delivery the net input is 0 either way.
    excite, inhibit, c = Neuron("e"), Neuron("i"), Neuron("c", threshold=1.0)
    excite.connect(c, weight=1.0)
    inhibit.connect(c, weight=-1.0)
    stimulus = [excite, inhibit] if order == "excite first" else [inhibit, excite]
    propagate(fire=stimulus)
    assert not c.has_fired and c.potential == 0.0


def test_external_inputs_fire_only_when_they_reach_threshold(capsys):
    a, b = Neuron("a", threshold=1.0), Neuron("b", threshold=1.0)
    waves = propagate(inputs={a: 1.0, b: 0.5})
    assert a.fired_in_wave == 0 and not b.has_fired and b.potential == 0.5
    assert waves[0].fired == [a]


def test_forced_neuron_ignores_threshold_and_is_not_fired_twice(capsys, monkeypatch):
    monkeypatch.setattr(Neuron, "verbose", True)
    a = Neuron("a", threshold=100.0)
    propagate(fire=[a, a])
    assert a.fired_in_wave == 0
    assert capsys.readouterr().out.count("fired") == 1


def test_already_fired_neuron_is_skipped(capsys):
    a = Neuron("a")
    a.fire()
    waves = propagate(fire=[a])
    assert waves[0].fired == []


def test_empty_stimulus_gives_one_empty_wave():
    assert propagate() == [Wave(number=0)]


def test_grid_waves_match_hex_distance_from_origin(capsys):
    grid = GridOfNeurons(columns=9, rows=7, omega=0)  # shortcuts would let waves jump
    waves = grid.activate_origin()
    assert grid.waves is waves
    assert waves[0].fired == [grid.get_origin_neuron()]
    assert len(waves[1].fired) == 18  # both rings around the origin fire in wave 1
    for (q, r), neuron in grid.neurons.items():
        distance = max(abs(q), abs(r), abs(q + r))
        assert neuron.fired_in_wave == (distance + 1) // 2  # two steps per wave


def test_grid_accepts_multiple_stimuli_in_one_epoch(capsys):
    grid = GridOfNeurons(columns=9, rows=7)
    origin, corner = grid.get_origin_neuron(), grid.get_neuron_at(0, 3)  # centre and left edge
    waves = grid.propagate(fire=[origin, corner])
    assert waves[0].fired == [origin, corner]
    assert len(grid.fired_neurons()) == len(grid.neurons)
    assert grid.get_neuron_at(1, 3).fired_in_wave == 1  # reached from the edge, not the origin


def test_grid_reset_clears_waves(capsys):
    grid = GridOfNeurons(columns=3, rows=3)
    grid.activate_origin()
    grid.reset()
    assert grid.waves == [] and grid.fired_neurons() == []


def test_large_grid_has_no_recursion_limit(capsys):
    grid = GridOfNeurons(columns=80, rows=60, omega=0)  # 4800 neurons; recursion died near 1000
    grid.activate_origin()
    assert len(grid.fired_neurons()) == len(grid.neurons)
    assert len(grid.waves) > 20  # two cells per wave now; recursion would still have died
