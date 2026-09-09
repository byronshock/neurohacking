import pytest
from walnutbutter.neuron import Neuron
from walnutbutter.propagation import propagate


def test_connect_is_one_way_and_recorded_on_both_ends():
    a, b = Neuron("a"), Neuron("b")
    conn = a.connect(b, connection_id=3, weight=0.4)
    assert a.outgoing == [conn] and a.incoming == []
    assert b.incoming == [conn] and b.outgoing == []
    assert conn.id == 3 and conn.weight == 0.4 and conn.is_active


def test_connection_to_targets_and_sources():
    a, b, c = Neuron("a"), Neuron("b"), Neuron("c")
    conn = a.connect(b)
    assert a.connection_to(b) is conn
    assert b.connection_to(a) is None  # no connection in the other direction
    assert a.connection_to(c) is None
    assert a.targets() == [b] and b.sources() == [a]


def test_fire_marks_wave_and_returns_active_outgoing_connections(capsys, monkeypatch):
    monkeypatch.setattr(Neuron, "verbose", True)
    a, b, c = Neuron("a"), Neuron("b"), Neuron("c")
    to_b = a.connect(b)
    a.connect(c).is_active = False
    assert a.fire(wave=2) == [to_b]
    assert a.has_fired and a.fired_in_wave == 2
    assert not b.has_fired  # fire() delivers nothing itself
    assert capsys.readouterr().out == "a fired in wave 2.\n"


def test_receive_accumulates_but_never_fires():
    a = Neuron("a", threshold=1.0)
    a.receive(0.5)
    assert a.potential == 0.5 and not a.ready
    a.receive(0.5)
    assert a.potential == 1.0 and a.ready and not a.has_fired


def test_fired_neuron_ignores_further_input():
    a = Neuron("a")
    a.fire()
    a.receive(5.0)
    assert a.potential == 0.0


def test_negative_input_inhibits():
    a = Neuron("a", threshold=1.0)
    a.receive(1.0)
    a.receive(-0.5)
    assert a.potential == 0.5 and not a.ready


def test_reset_clears_everything(capsys):
    a = Neuron("a", threshold=1.0)
    a.receive(0.5)
    a.fire(wave=3)
    a.reset()
    assert not a.has_fired and a.fired_in_wave is None and a.potential == 0.0


def test_signal_only_travels_in_the_connection_direction(capsys):
    a, b = Neuron("a"), Neuron("b")
    a.connect(b)
    propagate(fire=[b])
    assert b.has_fired and not a.has_fired
    b.reset()
    propagate(fire=[a])
    assert a.has_fired and b.has_fired


def test_list_connections_shows_ids_weights_and_state(capsys):
    a, b, c = Neuron("a"), Neuron("b"), Neuron("c")
    a.connect(b, 1, weight=0.75)
    a.connect(c, 2).is_active = False
    a.list_connections()
    out = capsys.readouterr().out
    assert "#1 b, weight 0.75" in out and "#2 c, weight 1 (inactive)" in out


def test_verbose_false_silences_firing(capsys, monkeypatch):
    monkeypatch.setattr(Neuron, "verbose", False)
    a = Neuron("a")
    a.fire()
    assert a.has_fired and capsys.readouterr().out == ""


# --- reset keeps sub-threshold charge -------------------------------------------


def test_reset_discharges_only_neurons_that_fired():
    fired, quiet = Neuron("f"), Neuron("q")
    fired.receive(0.3)
    fired.fire()
    quiet.receive(0.1)
    fired.reset(discharge=False)
    quiet.reset(discharge=False)
    assert fired.potential == 0.0 and not fired.has_fired
    assert quiet.potential == 0.1 and not quiet.has_fired  # carried over


def test_reset_discharges_by_default():
    quiet = Neuron("q")
    quiet.receive(0.1)
    quiet.reset()
    assert quiet.potential == 0.0


def test_charge_accumulates_across_epochs_until_the_neuron_fires(capsys):
    a, b = Neuron("a"), Neuron("b", threshold=0.25)
    a.connect(b, weight=0.1)
    fired_on = None
    for epoch in range(1, 6):
        a.reset(discharge=False)
        b.reset(discharge=False)
        propagate(fire=[a])
        if b.has_fired:
            fired_on = epoch
            break
    assert fired_on == 3  # 0.1 + 0.1 + 0.1 reaches 0.25 on the third epoch
    b.reset(discharge=False)
    assert b.potential == 0.0  # and it discharged because it fired



# --- minimum potential -------------------------------------------------------------


def test_inhibition_cannot_push_potential_below_the_minimum():
    a = Neuron("a", minimum_potential=-1.0)
    a.receive(-0.7)
    a.receive(-0.7)
    assert a.potential == -1.0
    a.receive(0.5)
    assert a.potential == pytest.approx(-0.5)  # recovery starts from the floor, not from -1.4


def test_minimum_potential_defaults_to_minus_one_and_is_configurable():
    assert Neuron("a").minimum_potential == -1.0
    b = Neuron("b", minimum_potential=-0.2)
    b.receive(-5.0)
    assert b.potential == -0.2


def test_carried_charge_respects_the_floor_across_epochs(capsys):
    a, b = Neuron("a"), Neuron("b", minimum_potential=-0.5)
    a.connect(b, weight=-0.4)
    for _ in range(5):
        a.reset(discharge=False)
        b.reset(discharge=False)
        propagate(fire=[a])
    assert b.potential == -0.5
