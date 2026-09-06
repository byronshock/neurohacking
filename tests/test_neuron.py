from neurohacking.neuron import Neuron
from neurohacking.propagation import propagate


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


def test_fire_marks_wave_and_returns_active_outgoing_connections(capsys):
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
