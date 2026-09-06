from neurohacking.neuron import Neuron


def test_connect_stores_one_shared_connection_on_both_neurons():
    a, b = Neuron("a"), Neuron("b")
    conn = a.connect(b, connection_id=3)
    assert a.connections == [conn]
    assert b.connections == [conn]
    assert conn.id == 3 and conn.is_active


def test_connection_to_and_neighbours():
    a, b, c = Neuron("a"), Neuron("b"), Neuron("c")
    conn = a.connect(b)
    assert a.connection_to(b) is conn
    assert b.connection_to(a) is conn
    assert a.connection_to(c) is None
    assert a.neighbours() == [b]


def test_activate_marks_fired_and_propagates_both_ways():
    a, b = Neuron("a"), Neuron("b")
    a.connect(b)
    b.activate()  # signal travels from b to a, not just a to b
    assert a.has_fired and b.has_fired


def test_connected_neurons_fire_once_each(capsys):
    a, b = Neuron("a"), Neuron("b")
    a.connect(b)
    a.activate()  # would recurse forever without the has_fired guard
    lines = capsys.readouterr().out.splitlines()
    assert lines == ["a received a signal.", "b received a signal."]


def test_already_fired_neuron_ignores_signal(capsys):
    a = Neuron("a")
    a.activate()
    a.activate()
    assert capsys.readouterr().out.count("received a signal") == 1


def test_inactive_connection_does_not_propagate():
    a, b = Neuron("a"), Neuron("b")
    a.connect(b).is_active = False
    a.activate()
    assert not b.has_fired


def test_reset_allows_firing_again(capsys):
    a = Neuron("a")
    a.activate()
    a.reset()
    assert not a.has_fired
    a.activate()
    assert capsys.readouterr().out.count("received a signal") == 2


def test_list_connections_shows_ids_and_state(capsys):
    a, b, c = Neuron("a"), Neuron("b"), Neuron("c")
    a.connect(b, 1)
    a.connect(c, 2).is_active = False
    a.list_connections()
    out = capsys.readouterr().out
    assert "#1 b" in out and "#2 c (inactive)" in out
