from error_rate_monitor.neuron import Neuron


def test_connect_adds_active_pathway():
    a, b = Neuron("a"), Neuron("b")
    a.connect(b)
    assert a.connections == [(b, True)]


def test_activate_marks_fired_and_propagates():
    a, b = Neuron("a"), Neuron("b")
    a.connect(b)
    a.activate()
    assert a.has_fired
    assert b.has_fired


def test_mutually_connected_neurons_fire_once_each(capsys):
    a, b = Neuron("a"), Neuron("b")
    a.connect(b)
    b.connect(a)
    a.activate()  # would recurse forever without the has_fired guard
    lines = capsys.readouterr().out.splitlines()
    assert lines == ["a received a signal.", "b received a signal."]


def test_already_fired_neuron_ignores_signal(capsys):
    a = Neuron("a")
    a.activate()
    a.activate()
    assert capsys.readouterr().out.count("received a signal") == 1


def test_inactive_pathway_does_not_propagate():
    a, b = Neuron("a"), Neuron("b")
    a.connections.append((b, False))
    a.activate()
    assert not b.has_fired


def test_reset_allows_firing_again(capsys):
    a = Neuron("a")
    a.activate()
    a.reset()
    assert not a.has_fired
    a.activate()
    assert capsys.readouterr().out.count("received a signal") == 2
