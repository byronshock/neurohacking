import pytest

from walnutbutter.connection import Connection
from walnutbutter.neuron import Neuron


def test_connection_has_direction_weight_and_state():
    a, b = Neuron("a"), Neuron("b")
    c = Connection(1, a, b, weight=0.25)
    assert c.source is a and c.target is b
    assert c.weight == 0.25
    assert c.is_active


def test_weight_defaults_to_one_and_is_stored_as_float():
    a, b = Neuron("a"), Neuron("b")
    assert Connection(1, a, b).weight == 1.0
    assert isinstance(Connection(2, a, b, weight=2).weight, float)


def test_joins_respects_direction():
    a, b = Neuron("a"), Neuron("b")
    conn = Connection(1, a, b)
    assert conn.joins(a, b)
    assert not conn.joins(b, a)


def test_neuron_cannot_connect_to_itself():
    a = Neuron("a")
    with pytest.raises(ValueError):
        Connection(1, a, a)


def test_connection_holds_references_not_copies():
    a, b = Neuron("a"), Neuron("b")
    conn = Connection(7, a, b, weight=0.5)
    a.name = "renamed"
    assert conn.source.name == "renamed"  # same object, seen through the connection
    assert repr(conn) == "Connection(7: renamed -> b, weight 0.5, active)"
