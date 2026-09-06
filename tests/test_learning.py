import statistics

import pytest

from neurohacking import learning
from neurohacking.grid import GridOfNeurons
from neurohacking.learning import Teacher, accuracy, expected_outputs, output_errors, output_row, teach
from neurohacking.monitor import main, run_epoch
from neurohacking.neuron import Neuron


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(Neuron, "verbose", False)


def test_output_row_is_the_top_row_left_to_right():
    grid = GridOfNeurons(columns=6, rows=4, omega=0)
    row = output_row(grid)
    highest_r = min(r for _, r in grid.neurons)
    assert [n.position[1] for n in row] == [highest_r] * 6
    assert [n.position[0] for n in row] == sorted(n.position[0] for n in row)
    assert row[0] is grid.get_neuron_at(0, 0)


def test_expected_outputs_reversed_and_copy():
    grid = GridOfNeurons(columns=6, rows=4, omega=0)
    grid.set_input([True, True, False, False, False, True])
    assert expected_outputs(grid, "reversed") == [True, False, False, False, True, True]
    assert expected_outputs(grid, "copy") == [True, True, False, False, False, True]
    assert expected_outputs(grid, "all-off") == [False] * 6
    assert expected_outputs(grid, "all-on") == [True] * 6


def test_expected_outputs_need_an_input():
    with pytest.raises(ValueError):
        expected_outputs(GridOfNeurons(columns=6, rows=4))


def test_errors_and_accuracy_against_the_top_row():
    grid = GridOfNeurons(columns=4, rows=3, omega=0)
    grid.set_input([True, False, False, False])  # reversed target: only column 3 should fire
    top = output_row(grid)
    top[3].fire()  # correct
    top[0].fire()  # should not have
    errors = output_errors(grid, "reversed")
    assert [errors[n] for n in top] == [-1, 0, 0, 0]
    assert accuracy(grid, "reversed") == 0.75


def test_teach_moves_weights_toward_the_target_and_keeps_them_in_range():
    grid = main(columns=8, rows=4, weight=1.0, seed=1, omega=0)  # weight 1: the whole top row fires
    before = {c.id: c.weight for c in grid.connections.values()}
    acc = teach(grid, lr=0.1, target="all-off")
    assert 0 <= acc <= 1
    changed = {c.id: c.weight for c in grid.connections.values() if c.weight != before[c.id]}
    assert changed  # something was learned from the errors
    assert all(-1.0 <= w <= 1.0 for w in changed.values())
    # with target all-off every error is -1, so no weight can have gone up
    assert all(changed[i] < before[i] for i in changed)


def test_teach_never_touches_connections_into_the_forced_inputs():
    grid = main(columns=8, rows=4, weight=None, seed=2, omega=0)
    forced = grid.input_neurons()  # bottom-row neurons whose bit is 0 are ordinary neurons
    into_inputs = {c.id: c.weight for n in forced for c in n.incoming}
    teach(grid, lr=0.5, target="all-on")
    assert {c.id: c.weight for n in forced for c in n.incoming} == into_inputs


def test_teach_only_changes_connections_from_neurons_that_fired():
    grid = main(columns=8, rows=4, weight=None, seed=3, omega=0)
    before = {c.id: c.weight for c in grid.connections.values()}
    teach(grid, lr=0.5, target="reversed")
    for c in grid.connections.values():
        if c.weight != before[c.id]:
            assert c.source.has_fired


def test_teach_with_no_errors_changes_nothing():
    grid = GridOfNeurons(columns=4, rows=3, weight=1.0, omega=0)
    grid.set_input([True, False, False, True])
    grid.fire_input()  # weight 1 fires everything, so the top row is all on
    before = [c.weight for c in grid.connections.values()]
    assert teach(grid, lr=0.5, target="all-on") == 1.0
    assert [c.weight for c in grid.connections.values()] == before


@pytest.mark.parametrize("target", ["all-off", "all-on"])
def test_rule_learns_input_independent_targets(target):
    grid = GridOfNeurons(columns=8, rows=4, weight=None, seed=1, omega=0.05)
    teacher = Teacher(grid, target=target, lr=0.05, window=50)
    accs = []
    for _ in range(400):
        run_epoch(grid, verbose=False)
        accs.append(teacher.step())
    assert statistics.mean(accs[:20]) < 1.0  # it did not start out right
    assert statistics.mean(accs[-50:]) > 0.9  # but it got there


def test_teacher_validates_and_tracks():
    grid = main(columns=8, rows=4, seed=1)
    with pytest.raises(ValueError):
        Teacher(grid, target="upside-down")
    with pytest.raises(ValueError):
        Teacher(grid, lr=-1)
    teacher = Teacher(grid, target="reversed", lr=0.05, window=10)
    assert "no epochs yet" in teacher.status()
    first = teacher.step()
    assert teacher.epochs == 1 and teacher.last_accuracy == first and teacher.average == first
    run_epoch(grid, verbose=False)
    teacher.step()
    assert teacher.epochs == 2 and 0 <= teacher.average <= 1
    assert "learning reversed (lr 0.05): accuracy" in teacher.status()


def test_targets_registry_has_the_four_builtin_targets():
    assert set(learning.TARGETS) == {"reversed", "copy", "all-off", "all-on"}
