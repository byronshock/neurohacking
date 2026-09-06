import random
import statistics

import pytest

from neurohacking import learning
from neurohacking.grid import GridOfNeurons
from neurohacking.learning import (
    Teacher,
    accuracy,
    delivered_connections,
    expected_outputs,
    output_errors,
    output_row,
    reinforce,
)
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


def test_expected_outputs_for_each_target():
    grid = GridOfNeurons(columns=6, rows=4, omega=0)
    grid.set_input([True, True, False, False, False, True])
    assert expected_outputs(grid, "reversed") == [True, False, False, False, True, True]
    assert expected_outputs(grid, "copy") == [True, True, False, False, False, True]
    assert expected_outputs(grid, "all-off") == [False] * 6
    assert expected_outputs(grid, "all-on") == [True] * 6
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


def test_run_epoch_noise_gives_every_neuron_a_remembered_starting_potential():
    grid = GridOfNeurons(columns=6, rows=4, weight=None, seed=1)
    run_epoch(grid, verbose=False, noise=0.1, rng=random.Random(1))
    noises = [n.noise for n in grid.neurons.values()]
    assert len(set(noises)) > 1 and abs(statistics.mean(noises)) < 0.05
    assert 0.05 < statistics.pstdev(noises) < 0.15
    forced = grid.input_neurons()[0]
    assert forced.has_fired and forced.fired_in_wave == 0
    run_epoch(grid, verbose=False)  # without noise everything starts from 0 again
    assert all(n.noise == 0.0 for n in grid.neurons.values())


def test_delivered_connections_are_those_whose_source_fired():
    grid = main(columns=8, rows=4, weight=None, seed=1)
    delivered = delivered_connections(grid)
    assert delivered
    assert all(c.source.has_fired for c in delivered)
    assert all(c in delivered for c in grid.connections.values() if c.source.has_fired and c.is_active)


def test_reinforce_moves_delivered_weights_by_advantage_times_noise():
    grid = GridOfNeurons(columns=8, rows=4, weight=None, seed=2, omega=0)
    run_epoch(grid, verbose=False, noise=0.1, rng=random.Random(2))
    before = {c.id: c.weight for c in grid.connections.values()}
    changed = reinforce(grid, advantage=0.5, lr=0.01, sigma=0.1)
    assert changed > 0
    for c in grid.connections.values():
        if c.target.fired_in_wave == 0 or c not in delivered_connections(grid):
            assert c.weight == before[c.id]  # forced inputs and idle connections untouched
        else:
            expected = max(-1.0, min(1.0, before[c.id] + 0.01 * 0.5 * c.target.noise / 0.1))
            assert c.weight == pytest.approx(expected)


def test_reinforce_with_zero_advantage_changes_nothing():
    grid = main(columns=8, rows=4, weight=None, seed=3)
    before = [c.weight for c in grid.connections.values()]
    assert reinforce(grid, advantage=0.0) == 0
    assert [c.weight for c in grid.connections.values()] == before


def test_hebbian_eligibility_uses_target_firing_and_keeps_weights_in_range():
    grid = main(columns=8, rows=4, weight=None, seed=4)
    before = {c.id: c.weight for c in grid.connections.values()}
    reinforce(grid, advantage=-1.0, lr=0.5, eligibility="hebb")
    for c in delivered_connections(grid):
        if c.target.fired_in_wave == 0:
            continue
        if c.target.has_fired:
            assert c.weight <= before[c.id]  # negative advantage x positive eligibility
        else:
            assert c.weight >= before[c.id]
        assert -1.0 <= c.weight <= 1.0
    with pytest.raises(ValueError):
        reinforce(grid, 1.0, eligibility="magic")


def test_global_reward_learns_to_silence_the_output():
    # With 18 inputs per neuron the reward has more perturbations to attribute, so use a higher rate.
    grid = GridOfNeurons(columns=8, rows=4, weight=None, seed=2, omega=0.05)
    teacher = Teacher(grid, target="all-off", lr=0.1, seed=2)
    rewards = [teacher.epoch(verbose=False) for _ in range(2000)]
    assert statistics.mean(rewards[:100]) < 0.9
    assert statistics.mean(rewards[-200:]) > 0.9


def test_global_reward_learns_to_light_the_output():
    # Reinforcement is slow when the network must become more active; ask for clear progress, not perfection.
    grid = GridOfNeurons(columns=8, rows=4, weight=None, seed=1, omega=0.05)
    teacher = Teacher(grid, target="all-on", lr=0.03, seed=1)
    rewards = [teacher.epoch(verbose=False) for _ in range(2500)]
    assert statistics.mean(rewards[-300:]) > statistics.mean(rewards[:300]) + 0.25


def test_teacher_validates_tracks_and_reports():
    grid = main(columns=8, rows=4, seed=1)
    with pytest.raises(ValueError):
        Teacher(grid, target="upside-down")
    with pytest.raises(ValueError):
        Teacher(grid, eligibility="magic")
    with pytest.raises(ValueError):
        Teacher(grid, lr=-1)
    teacher = Teacher(grid, target="reversed", lr=0.01, window=10, seed=1)
    assert "no epochs yet" in teacher.status()
    first = teacher.step()
    assert teacher.epochs == 1 and teacher.last_reward == first == teacher.average == teacher.baseline
    second = teacher.epoch(verbose=False)
    assert teacher.epochs == 2 and 0 <= teacher.average <= 1 and 0 <= second <= 1
    assert grid.epoch == 2
    assert "learning reversed (perturb, lr 0.01): accuracy" in teacher.status()
    hebb = Teacher(grid, eligibility="hebb")
    assert hebb.sigma == 0.0  # no exploration noise for the Hebbian variant


def test_teacher_with_seed_is_reproducible():
    def run(seed):
        grid = GridOfNeurons(columns=8, rows=4, weight=None, seed=1)
        teacher = Teacher(grid, target="copy", seed=seed)
        for _ in range(30):
            teacher.epoch(verbose=False)
        return [c.weight for c in grid.connections.values()]

    assert run(5) == run(5)
    assert run(5) != run(6)


def test_targets_registry_has_the_four_builtin_targets():
    assert set(learning.TARGETS) == {"reversed", "copy", "all-off", "all-on"}


def test_accuracy_to_date_is_the_mean_over_all_epochs():
    grid = main(columns=8, rows=4, seed=1)
    teacher = Teacher(grid, seed=1)
    assert teacher.accuracy_to_date is None
    rewards = [teacher.step()] + [teacher.epoch(verbose=False) for _ in range(9)]
    assert teacher.accuracy_to_date == pytest.approx(sum(rewards) / 10)
    assert "to date over 10 epochs" in teacher.status()
