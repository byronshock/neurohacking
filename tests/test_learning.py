import random
import statistics

import pytest

from walnutbutter import learning
from walnutbutter.grid import GridOfNeurons
from walnutbutter.learning import (
    Teacher,
    accuracy,
    delivered_connections,
    expected_outputs,
    output_errors,
    output_row,
    reinforce,
)
from walnutbutter.monitor import main, run_epoch
from walnutbutter.neuron import Neuron


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
    assert "learning reversed (perturb, lr 0.01, sigma 0.1, homeostasis 1e-06 toward 0.4 in [-5, 5], unstick 0.001): accuracy" in teacher.status()
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


def test_reinforce_clips_to_the_grid_weight_range():
    grid = GridOfNeurons(columns=8, rows=4, weight=None, seed=2, omega=0, weight_range=(0.001, 1.0), threshold=2.0)
    run_epoch(grid, verbose=False, noise=0.1, rng=random.Random(2))
    reinforce(grid, advantage=-1.0, lr=50.0)  # a huge negative push: everything touched should hit the floor, not go negative
    touched = [c for c in delivered_connections(grid) if c.target.fired_in_wave != 0 and c.target.noise > 0]
    assert touched
    assert all(c.weight == 0.001 for c in touched)
    assert all(c.weight >= 0.001 for c in grid.connections.values())


def test_rates_track_firing_and_stuck_neurons_are_counted():
    from walnutbutter.learning import stuck_neurons, update_rates, RATE_MEMORY
    grid = GridOfNeurons(columns=4, rows=3, omega=0)
    always, never = grid.get_neuron_at(0, 0), grid.get_neuron_at(1, 0)
    for _ in range(600):
        grid.reset()
        always.fire()
        update_rates(grid)
    assert always.rate > 0.99 and never.rate < 0.01
    on, off = stuck_neurons(grid)
    assert always in on and never in off
    assert not any(n in on or n in off for n in grid.input_row())  # the input row is never counted
    assert abs((0.5 + RATE_MEMORY * 0.5) - 0.505) < 1e-12  # one step from the initial 0.5 toward 1


def test_homeostasis_moves_thresholds_toward_the_target_rate_and_stays_in_range():
    from walnutbutter.learning import THRESHOLD_RANGE, homeostasis
    grid = GridOfNeurons(columns=4, rows=3, omega=0)
    hot, cold = grid.get_neuron_at(0, 0), grid.get_neuron_at(1, 0)
    hot.rate, cold.rate = 1.0, 0.0
    before = {n: n.threshold for n in grid.neurons.values()}
    assert homeostasis(grid, rate=0.1, target=0.5) == 8  # 12 neurons minus the 4 in the input row
    assert hot.threshold == pytest.approx(before[hot] + 0.05)
    assert cold.threshold == pytest.approx(before[cold] - 0.05)
    hot.threshold, cold.threshold = before[hot], before[cold]
    homeostasis(grid, rate=0.1)  # the default target is 0.4
    assert hot.threshold == pytest.approx(before[hot] + 0.06)
    assert cold.threshold == pytest.approx(before[cold] - 0.04)
    hot.threshold, cold.threshold = before[hot], before[cold]
    homeostasis(grid, rate=0.1, target=0.5)
    assert hot.threshold == pytest.approx(before[hot] + 0.05)
    assert cold.threshold == pytest.approx(before[cold] - 0.05)
    assert all(n.threshold == before[n] for n in grid.input_row())
    assert homeostasis(grid, rate=0.0) == 0
    for _ in range(500):
        homeostasis(grid, rate=1.0)
    assert hot.threshold == THRESHOLD_RANGE[1] and cold.threshold == THRESHOLD_RANGE[0]


def test_teacher_homeostasis_reduces_stuck_neurons():
    from walnutbutter.learning import stuck_neurons
    def run(homeostasis):
        grid = GridOfNeurons(columns=8, rows=6, weight=None, seed=1)
        teacher = Teacher(grid, seed=1, homeostasis=homeostasis, target_rate=0.5)
        for _ in range(3000):
            teacher.epoch(verbose=False)
        on, off = stuck_neurons(grid)
        return len(on) + len(off)
    assert run(0.0) > run(0.01)


def test_teacher_validates_homeostasis_and_reports_it():
    grid = main(columns=8, rows=4, seed=1)
    with pytest.raises(ValueError):
        Teacher(grid, homeostasis=-0.1)
    with pytest.raises(ValueError):
        Teacher(grid, target_rate=1.5)
    teacher = Teacher(grid, homeostasis=0.01, target_rate=0.4, seed=1)
    teacher.step()
    assert "homeostasis 0.01 toward 0.4 in [-5, 5]" in teacher.status() and "stuck" in teacher.status()
    default = Teacher(grid, seed=1)
    default.step()
    assert "homeostasis 1e-06 toward 0.4" in default.status() and "sigma 0.1" in default.status()
    assert default.homeostasis == 1e-6 and default.target_rate == 0.4
    off = Teacher(grid, seed=1, homeostasis=0)
    off.step()
    assert "homeostasis" not in off.status()


def test_threshold_range_is_configurable_and_validated():
    from walnutbutter.learning import homeostasis
    grid = GridOfNeurons(columns=4, rows=3, omega=0)
    hot = grid.get_neuron_at(0, 0)
    hot.rate = 1.0
    for _ in range(200):
        homeostasis(grid, rate=1.0, threshold_range=(0.0, 1.5))
    assert hot.threshold == 1.5
    teacher = Teacher(grid, threshold_range=(0, 2), seed=1)
    assert teacher.threshold_range == (0.0, 2.0) and "in [0, 2]" in teacher.status() or teacher.average is None
    with pytest.raises(ValueError):
        Teacher(grid, threshold_range=(3, 1))


def test_discharge_is_the_default_and_carry_over_is_available():
    grid = main(columns=8, rows=4, seed=1)
    default = Teacher(grid, seed=1)
    assert default.carry_over is False
    default.step()
    assert "carry-over" not in default.status()
    carrying = Teacher(grid, seed=1, carry_over=True)
    carrying.step()
    assert "carry-over" in carrying.status()



# --- un-sticking from the output end -------------------------------------------


def test_unstick_touches_only_stuck_output_neurons():
    from walnutbutter.learning import unstick_outputs
    grid = GridOfNeurons(columns=6, rows=4, omega=0)
    outputs = output_row(grid)
    hot, cold, fine = outputs[0], outputs[1], outputs[2]
    hot.rate, cold.rate, fine.rate = 1.0, 0.0, 0.5
    interior = grid.get_neuron_at(3, 1)
    interior.rate = 1.0  # stuck, but not an output: must be left alone
    before = {n: n.threshold for n in grid.neurons.values()}
    nudged = unstick_outputs(grid, rate=0.1)
    assert nudged == [hot, cold]
    assert hot.threshold == pytest.approx(before[hot] + 0.05)
    assert cold.threshold == pytest.approx(before[cold] - 0.05)
    assert fine.threshold == before[fine] and interior.threshold == before[interior]
    assert unstick_outputs(grid, rate=0.0) == []


def test_unstick_stops_once_the_neuron_is_no_longer_stuck_and_respects_the_range():
    from walnutbutter.learning import unstick_outputs
    grid = GridOfNeurons(columns=6, rows=4, omega=0)
    hot = output_row(grid)[0]
    hot.rate = 1.0
    for _ in range(300):
        unstick_outputs(grid, rate=1.0, threshold_range=(-2.0, 2.0))
    assert hot.threshold == 2.0
    hot.rate = 0.6  # out of the stuck band: nothing more happens
    assert unstick_outputs(grid, rate=1.0) == []
    assert hot.threshold == 2.0


def test_teacher_applies_unsticking_and_reports_it():
    grid = main(columns=8, rows=4, weight=1.0, seed=1)  # weight 1: every output fires every epoch
    teacher = Teacher(grid, seed=1, unstick=0.01, homeostasis=0)
    for _ in range(600):
        teacher.epoch(verbose=False)
    assert teacher.unstuck_count > 0
    assert any(n.threshold > 0.25 for n in output_row(grid))
    assert "unstick 0.01" in teacher.status()
    off = Teacher(grid, seed=1, unstick=0)
    off.step()
    assert "unstick" not in off.status() and off.unstuck_count == 0
    with pytest.raises(ValueError):
        Teacher(grid, unstick=-1)
    with pytest.raises(ValueError):
        Teacher(grid, unstick_target=0)


def test_unstick_defaults_on_at_one_thousandth():
    teacher = Teacher(main(columns=8, rows=4, seed=1))
    assert teacher.unstick == 1e-3 and teacher.unstick_target == 0.5
