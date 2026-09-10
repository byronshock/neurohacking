import json

import pytest

from walnutbutter.grid import GridOfNeurons
from walnutbutter.learning import Teacher
from walnutbutter.monitor import main, run_epoch
from walnutbutter.neuron import Neuron
from walnutbutter.persistence import checkpoint, load_weights, read_checkpoint, restore, resume_teacher


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(Neuron, "verbose", False)


def test_checkpoint_round_trips_weights_settings_and_permutation(tmp_path):
    grid = main(across=8, rows=6, seed=5, omega=0.1)
    teacher = Teacher(grid, seed=5)
    for _ in range(20):
        teacher.epoch(verbose=False)
    path = tmp_path / "w.json"
    written = checkpoint(grid, path, teacher)

    restored, data = restore(path)
    assert data == written
    assert (restored.across, restored.rows, restored.omega, restored.seed) == (8, 6, 0.1, 5)
    assert restored.permutation == grid.permutation
    assert restored.epoch == grid.epoch == 21
    assert [c.weight for c in restored.connections.values()] == [c.weight for c in grid.connections.values()]
    # the mesh itself was rebuilt identically: same shortcuts, same neighbours
    assert [(c.source.name, c.target.name, c.kind) for c in restored.connections.values()] == [
        (c.source.name, c.target.name, c.kind) for c in grid.connections.values()
    ]


def test_checkpoint_is_written_atomically_and_is_json(tmp_path):
    grid = GridOfNeurons(across=4, rows=3, seed=1)
    path = tmp_path / "w.json"
    checkpoint(grid, path)
    assert json.loads(path.read_text())["connections"] == len(grid.connections)
    assert not (tmp_path / "w.json.tmp").exists()


def test_resume_teacher_continues_the_statistics(tmp_path):
    grid = main(across=8, rows=4, seed=2)
    teacher = Teacher(grid, target="all-off", seed=2)
    for _ in range(30):
        teacher.epoch(verbose=False)
    path = tmp_path / "w.json"
    checkpoint(grid, path, teacher)

    restored, data = restore(path)
    fresh = Teacher(restored, target="all-off", seed=3)
    resume_teacher(fresh, data)
    assert fresh.epochs == 30
    assert fresh.accuracy_to_date == pytest.approx(teacher.accuracy_to_date)
    assert fresh.average == teacher.average and fresh.baseline == teacher.baseline
    assert "to date over 30 epochs" in fresh.status()


def test_load_weights_rejects_a_different_mesh(tmp_path):
    grid = GridOfNeurons(across=6, rows=4, seed=1)
    path = tmp_path / "w.json"
    checkpoint(grid, path)
    data = read_checkpoint(path)
    with pytest.raises(ValueError):
        load_weights(GridOfNeurons(across=6, rows=5, seed=1), data)
    with pytest.raises(ValueError, match="seed"):
        load_weights(GridOfNeurons(across=6, rows=4, seed=2), data)  # same size, different shortcuts
    forged = dict(data, seed=2)  # even with the seed faked, the recorded shortcuts give it away
    with pytest.raises(ValueError, match="shortcuts"):
        load_weights(GridOfNeurons(across=6, rows=4, seed=2), forged)


def test_restore_refuses_an_unseeded_mesh(tmp_path):
    grid = GridOfNeurons(across=4, rows=3)  # no seed: shortcuts cannot be rebuilt
    path = tmp_path / "w.json"
    checkpoint(grid, path)
    with pytest.raises(ValueError, match="seed"):
        restore(path)


def test_unknown_format_is_rejected(tmp_path):
    path = tmp_path / "w.json"
    path.write_text(json.dumps({"format": 99}))
    with pytest.raises(ValueError):
        read_checkpoint(path)


def test_checkpoint_without_teacher_has_no_learning_record(tmp_path):
    grid = GridOfNeurons(across=4, rows=3, seed=1)
    path = tmp_path / "w.json"
    checkpoint(grid, path)
    data = read_checkpoint(path)
    assert "learning" not in data
    teacher = Teacher(restore(path)[0])
    resume_teacher(teacher, data)  # a no-op
    assert teacher.epochs == 0


def test_checkpoint_keeps_the_weight_range(tmp_path):
    grid = GridOfNeurons(across=6, rows=4, weight=None, seed=1, weight_range=(0.01, 1.0))
    path = tmp_path / "w.json"
    checkpoint(grid, path)
    restored, data = restore(path)
    assert data["weight_range"] == [0.01, 1.0]
    assert restored.weight_range == (0.01, 1.0)
    assert [c.weight for c in restored.connections.values()] == [c.weight for c in grid.connections.values()]


def test_checkpoint_keeps_per_neuron_thresholds_and_rates(tmp_path):
    grid = main(across=8, rows=4, seed=1)
    teacher = Teacher(grid, seed=1, homeostasis=0.05)
    for _ in range(200):
        teacher.epoch(verbose=False)
    thresholds = [n.threshold for n in grid.neurons.values()]
    assert len(set(thresholds)) > 1  # homeostasis has spread them out
    path = tmp_path / "w.json"
    checkpoint(grid, path, teacher)
    restored, data = restore(path)
    assert [n.threshold for n in restored.neurons.values()] == thresholds
    assert [n.rate for n in restored.neurons.values()] == [n.rate for n in grid.neurons.values()]
    assert data["learning"]["homeostasis"] == 0.05


def test_checkpoint_keeps_the_minimum_potential(tmp_path):
    grid = GridOfNeurons(across=6, rows=4, seed=1, minimum_potential=-0.4)
    path = tmp_path / "w.json"
    checkpoint(grid, path)
    restored, data = restore(path)
    assert data["minimum_potential"] == -0.4
    assert all(n.minimum_potential == -0.4 for n in restored.neurons.values())


def test_checkpoint_without_a_floor_restores_without_one(tmp_path):
    grid = GridOfNeurons(across=6, rows=4, seed=1)
    path = tmp_path / "w.json"
    data = checkpoint(grid, path)
    del data["minimum_potential"]
    path.write_text(json.dumps(data))
    restored, _ = restore(path)
    assert all(n.minimum_potential == float("-inf") for n in restored.neurons.values())


def test_checkpoint_records_the_unstick_settings(tmp_path):
    grid = main(across=8, rows=4, seed=1)
    teacher = Teacher(grid, seed=1, unstick=0.005, unstick_target=0.45)
    teacher.step()
    path = tmp_path / "w.json"
    data = checkpoint(grid, path, teacher)
    assert data["learning"]["unstick"] == 0.005 and data["learning"]["unstick_target"] == 0.45


def test_checkpoint_carries_the_accuracy_history_and_resume_continues_it(tmp_path):
    grid = main(across=8, rows=4, seed=1)
    teacher = Teacher(grid, seed=1)
    teacher.step()
    for i in range(3):
        teacher.epoch(verbose=False)
        teacher.record(elapsed=float(i), epochs_per_second=100.0)
    path = tmp_path / "w.json"
    data = checkpoint(grid, path, teacher)
    assert len(data["learning"]["history"]) == 3
    restored, data = restore(path)
    resumed = Teacher(restored, seed=2)
    resume_teacher(resumed, data)
    assert resumed.history == teacher.history
    resumed.epoch(verbose=False)
    resumed.record(elapsed=0.5)
    assert len(resumed.history) == 4 and resumed.history[-1]["epoch"] == 5
