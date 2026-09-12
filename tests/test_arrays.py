"""The array engine is the same network as the object engine: run them side by side."""

import json
import random

import pytest

np = pytest.importorskip("numpy")  # the array engine is optional: without numpy and scipy these tests skip
pytest.importorskip("scipy")

from walnutbutter.arrays import ArrayNetwork  # noqa: E402
from walnutbutter.cartesian import CartesianNodes
from walnutbutter.cli import cli_main
from walnutbutter.grid import GridOfNeurons
from walnutbutter.learning import Teacher, accuracy, reward, stuck_neurons
from walnutbutter.monitor import run_epoch
from walnutbutter.neuron import Neuron
from walnutbutter.persistence import checkpoint, restore


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(Neuron, "verbose", False)


def pair(seed=3, **kwargs):
    """The same mesh twice: one run by objects, one by arrays."""
    mesh = GridOfNeurons(weight=None, seed=seed, **kwargs)
    twin = GridOfNeurons(weight=None, seed=seed, **kwargs)
    return mesh, ArrayNetwork(twin)


def fired_waves(grid) -> list:
    """The wave each neuron fired in, -1 for none, in mesh order."""
    if getattr(grid, "engine", "objects") == "arrays":
        return grid.fired_wave.tolist()
    return [-1 if n.fired_in_wave is None else n.fired_in_wave for n in grid.all_neurons()]


def weights(grid) -> np.ndarray:
    if getattr(grid, "engine", "objects") == "arrays":
        return grid.weight.copy()
    return np.array([grid.connections[i].weight for i in range(1, len(grid.connections) + 1)])


def thresholds(grid) -> np.ndarray:
    if getattr(grid, "engine", "objects") == "arrays":
        return grid.threshold_v.copy()
    return np.array([n.threshold for n in grid.all_neurons()])


def rates(grid) -> np.ndarray:
    if getattr(grid, "engine", "objects") == "arrays":
        return grid.rate.copy()
    return np.array([n.rate for n in grid.all_neurons()])


def test_wrapping_copies_the_mesh_exactly():
    mesh = GridOfNeurons(weight=None, seed=1)
    mesh.use_ecc(False)
    net = ArrayNetwork(mesh)
    assert net.across == 8 and net.rows == 10 and len(net) == 80 and len(net.weight) == len(mesh.connections)
    assert net.permutation == mesh.permutation and net.seed == 1
    assert np.array_equal(weights(net), weights(mesh))
    assert [net.neurons_list[i].name for i in net.input_index] == [n.name for n in mesh.input_row()]
    assert [net.neurons_list[i] for i in net.output_index] == [mesh.get_neuron_at(c, 0) for c in range(8)]
    assert net.mesh is mesh and repr(net).startswith("ArrayNetwork(")


def test_both_engines_draw_the_same_inputs_and_fire_the_same_neurons_wave_by_wave():
    mesh, net = pair()
    for _ in range(50):
        run_epoch(mesh, verbose=False)
        run_epoch(net, verbose=False)
        assert net.input_bits == mesh.input_bits and net.input_pattern == mesh.input_pattern
        assert fired_waves(net) == fired_waves(mesh)
        assert len(net.waves) == len(mesh.waves)
        mesh_index = {n: i for i, n in enumerate(mesh.all_neurons())}
        assert [w.fired.tolist() for w in net.waves] == [sorted(mesh_index[n] for n in w.fired) for w in mesh.waves]
    assert net.epoch == mesh.epoch == 50


@pytest.mark.parametrize("late", ["count", "ignore", "depress"])
@pytest.mark.parametrize("eligibility", ["perturb", "hebb"])
def test_teaching_moves_both_engines_identically(late, eligibility):
    mesh, net = pair(seed=5)
    kw = dict(seed=7, late=late, eligibility=eligibility, homeostasis=0.01, unstick=0.1)  # fast enough to act in the test
    a, b = Teacher(mesh, **kw), Teacher(net, **kw)
    for _ in range(150):
        ra, rb = a.epoch(verbose=False), b.epoch(verbose=False)
        assert ra == rb
        assert fired_waves(net) == fired_waves(mesh)
        assert np.allclose(weights(net), weights(mesh), atol=1e-12)
        assert np.allclose(thresholds(net), thresholds(mesh), atol=1e-12)
        assert np.allclose(rates(net), rates(mesh), atol=1e-12)
    assert a.accuracy_to_date == b.accuracy_to_date and a.unstuck_count == b.unstuck_count
    on_a, off_a = stuck_neurons(mesh)
    on_b, off_b = stuck_neurons(net)
    assert (len(on_a), len(off_a)) == (len(on_b), len(off_b))
    assert a.record()["stuck_on"] == b.record()["stuck_on"]


def test_the_decoded_critics_agree_on_a_hamming_mesh():
    mesh = GridOfNeurons(across=14, rows=6, weight=None, seed=2)
    mesh.use_ecc()
    twin = GridOfNeurons(across=14, rows=6, weight=None, seed=2)
    twin.use_ecc()
    net = ArrayNetwork(twin)
    for critic in ("row", "decoded", "decoded-exact"):
        run_epoch(mesh, verbose=False)
        run_epoch(net, verbose=False)
        assert reward(net, critic=critic) == reward(mesh, critic=critic)


def test_carry_over_keeps_the_same_potentials_within_rounding():
    mesh, net = pair(seed=9, across=6, rows=4)
    for _ in range(30):
        run_epoch(mesh, verbose=False, discharge=False)
        run_epoch(net, verbose=False, discharge=False)
        assert fired_waves(net) == fired_waves(mesh)
        assert np.allclose(net.potential, [n.potential for n in mesh.all_neurons()], atol=1e-9)


def test_an_inactive_connection_carries_nothing_in_either_engine():
    a = GridOfNeurons(weight=1.0, seed=4, across=6, rows=4, omega=0)
    b = GridOfNeurons(weight=1.0, seed=4, across=6, rows=4, omega=0)
    for g in (a, b):
        for c in g.get_neuron_at(0, 3).outgoing:
            c.is_active = False  # the bottom-left neuron is cut off from everyone
    net = ArrayNetwork(b)
    a.set_input([True] + [False] * 5)
    net.set_input([True] + [False] * 5)
    a.fire_input()
    net.fire_input()
    assert fired_waves(net) == fired_waves(a)
    assert sum(w >= 0 for w in fired_waves(net)) == 1  # only the forced neuron fired


def test_the_lattice_runs_on_arrays_too():
    a = CartesianNodes(across=6, rows=5, seed=3)
    a.connect_within(reach=2.0, weight=None)
    b = CartesianNodes(across=6, rows=5, seed=3)
    b.connect_within(reach=2.0, weight=None)
    net = ArrayNetwork(b)
    ta, tb = Teacher(a, seed=1), Teacher(net, seed=1)
    for _ in range(40):
        assert ta.epoch(verbose=False) == tb.epoch(verbose=False)
        assert fired_waves(net) == fired_waves(a)
    assert np.allclose(weights(net), weights(a), atol=1e-12)


def test_checkpoints_cross_between_engines(tmp_path):
    mesh, net = pair(seed=11, across=6, rows=4)
    teacher = Teacher(net, seed=2)
    for _ in range(20):
        teacher.epoch(verbose=False)
    path = tmp_path / "arrays.json"
    data = checkpoint(net, path, teacher)
    assert data["engine"] == "arrays" and data["epoch"] == 20
    assert data["weights"] == net.weight.tolist() and data["thresholds"] == net.threshold_v.tolist()
    restored, record = restore(path)  # an object mesh, with the array engine's weights
    assert np.array_equal(weights(restored), weights(net)) and restored.epoch == 20
    again = ArrayNetwork(restored)
    assert np.array_equal(again.weight, net.weight) and again.epoch == 20
    # and the other way: the object engine's checkpoint loads into arrays
    objects_teacher = Teacher(mesh, seed=2)
    for _ in range(20):
        objects_teacher.epoch(verbose=False)
    path2 = tmp_path / "objects.json"
    checkpoint(mesh, path2, objects_teacher)
    assert json.loads(path2.read_text())["engine"] == "objects"
    wrapped = ArrayNetwork(restore(path2)[0])
    assert np.allclose(wrapped.weight, weights(mesh), atol=1e-12)


def test_fired_neurons_and_accuracy_read_through_the_mesh():
    _, net = pair(seed=6, across=6, rows=4)
    run_epoch(net, verbose=False)
    fired = net.fired_neurons()
    assert fired and all(n.has_fired for n in fired)
    assert len(fired) == sum(w >= 0 for w in fired_waves(net))
    assert 0.0 <= accuracy(net) <= 1.0
    assert net.mesh.epoch == 1 and len(net.mesh.waves) == len(net.waves)


def test_cli_engine_arrays_runs_headless_learns_and_checkpoints(tmp_path, capsys):
    save = tmp_path / "arrays.json"
    assert cli_main(["--engine", "arrays", "--headless", "--epochs", "30", "--seed", "1", "--save-weights", str(save)]) == 0
    err = capsys.readouterr().err
    assert "engine: arrays" in err and "after 30 epochs" in err
    data = json.loads(save.read_text())
    assert data["engine"] == "arrays" and data["epoch"] == 30 and data["learning"]["epochs"] == 30
    # a loaded checkpoint keeps its engine
    assert cli_main(["--load-weights", str(save), "--headless", "--epochs", "5", "--no-save"]) == 0
    assert "engine: arrays" in capsys.readouterr().err
    # unless told otherwise
    assert cli_main(["--load-weights", str(save), "--engine", "objects", "--headless", "--epochs", "5", "--no-save"]) == 0
    assert "engine: arrays" not in capsys.readouterr().err


def test_cli_engine_arrays_seed_batch_and_lattice(tmp_path, capsys):
    assert cli_main(["--engine", "arrays", "--seeds", "2", "--seed", "1", "--epochs", "20", "--no-save"]) == 0
    out = capsys.readouterr().out
    assert out.count("%") >= 4  # two rows of two percentages
    assert cli_main(["--engine", "arrays", "--nodes", "--headless", "--epochs", "10", "--no-save"]) == 0
    assert "engine: arrays" in capsys.readouterr().err


def test_arrays_are_much_faster_on_a_big_mesh():
    import time
    mesh, net = pair(seed=1, across=24, rows=20)
    ta, tb = Teacher(mesh, seed=1), Teacher(net, seed=1)
    def rate(t, n):
        t0 = time.perf_counter()
        for _ in range(n):
            t.epoch(verbose=False)
        return n / (time.perf_counter() - t0)
    objects, arrays = rate(ta, 20), rate(tb, 200)
    assert arrays > objects  # a loose bound: the point is that it is not slower


def test_the_visualizer_draws_an_array_network_through_its_mesh(tmp_path, monkeypatch):
    pytest.importorskip("pygame")
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from walnutbutter import visualizer as viz

    _, net = pair(seed=6, across=6, rows=4)
    teacher = Teacher(net, seed=1)
    teacher.epoch(verbose=False)
    path = tmp_path / "arrays.png"
    viz.save(net, str(path), 200, 150)
    assert path.exists() and path.stat().st_size > 0
    text = viz.caption(net, teacher)
    assert "fired" in text and "scoring" in text
    assert [n.has_fired for n in net.mesh.all_neurons()] == [w >= 0 for w in net.fired_wave.tolist()]
