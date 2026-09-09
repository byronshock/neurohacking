import pytest

from walnutbutter import main
from walnutbutter.cli import cli_main
from walnutbutter.grid import GridOfNeurons
from walnutbutter.monitor import run_epoch
from walnutbutter.neuron import Neuron


def test_main_fires_the_bottom_row_and_returns_the_grid(capsys):
    grid = main(columns=4, rows=3, weight=1.0, seed=1)
    out = capsys.readouterr().out
    assert len(grid.fired_neurons()) == 12
    assert grid.waves[0].fired == grid.input_neurons()
    assert all(n.position[1] == max(r for _, r in grid.neurons) for n in grid.waves[0].fired)
    assert "input " in out and "-> bottom row " in out


def test_main_complement_codes_the_input():
    grid = main(columns=6, rows=3, weight=1.0, input_bits=[True, False, True], permute=False)
    assert grid.input_coded == [True, False, True, False, True, False]
    assert grid.input_pattern == grid.input_coded
    assert len(grid.waves[0].fired) == 3


def test_main_permutes_the_coded_input_with_a_fixed_permutation(capsys):
    grid = main(columns=8, rows=4, weight=1.0, seed=1, input_bits=[True, True, False, False])
    assert grid.input_coded == [True, True, False, False, False, False, True, True]
    assert sorted(grid.permutation) == list(range(8)) and grid.permutation != list(range(8))
    assert grid.input_pattern == [grid.input_coded[i] for i in grid.permutation]
    first_permutation = list(grid.permutation)
    run_epoch(grid)
    assert grid.permutation == first_permutation  # the same scramble every epoch
    assert grid.input_pattern == [grid.input_coded[i] for i in grid.permutation]
    assert sum(grid.input_pattern) == 4


def test_run_epoch_requires_even_columns_and_the_right_bit_count(capsys):
    with pytest.raises(ValueError):
        run_epoch(GridOfNeurons(columns=5, rows=3))
    with pytest.raises(ValueError):
        run_epoch(GridOfNeurons(columns=6, rows=3), bits=[True, False])


def test_run_epoch_resets_the_mesh_and_presents_a_new_input(capsys):
    grid = main(columns=8, rows=4, weight=1.0, seed=3)
    first_bits, first_fired = grid.input_bits, [n.fired_in_wave for n in grid.neurons.values()]
    assert grid.epoch == 1
    seen = {tuple(first_bits)}
    for _ in range(6):
        run_epoch(grid)
        seen.add(tuple(grid.input_bits))
        assert all(n.potential >= 0 or n.has_fired for n in grid.neurons.values())
    assert grid.epoch == 7
    assert len(seen) > 1  # the input actually changes between epochs
    # a second grid with the same seed replays the same sequence of inputs
    other = main(columns=8, rows=4, weight=1.0, seed=3)
    for _ in range(6):
        run_epoch(other)
    assert other.input_bits == grid.input_bits


def test_run_epoch_clears_every_neuron_before_firing(capsys):
    grid = main(columns=6, rows=3, weight=1.0, seed=1, permute=False)
    fired_before = {n.name: n.fired_in_wave for n in grid.neurons.values()}
    run_epoch(grid, bits=[False, True, False])  # a specific, different input
    assert grid.waves[0].fired == grid.input_neurons()
    assert grid.input_pattern == [False, True, False, True, False, True]
    assert any(fired_before[n.name] != n.fired_in_wave for n in grid.neurons.values())
    assert "epoch 2: input 010 -> coded 010101 -> bottom row 010101" in capsys.readouterr().out


def test_run_epoch_keeps_weights_and_shortcuts(capsys):
    grid = main(columns=8, rows=4, seed=2, omega=0.2)
    weights = [c.weight for c in grid.connections.values()]
    shortcuts = len(grid.small_world_connections())
    run_epoch(grid)
    assert [c.weight for c in grid.connections.values()] == weights
    assert len(grid.small_world_connections()) == shortcuts


def test_main_random_input_is_reproducible_by_seed(capsys):
    a = main(columns=8, rows=4, seed=4)
    b = main(columns=8, rows=4, seed=4)
    assert a.input_pattern == b.input_pattern
    assert [n.has_fired for n in a.neurons.values()] == [n.has_fired for n in b.neurons.values()]
    assert a.input_pattern != main(columns=8, rows=4, seed=5).input_pattern


def test_cli_runs_and_returns_zero(capsys):
    assert cli_main(["--headless", "-v", "--weight", "1"]) == 0
    captured = capsys.readouterr()
    assert captured.out.count("fired in wave 0.") == 4  # half of the 8-column bottom row
    assert "80 of 80 neurons fired" in captured.err  # default 8 x 10
    assert "input permutation:" in captured.err


def test_cli_input_option_sets_the_pattern(capsys):
    assert cli_main(["--headless", "-v", "--columns", "6", "--rows", "3", "--weight", "1", "--input", "110", "--no-permute"]) == 0
    captured = capsys.readouterr()
    assert "epoch 1: input 110 -> coded 110001 -> bottom row 110001" in captured.out
    assert captured.out.count("fired in wave 0.") == 3
    assert "input permutation:" not in captured.err


@pytest.mark.parametrize("bad", [["--input", "10"], ["--input", "1x1"], ["--columns", "5", "--rows", "3"]])
def test_cli_rejects_bad_input(bad, capsys):
    args = ["--headless", "--columns", "6", "--rows", "3"] + bad if "--columns" not in bad else ["--headless"] + bad
    assert cli_main(args) == 2
    assert "error:" in capsys.readouterr().err


def test_cli_defaults_to_random_weights_and_reports_the_seed(capsys):
    assert cli_main(["--headless", "--columns", "6", "--rows", "6"]) == 0
    err = capsys.readouterr().err
    assert "seed " in err and "of 36 neurons fired" in err


def test_cli_seed_makes_runs_repeatable(capsys):
    cli_main(["--headless", "--columns", "10", "--rows", "8", "--seed", "11", "--no-save"])  # no timestamped path in the output
    first = capsys.readouterr()
    cli_main(["--headless", "--columns", "10", "--rows", "8", "--seed", "11", "--no-save"])
    second = capsys.readouterr()
    assert first.out == second.out and first.err == second.err
    assert "seed 11" in first.err


def test_cli_columns_and_rows_options(capsys):
    assert cli_main(["--headless", "-v", "--columns", "4", "--rows", "3", "--weight", "1"]) == 0
    assert capsys.readouterr().out.count("fired") == 12


def test_cli_rejects_unknown_arguments():
    with pytest.raises(SystemExit) as exc:
        cli_main(["--headless", "0.5"])
    assert exc.value.code == 2


def test_cli_weight_and_threshold_options(capsys):
    args = ["--headless", "-v", "--columns", "6", "--rows", "5", "--weight", "0.2", "--threshold", "1", "--input", "101"]
    assert cli_main(args) == 0
    assert capsys.readouterr().out.count("fired") == 3  # only the input neurons: 0.4 max input < 1


def test_main_passes_weight_and_threshold_through(capsys):
    grid = main(columns=4, rows=3, weight=0.25, threshold=0.25, seed=1)
    assert len(grid.fired_neurons()) == 12


def test_cli_omega_option_reports_shortcuts(capsys):
    assert cli_main(["--headless", "--columns", "6", "--rows", "6", "--weight", "1", "--omega", "0.2", "--seed", "1"]) == 0
    err = capsys.readouterr().err
    assert "omega 0.2:" in err and "small-world connections" in err and "seed 1" in err


def test_cli_rejects_omega_out_of_range(capsys):
    assert cli_main(["--headless", "--columns", "4", "--rows", "3", "--omega", "1"]) == 2
    assert "omega" in capsys.readouterr().err


def test_main_passes_omega_through(capsys):
    grid = main(columns=6, rows=6, weight=1.0, omega=0.25, seed=2)
    assert grid.omega == 0.25 and len(grid.small_world_connections()) > 0


def test_run_epoch_verbose_false_prints_nothing_about_the_input(capsys, monkeypatch):
    monkeypatch.setattr(Neuron, "verbose", False)
    grid = main(columns=6, rows=3, weight=1.0, seed=1)
    capsys.readouterr()
    run_epoch(grid, verbose=False)
    assert capsys.readouterr().out == ""


def test_cli_is_silent_by_default_and_verbose_on_request(capsys):
    assert cli_main(["--headless", "--columns", "6", "--rows", "3", "--weight", "1", "--no-save"]) == 0
    out = capsys.readouterr().out
    assert out == ""  # nothing per epoch and nothing per neuron: printing is slower than learning
    assert Neuron.verbose is False  # the process-wide flag is restored once the command finishes
    assert cli_main(["--headless", "-v", "--columns", "6", "--rows", "3", "--weight", "1", "--no-save"]) == 0
    out = capsys.readouterr().out
    assert "fired in wave" in out and "epoch 1: input" in out
    assert cli_main(["--headless", "-v", "-q", "--columns", "6", "--rows", "3", "--weight", "1", "--no-save"]) == 0
    assert capsys.readouterr().out == ""  # --quiet still wins if both are given


def test_cli_learn_runs_epochs_and_reports_accuracy(capsys):
    args = ["--headless", "--columns", "8", "--rows", "4", "--seed", "2", "--quiet", "--target", "all-off",
            "--lr", "0.1", "--epochs", "2000"]
    assert cli_main(args) == 0
    err = capsys.readouterr().err
    assert "learning all-off (perturb, lr 0.1, sigma 0.1, homeostasis 1e-06 toward 0.5 in [-5, 5], unstick 0.001): accuracy" in err
    assert "after 2000 epochs:" in err and "to date over 2,000 epochs" in err
    final = float(err.rsplit("% recent", 1)[0].rsplit(" ", 1)[1])
    assert final > 70  # well above the 50% an untrained output row scores on all-off


def test_cli_epochs_without_learn_just_runs_them(capsys):
    assert cli_main(["--headless", "-v", "--columns", "8", "--rows", "4", "--seed", "1", "--epochs", "5", "--no-learn"]) == 0
    out = capsys.readouterr().out
    assert out.count("epoch ") == 5 and "epoch 5:" in out


def test_cli_rejects_unknown_target():
    with pytest.raises(SystemExit):
        cli_main(["--headless", "--target", "sideways"])


def test_cli_learns_by_default_and_no_learn_switches_it_off(capsys):
    assert cli_main(["--headless", "--columns", "8", "--rows", "4", "-q", "--seed", "1", "--epochs", "3"]) == 0
    assert "learning reversed" in capsys.readouterr().err
    assert cli_main(["--headless", "--columns", "8", "--rows", "4", "-q", "--seed", "1", "--epochs", "3", "--no-learn"]) == 0
    assert "learning" not in capsys.readouterr().err


def test_cli_eligibility_and_sigma_options(capsys):
    args = ["--headless", "--columns", "8", "--rows", "4", "--seed", "1", "-q", "--eligibility", "hebb",
            "--sigma", "0.3", "--lr", "0.02", "--epochs", "20"]
    assert cli_main(args) == 0
    assert "learning reversed (hebb, lr 0.02, sigma 0, homeostasis 1e-06 toward 0.5 in [-5, 5], unstick 0.001)" in capsys.readouterr().err


def test_cli_saves_and_loads_weights(tmp_path, capsys):
    from walnutbutter.persistence import read_checkpoint
    path = tmp_path / "weights.json"
    args = ["--headless", "--columns", "8", "--rows", "4", "--seed", "7", "-q", "--epochs", "200", "--save-weights", str(path)]
    assert cli_main(args) == 0
    err = capsys.readouterr().err
    assert f"saved weights to {path}" in err
    data = read_checkpoint(path)
    assert data["epoch"] == 200 and data["seed"] == 7 and data["learning"]["epochs"] == 200

    # resume: the mesh comes from the file, the epoch count carries on, and settings on the
    # command line that describe the mesh are overridden by the checkpoint
    args = ["--headless", "--columns", "3", "--rows", "3", "-q", "--epochs", "50",
            "--load-weights", str(path), "--save-weights", str(path)]
    assert cli_main(args) == 0
    err = capsys.readouterr().err
    assert "loaded" in err and "8x4, seed 7, 200 epochs so far" in err
    assert "to date over 250 epochs" in err
    assert read_checkpoint(path)["epoch"] == 250


def test_cli_reports_a_bad_checkpoint(tmp_path, capsys):
    path = tmp_path / "nope.json"
    assert cli_main(["--headless", "--load-weights", str(path)]) == 2
    assert "cannot load" in capsys.readouterr().err


def test_cli_positive_weights_option(capsys):
    args = ["--headless", "--columns", "8", "--rows", "4", "--seed", "1", "-q", "--epochs", "5",
            "--positive_weights", "--epsilon", "0.01", "--threshold", "2"]
    assert cli_main(args) == 0
    assert "positive weights: every weight kept between 0.01 and 1" in capsys.readouterr().err


def test_cli_positive_weights_rejects_bad_epsilon(capsys):
    assert cli_main(["--headless", "--columns", "8", "--rows", "4", "--positive-weights", "--epsilon", "0"]) == 2
    assert "epsilon" in capsys.readouterr().err


def test_cli_load_restores_positive_weights(tmp_path, capsys):
    path = tmp_path / "pos.json"
    base = ["--headless", "--columns", "8", "--rows", "4", "--seed", "3", "-q", "--epochs", "5"]
    assert cli_main(base + ["--positive-weights", "--epsilon", "0.05", "--save-weights", str(path)]) == 0
    capsys.readouterr()
    assert cli_main(["--headless", "-q", "--epochs", "5", "--load-weights", str(path)]) == 0
    assert "positive weights: every weight kept between 0.05 and 1" in capsys.readouterr().err


def test_cli_homeostasis_options(capsys):
    args = ["--headless", "--columns", "8", "--rows", "4", "--seed", "1", "-q", "--epochs", "20",
            "--homeostasis", "0.01", "--target-rate", "0.3"]
    assert cli_main(args) == 0
    err = capsys.readouterr().err
    assert "homeostasis 0.01 toward 0.3 in [-5, 5]" in err and "stuck" not in err


def test_cli_threshold_range_option(capsys):
    args = ["--headless", "--columns", "8", "--rows", "4", "--seed", "1", "-q", "--epochs", "5",
            "--threshold-range", "0", "3"]
    assert cli_main(args) == 0
    assert "in [0, 3]" in capsys.readouterr().err


def test_run_epoch_discharges_by_default_and_can_carry_over(capsys):
    import random
    grid = GridOfNeurons(columns=6, rows=4, weight=None, seed=1, omega=0)
    run_epoch(grid, verbose=False)
    before = {n: n.potential for n in grid.neurons.values() if not n.has_fired and n.potential != 0}
    assert before  # some neurons received input without firing
    run_epoch(grid, verbose=False, discharge=False, noise=0.1, rng=random.Random(5))
    assert any(n.fired_in_wave is None and n.potential != n.noise for n in before)  # old charge carried
    run_epoch(grid, verbose=False)  # the default clears everything before the input arrives
    grid.reset()
    assert all(n.potential == 0.0 for n in grid.neurons.values())


def test_cli_carry_over_option(capsys):
    assert cli_main(["--headless", "--columns", "8", "--rows", "4", "--seed", "1", "-q", "--epochs", "5", "--carry-over"]) == 0
    assert ", carry-over)" in capsys.readouterr().err
    assert cli_main(["--headless", "--columns", "8", "--rows", "4", "--seed", "1", "-q", "--epochs", "5"]) == 0
    assert "carry-over" not in capsys.readouterr().err


def test_noise_respects_the_minimum_potential(capsys):
    import random
    grid = GridOfNeurons(columns=6, rows=4, weight=None, seed=1, minimum_potential=-0.05)
    for _ in range(20):
        run_epoch(grid, verbose=False, noise=0.5, rng=random.Random(1))
    assert all(n.potential >= -0.05 for n in grid.neurons.values())


def test_cli_minimum_potential_option(capsys):
    assert cli_main(["--headless", "--columns", "8", "--rows", "4", "--seed", "1", "-q", "--epochs", "3",
                     "--minimum-potential", "-0.5"]) == 0
    assert cli_main(["--headless", "--columns", "8", "--rows", "4", "--minimum-potential", "0.5"]) == 2
    assert "must be below the threshold" in capsys.readouterr().err


def test_cli_unstick_options(capsys):
    args = ["--headless", "--columns", "8", "--rows", "4", "--seed", "1", "-q", "--epochs", "5",
            "--unstick", "0.02", "--unstick-target", "0.4"]
    assert cli_main(args) == 0
    assert "unstick 0.02" in capsys.readouterr().err
    assert cli_main(["--headless", "--columns", "8", "--rows", "4", "--seed", "1", "-q", "--epochs", "5", "--unstick", "0"]) == 0
    assert "unstick" not in capsys.readouterr().err


def test_headless_run_records_history_and_default_report_is_one_second(tmp_path, capsys):
    from walnutbutter.cli import build_parser
    from walnutbutter.persistence import read_checkpoint
    assert build_parser().parse_args([]).report == 1.0
    path = tmp_path / "w.json"
    args = ["--headless", "--columns", "8", "--rows", "4", "--seed", "1", "-q", "--epochs", "50", "--save-weights", str(path)]
    assert cli_main(args) == 0
    history = read_checkpoint(path)["learning"]["history"]
    assert len(history) == 10 and history[-1]["epoch"] == 50 and history[0]["epoch"] == 5


def test_checkpoints_are_written_by_default_to_a_timestamped_file(tmp_path, capsys):
    from pathlib import Path
    from walnutbutter.persistence import read_checkpoint
    assert cli_main(["--headless", "--columns", "8", "--rows", "4", "--seed", "7", "-q", "--epochs", "20"]) == 0
    err = capsys.readouterr().err
    files = list(Path("runs").glob("*-seed7.json"))
    assert len(files) == 1 and f"checkpointing to {files[0]}" in err and f"saved weights to {files[0]}" in err
    data = read_checkpoint(files[0])
    assert data["epoch"] == 20 and data["seed"] == 7 and len(data["learning"]["history"]) > 0


def test_no_save_writes_nothing_and_explicit_path_wins(tmp_path, capsys):
    from pathlib import Path
    assert cli_main(["--headless", "--columns", "8", "--rows", "4", "--seed", "7", "-q", "--epochs", "5", "--no-save"]) == 0
    assert not Path("runs").exists() and "checkpointing" not in capsys.readouterr().err
    explicit = tmp_path / "mine.json"
    assert cli_main(["--headless", "--columns", "8", "--rows", "4", "--seed", "7", "-q", "--epochs", "5",
                     "--save-weights", str(explicit)]) == 0
    assert explicit.exists() and not Path("runs").exists()


def test_resumed_run_gets_its_own_file(tmp_path, capsys):
    from pathlib import Path
    first = tmp_path / "first.json"
    assert cli_main(["--headless", "--columns", "8", "--rows", "4", "--seed", "7", "-q", "--epochs", "5",
                     "--save-weights", str(first)]) == 0
    before = first.read_text()
    assert cli_main(["--headless", "-q", "--epochs", "5", "--load-weights", str(first)]) == 0
    assert first.read_text() == before  # the loaded file is untouched
    assert len(list(Path("runs").glob("*-seed7.json"))) == 1


def test_seeds_runs_in_parallel_and_reports_a_sorted_table(tmp_path, capsys):
    from pathlib import Path
    from walnutbutter.persistence import read_checkpoint
    args = ["--seeds", "3", "--seed", "10", "--columns", "8", "--rows", "4", "--epochs", "40"]
    assert cli_main(args) == 0
    captured = capsys.readouterr()
    lines = [l for l in captured.out.splitlines() if l.strip() and not l.startswith(" " * 6 + "seed")]
    rows = [l.split() for l in lines[1:]]
    assert [int(r[0]) for r in rows] and sorted(int(r[0]) for r in rows) == [10, 11, 12]
    recents = [float(r[1].rstrip("%")) for r in rows]
    assert recents == sorted(recents, reverse=True)
    assert "3 seeds from 10 on" in captured.err and "best seed" in captured.err
    files = sorted(Path("runs").glob("*-seed1?.json"))
    assert len(files) == 3 and all(read_checkpoint(f)["epoch"] == 40 for f in files)


def test_seeds_rejects_bad_arguments(capsys):
    assert cli_main(["--seeds", "0", "--epochs", "10"]) == 2
    assert cli_main(["--seeds", "2", "--epochs", "1"]) == 2
    assert cli_main(["--seeds", "2", "--epochs", "10", "--no-learn"]) == 2


def test_seeds_with_no_save_writes_nothing(capsys):
    from pathlib import Path
    assert cli_main(["--seeds", "2", "--seed", "1", "--columns", "8", "--rows", "4", "--epochs", "10", "--no-save"]) == 0
    assert not Path("runs").exists()
    assert "  -" in capsys.readouterr().out


def test_seeds_runs_the_lattice_when_nodes_is_given(tmp_path, capsys):
    from pathlib import Path
    from walnutbutter.cartesian import CartesianNodes
    from walnutbutter.persistence import restore
    assert cli_main(["--nodes", "--seeds", "2", "--seed", "20", "--columns", "8", "--rows", "4", "--epochs", "30"]) == 0
    captured = capsys.readouterr()
    assert "lattice, reach 2" in captured.err
    files = sorted(Path("runs").glob("*-seed2?.json"))
    assert len(files) == 2
    for f in files:
        restored, data = restore(f)
        assert isinstance(restored, CartesianNodes) and data["epoch"] == 30 and data["reach"] == 2.0
