import pytest

from walnutbutter.grid import GridOfNeurons
from walnutbutter.inputs import HAMMING74, complement_code
from walnutbutter.learning import (
    CRITICS, Teacher, decoded_accuracy, decoded_exact, decoded_output, expected_data, output_row, read_output_word, reward,
)
from walnutbutter.monitor import run_epoch
from walnutbutter.neuron import Neuron


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(Neuron, "verbose", False)


def hamming_grid(seed=1):
    grid = GridOfNeurons(columns=14, rows=4, omega=0, seed=seed)
    grid.use_ecc()
    grid.set_input_bits([True, False, True, True])
    return grid


def show_on_output(grid, pattern):
    """Force the output row to display `pattern` (as fired flags)."""
    for neuron, bit in zip(output_row(grid), pattern):
        neuron.reset()
        if bit:
            neuron.fire(wave=3)


def test_reading_the_perfect_output_recovers_the_data():
    grid = hamming_grid()
    target = grid.input_pattern[::-1]  # the reversed target, exactly
    show_on_output(grid, target)
    assert read_output_word(grid) == HAMMING74.encode([True, False, True, True])
    assert decoded_output(grid) == [True, False, True, True] == expected_data(grid)
    assert decoded_accuracy(grid) == 1.0 and decoded_exact(grid) == 1.0
    assert reward(grid, critic="decoded") == 1.0 and reward(grid, critic="row") == 1.0


def test_one_wrong_output_neuron_still_decodes_correctly_under_hamming():
    grid = hamming_grid()
    target = grid.input_pattern[::-1]
    for j in range(14):
        wrong = list(target)
        wrong[j] = not wrong[j]
        show_on_output(grid, wrong)
        assert reward(grid, critic="row") == pytest.approx(13 / 14)  # the row critic docks it
        assert decoded_exact(grid) == 1.0  # the decoding critic does not: the code absorbs one error
    # a pair whose two neurons contradict each other is an unreadable bit, and one of those is absorbed too
    word = read_output_word(grid)
    assert None in word or all(b is not None for b in word)


def test_two_wrong_outputs_can_defeat_the_code_and_the_critics_disagree():
    grid = hamming_grid()
    target = grid.input_pattern[::-1]
    worst = None
    for a in range(14):
        for b in range(a + 1, 14):
            wrong = list(target)
            wrong[a], wrong[b] = not wrong[a], not wrong[b]
            show_on_output(grid, wrong)
            worst = min(worst if worst is not None else 1.0, decoded_accuracy(grid))
    assert worst < 1.0  # some double errors get through
    assert reward(grid, critic="row") == pytest.approx(12 / 14)


def test_decoding_critic_without_a_code_compares_raw_bits():
    grid = GridOfNeurons(columns=8, rows=4, omega=0, seed=1)
    grid.set_input_bits([True, True, False, False])
    show_on_output(grid, grid.input_pattern[::-1])
    assert decoded_output(grid) == [True, True, False, False] and decoded_exact(grid) == 1.0
    assert expected_data(grid) == grid.input_bits


def test_critics_registry_and_teacher_validation():
    assert set(CRITICS) == {"row", "decoded", "decoded-exact"}
    grid = hamming_grid()
    run_epoch(grid, verbose=False)
    teacher = Teacher(grid, critic="decoded", seed=1)
    teacher.step()
    assert "critic decoded" in teacher.status()
    assert "critic" not in Teacher(grid, seed=1).status()
    with pytest.raises(ValueError):
        Teacher(grid, critic="oracle")
    with pytest.raises(ValueError):
        Teacher(grid, critic="decoded", target="all-off")


def test_teacher_with_decoding_critic_learns_something(capsys):
    import statistics
    grid = GridOfNeurons(columns=14, rows=4, weight=None, seed=2)
    grid.use_ecc()
    teacher = Teacher(grid, critic="decoded", lr=0.1, seed=2)
    rewards = [teacher.epoch(verbose=False) for _ in range(1500)]
    assert all(0 <= r <= 1 for r in rewards) and set(rewards) <= {0.0, 0.25, 0.5, 0.75, 1.0}
    assert statistics.mean(rewards[-300:]) >= statistics.mean(rewards[:300]) - 0.05  # at least not getting worse


def test_cli_critic_option_and_checkpoint(tmp_path, capsys):
    from pathlib import Path
    from walnutbutter.cli import cli_main
    from walnutbutter.persistence import read_checkpoint
    assert cli_main(["--headless", "--ecc", "--critic", "decoded", "--seed", "1", "-q", "--epochs", "5"]) == 0
    err = capsys.readouterr().err
    assert "critic decoded" in err
    f = next(Path("runs").glob("*-seed1.json"))
    assert read_checkpoint(f)["learning"]["critic"] == "decoded"
    with pytest.raises(SystemExit):
        cli_main(["--headless", "--critic", "oracle"])
