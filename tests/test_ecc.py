import itertools
import json

import pytest

from walnutbutter.cartesian import CartesianNodes
from walnutbutter.grid import GridOfNeurons
from walnutbutter.inputs import CODES, DEFAULT_CODE, HAMMING74, PARITY64, complement_code, format_bits
from walnutbutter.monitor import run_epoch
from walnutbutter.neuron import Neuron


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(Neuron, "verbose", False)


def words_of(code):
    return [code.encode(d) for d in itertools.product([False, True], repeat=code.data_bits)]


def flips(word):
    return [word[:i] + [not word[i]] + word[i + 1:] for i in range(len(word))]


def test_hamming_is_the_default_and_the_codes_are_systematic():
    assert DEFAULT_CODE == "hamming74" and set(CODES) == {"hamming74", "parity64"}
    assert HAMMING74.encode([True, False, True, True]) == [True, False, True, True, False, True, False]
    assert PARITY64.encode([True, False, True, True]) == [True, False, True, True, False, False]
    for code in CODES.values():
        for data in itertools.product([False, True], repeat=4):
            word = code.encode(data)
            assert word[:4] == list(data) and code.syndrome(word) == (0,) * len(code.covers)
        with pytest.raises(ValueError):
            code.encode([True] * 5)
        with pytest.raises(ValueError):
            code.syndrome([True] * 3)


def test_hamming_corrects_every_single_flip_and_parity_only_detects():
    ham = words_of(HAMMING74)
    assert min(sum(a != b for a, b in zip(u, v)) for u in ham for v in ham if u != v) == 3
    assert HAMMING74.corrects_single_errors
    for word in ham:
        for i, corrupted in enumerate(flips(word)):
            fixed, position = HAMMING74.correct(corrupted)
            assert fixed == word and position == i
            assert HAMMING74.decode(corrupted) == word[:4]
    par = words_of(PARITY64)
    assert min(sum(a != b for a, b in zip(u, v)) for u in par for v in par if u != v) == 2
    assert not PARITY64.corrects_single_errors
    for word in par:
        for corrupted in flips(word):
            assert any(PARITY64.syndrome(corrupted))  # detected
            assert PARITY64.correct(corrupted) == (corrupted, None)  # but not repaired


def test_two_flips_defeat_hamming_gracefully():
    word = HAMMING74.encode([True, False, False, True])
    twice = word[:]
    twice[0] = not twice[0]
    twice[5] = not twice[5]
    fixed, position = HAMMING74.correct(twice)
    assert fixed != word  # a double error is beyond the code; it corrects to some other word or leaves it
    assert HAMMING74.syndrome(twice) != (0, 0, 0)


def test_network_code_stage_hamming_fills_fourteen_columns():
    grid = GridOfNeurons(columns=14, rows=4, omega=0, permute=False)
    grid.use_ecc()
    assert grid.ecc == "hamming74" and grid.code is HAMMING74 and grid.raw_bit_count() == 4
    grid.set_input_bits([True, False, True, True])
    assert grid.input_data == [True, False, True, True]
    assert grid.input_bits == HAMMING74.encode([True, False, True, True])
    assert grid.input_pattern == complement_code(grid.input_bits) and sum(grid.input_pattern) == 7
    with pytest.raises(ValueError):
        grid.set_input_bits([True] * 7)
    with pytest.raises(ValueError):
        GridOfNeurons(columns=12, rows=4).use_ecc("hamming74")  # needs 14 columns
    with pytest.raises(ValueError):
        GridOfNeurons(columns=14, rows=4).use_ecc("nonsense")
    six = GridOfNeurons(columns=12, rows=4)
    six.use_ecc("parity64")
    assert six.code is PARITY64 and six.raw_bit_count() == 4
    six.use_ecc(False)
    assert six.code is None and six.raw_bit_count() == 6


def test_random_inputs_with_a_code_are_always_valid_codewords(capsys):
    grid = GridOfNeurons(columns=14, rows=4, weight=None, seed=1)
    grid.use_ecc()
    for _ in range(50):
        run_epoch(grid, verbose=False)
        assert HAMMING74.syndrome(grid.input_bits) == (0, 0, 0)
        assert len(grid.input_data) == 4 and sum(grid.input_pattern) == 7
        assert [n.fired_in_wave == 0 for n in grid.input_row()] == grid.input_pattern
    run_epoch(grid)
    out = capsys.readouterr().out
    assert "data " in out and " -> hamming74 " in out and " -> bottom row " in out


def test_lattice_takes_the_code_stage_too(capsys):
    nodes = CartesianNodes(columns=14, rows=10, seed=1)
    nodes.connect_within()
    nodes.use_ecc()
    run_epoch(nodes, verbose=False)
    assert len(nodes.input_row()) == 14 and sum(nodes.input_pattern) == 7 and len(nodes.input_data) == 4


def test_cli_ecc_defaults_to_hamming_on_fourteen_columns_and_checkpoints_the_code(tmp_path, capsys):
    from pathlib import Path
    from walnutbutter.cli import cli_main
    from walnutbutter.persistence import restore
    assert cli_main(["--headless", "-v", "--ecc", "--seed", "1", "--epochs", "3"]) == 0
    err = capsys.readouterr().err
    assert "hamming74 (7, 4) code, corrects single errors -> complement code -> 14 columns" in err
    f = next(Path("runs").glob("*-seed1.json"))
    restored, data = restore(f)
    assert data["ecc"] == "hamming74" and restored.ecc == "hamming74" and restored.columns == 14 and restored.rows == 10
    assert cli_main(["--headless", "-v", "--epochs", "2", "--load-weights", str(f), "--no-save"]) == 0
    assert "hamming74" in capsys.readouterr().err
    assert cli_main(["--headless", "-v", "--ecc", "parity64", "--seed", "1", "--epochs", "2", "--no-save"]) == 0
    assert "parity64 (6, 4) code, detects single errors -> complement code -> 12 columns" in capsys.readouterr().err
    assert cli_main(["--headless", "-v", "--ecc", "--columns", "8", "--rows", "4", "--no-save"]) == 2
    assert cli_main(["--headless", "-v", "--ecc", "--input", "1011", "--seed", "1", "--no-save"]) == 0
    assert "data 1011 -> hamming74 1011010" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli_main(["--headless", "-v", "--ecc", "golay", "--no-save"])


def test_old_checkpoints_with_ecc_true_mean_the_parity_code(tmp_path):
    from walnutbutter.persistence import checkpoint, restore
    grid = GridOfNeurons(columns=12, rows=4, seed=1)
    grid.use_ecc("parity64")
    path = tmp_path / "old.json"
    data = checkpoint(grid, path)
    data["ecc"] = True
    path.write_text(json.dumps(data))
    restored, _ = restore(path)
    assert restored.ecc == "parity64"


def test_cli_ecc_on_the_lattice_and_in_a_seed_batch(tmp_path, capsys):
    from pathlib import Path
    from walnutbutter.cli import cli_main
    from walnutbutter.persistence import read_checkpoint
    assert cli_main(["--headless", "--ecc", "--nodes", "--seed", "2", "-q", "--epochs", "3"]) == 0
    assert "14x10 hexagonal lattice" in capsys.readouterr().err
    assert cli_main(["--ecc", "--seeds", "2", "--seed", "30", "--rows", "4", "--epochs", "10"]) == 0
    for f in Path("runs").glob("*-seed3?.json"):
        assert read_checkpoint(f)["ecc"] == "hamming74"
