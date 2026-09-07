import pytest

from walnutbutter.inputs import complement_code, format_bits, parse_bits, random_bits


def test_random_bits_has_the_right_length_and_is_reproducible():
    a = random_bits(12, seed=1)
    assert len(a) == 12 and all(isinstance(b, bool) for b in a)
    assert a == random_bits(12, seed=1)
    assert a != random_bits(12, seed=2)


def test_random_bits_are_not_all_the_same_over_many_draws():
    assert 0 < sum(random_bits(200, seed=3)) < 200


def test_complement_code_appends_the_negations():
    assert complement_code([True, False, False]) == [True, False, False, False, True, True]
    assert complement_code([]) == []


def test_complement_code_always_sets_exactly_half_the_bits():
    for seed in range(5):
        coded = complement_code(random_bits(12, seed))
        assert len(coded) == 24 and sum(coded) == 12


def test_parse_and_format_bits_round_trip():
    assert parse_bits("1011 0010") == [True, False, True, True, False, False, True, False]
    assert format_bits(parse_bits("101100")) == "101100"


@pytest.mark.parametrize("bad", ["", "10x1", "2", "  "])
def test_parse_bits_rejects_non_binary_text(bad):
    with pytest.raises(ValueError):
        parse_bits(bad)
