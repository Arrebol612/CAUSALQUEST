import pytest

from causalquest.data.splits import SplitFractions, assign_group


def test_group_assignment_is_stable():
    key = "open_bandit|men|day1-day2|random-to-bts|ope"
    assert assign_group(key) == assign_group(key)


def test_empty_group_is_rejected():
    with pytest.raises(ValueError):
        assign_group("  ")


def test_split_fractions_must_sum_to_one():
    with pytest.raises(ValueError):
        SplitFractions(train=0.5)
