import pytest

from aftertrace.world.seeds import SEED_RANGES, assert_phase


def test_phase_seed_ranges_are_disjoint() -> None:
    ranges = tuple(SEED_RANGES.values())
    for index, left in enumerate(ranges):
        for right in ranges[index + 1 :]:
            assert left.stop <= right.start or right.stop <= left.start


def test_assert_phase_accepts_only_the_matching_range() -> None:
    assert_phase(0, "reader_train")
    assert_phase(1_000_000, "policy_tune")
    assert_phase(2_000_000, "eval")
    with pytest.raises(ValueError):
        assert_phase(2_000_000, "reader_train")
    with pytest.raises(ValueError):
        assert_phase(0, "unknown")

