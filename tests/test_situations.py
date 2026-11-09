import pytest

from film_analysis.domain.situations import (
    DownDistanceBucket,
    bucket_down_and_distance,
    is_goal_to_go,
    is_red_zone,
    is_successful_play,
)


def test_first_and_ten_is_its_own_bucket() -> None:
    assert bucket_down_and_distance(1, 10) is DownDistanceBucket.FIRST_AND_TEN


def test_length_splits() -> None:
    assert bucket_down_and_distance(3, 2) is DownDistanceBucket.THIRD_AND_SHORT
    assert bucket_down_and_distance(2, 6) is DownDistanceBucket.SECOND_AND_MEDIUM
    assert bucket_down_and_distance(1, 15) is DownDistanceBucket.FIRST_AND_LONG
    assert bucket_down_and_distance(4, 1) is DownDistanceBucket.FOURTH_AND_SHORT


def test_red_zone_starts_at_the_twenty() -> None:
    assert is_red_zone(20) is True
    assert is_red_zone(21) is False


def test_goal_to_go_when_the_sticks_are_the_goal_line() -> None:
    assert is_goal_to_go(8, 8) is True
    assert is_goal_to_go(7, 8) is False


def test_success_rate_thresholds() -> None:
    assert is_successful_play(1, 10, 4) is True
    assert is_successful_play(1, 10, 3) is False
    assert is_successful_play(3, 8, 8) is True
    assert is_successful_play(3, 8, 7) is False
    assert is_successful_play(1, 3, 2) is True
    assert is_successful_play(1, 3, 1) is False


def test_success_rate_rejects_a_missing_distance() -> None:
    with pytest.raises(ValueError):
        is_successful_play(1, 0, 0)
