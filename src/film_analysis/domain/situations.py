"""Down-and-distance and field-position situations used when charting film."""

from enum import StrEnum


class DownDistanceBucket(StrEnum):
    """Coarse down-and-distance call, in the language used on a cutup."""

    FIRST_AND_TEN = "1st & 10"
    FIRST_AND_SHORT = "1st & short"
    FIRST_AND_MEDIUM = "1st & medium"
    FIRST_AND_LONG = "1st & long"
    SECOND_AND_SHORT = "2nd & short"
    SECOND_AND_MEDIUM = "2nd & medium"
    SECOND_AND_LONG = "2nd & long"
    THIRD_AND_SHORT = "3rd & short"
    THIRD_AND_MEDIUM = "3rd & medium"
    THIRD_AND_LONG = "3rd & long"
    FOURTH_AND_SHORT = "4th & short"
    FOURTH_AND_MEDIUM = "4th & medium"
    FOURTH_AND_LONG = "4th & long"


_DOWN_PREFIX = {1: "FIRST", 2: "SECOND", 3: "THIRD", 4: "FOURTH"}

# Football Outsiders success rate: 40% of yards to go on 1st, 60% on 2nd,
# and 100% on 3rd or 4th. Stored as fifths so the comparison stays in integers.
_SUCCESS_FIFTHS = {1: 2, 2: 3, 3: 5, 4: 5}


def bucket_down_and_distance(down: int, distance: int) -> DownDistanceBucket:
    """Group a down and distance into a charting bucket.

    Short is 0–3 yards, medium is 4–6, and long is 7 or more.
    First-and-10 is its own bucket. Anything else on first down uses the length split.
    """

    if down not in _DOWN_PREFIX:
        raise ValueError(f"down must be 1-4, got {down}")
    if distance < 0:
        raise ValueError(f"distance must be >= 0, got {distance}")
    if down == 1 and distance == 10:
        return DownDistanceBucket.FIRST_AND_TEN

    if distance <= 3:
        length = "SHORT"
    elif distance <= 6:
        length = "MEDIUM"
    else:
        length = "LONG"
    return DownDistanceBucket[f"{_DOWN_PREFIX[down]}_AND_{length}"]


def is_red_zone(yards_to_goal: int) -> bool:
    """Return whether the ball is on or inside the opponent 20."""

    if not 1 <= yards_to_goal <= 99:
        raise ValueError(f"yards_to_goal must be 1-99, got {yards_to_goal}")
    return yards_to_goal <= 20


def is_goal_to_go(distance: int, yards_to_goal: int) -> bool:
    """Return whether a first down would score."""

    if distance < 0:
        raise ValueError(f"distance must be >= 0, got {distance}")
    if not 1 <= yards_to_goal <= 99:
        raise ValueError(f"yards_to_goal must be 1-99, got {yards_to_goal}")
    return distance >= yards_to_goal


def is_successful_play(down: int, distance: int, yards_gained: int) -> bool:
    """Return whether a play meets the standard success-rate threshold.

    A play is successful when it gains at least 40% of yards to go on first down,
    60% on second down, and 100% on third or fourth down.
    """

    if down not in _SUCCESS_FIFTHS:
        raise ValueError(f"down must be 1-4, got {down}")
    if distance < 1:
        raise ValueError(f"distance must be at least 1, got {distance}")
    return yards_gained * 5 >= _SUCCESS_FIFTHS[down] * distance
