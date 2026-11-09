"""Counts and rates an agent can cite without re-deriving the math."""

from collections.abc import Sequence

from film_analysis.domain.models import Play, PlayType
from film_analysis.domain.situations import is_successful_play

_SCRIMMAGE = {PlayType.RUN, PlayType.PASS}


def count_by_personnel(plays: Sequence[Play]) -> dict[str, int]:
    """Count snaps by personnel label. Plays without personnel are skipped."""

    counts: dict[str, int] = {}
    for play in plays:
        if play.personnel is None:
            continue
        counts[play.personnel.label] = counts.get(play.personnel.label, 0) + 1
    return dict(sorted(counts.items()))


def success_rate(plays: Sequence[Play]) -> float | None:
    """Share of run and pass plays that meet the success-rate threshold.

    Returns None when no play has a down, a distance of at least 1, a gain, and a
    run or pass type. Kneels, spikes, and kicks are left out.
    """

    graded: list[tuple[int, int, int]] = []
    for play in plays:
        if play.play_type not in _SCRIMMAGE:
            continue
        if play.down is None or play.distance is None or play.yards_gained is None:
            continue
        if play.distance < 1:
            continue
        graded.append((play.down, play.distance, play.yards_gained))

    if not graded:
        return None

    successes = sum(
        is_successful_play(down, distance, yards_gained) for down, distance, yards_gained in graded
    )
    return successes / len(graded)
