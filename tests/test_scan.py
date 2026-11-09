import threading
from pathlib import Path

import pytest

from film_analysis.agents.scan import (
    MAX_WORKERS,
    WorkerLimit,
    family_at,
    merge_plays,
    open_window,
    ready_non_plays,
    ready_plays,
    resolve_held,
    review_ranges,
    scan_film,
    stable_through,
    window_bounds,
)
from film_analysis.agents.segments import FilmSpan


def test_windows_cover_a_long_film_with_overlap() -> None:
    assert window_bounds(100) == [(0.0, 40.0), (30.0, 70.0), (60.0, 100.0)]
    assert stable_through(1, window_bounds(100), 100) == 30.0
    assert stable_through(3, window_bounds(100), 100) == 100.0


def test_a_play_waits_until_its_end_has_settled() -> None:
    spans = [
        FilmSpan(kind="play", start_seconds=28, end_seconds=36),
        FilmSpan(kind="huddle", start_seconds=0, end_seconds=10),
        FilmSpan(kind="replay", start_seconds=36, end_seconds=50),
    ]

    assert ready_plays(spans, 30) == []
    assert ready_plays(spans, 60) == [(28, 36)]
    assert [span.kind for span in ready_non_plays(spans, 30)] == ["huddle"]
    assert [span.kind for span in ready_non_plays(spans, 60)] == ["huddle", "replay"]


def test_overlapping_play_labels_collapse() -> None:
    spans = [
        FilmSpan(kind="play", start_seconds=28, end_seconds=36),
        FilmSpan(kind="play", start_seconds=30, end_seconds=36),
        FilmSpan(kind="dead", start_seconds=32, end_seconds=34),
    ]

    assert merge_plays(spans) == [(28, 36)]
    assert ready_non_plays(spans, 60) == []


def test_scrimmage_and_special_teams_are_plays_and_junk_is_not() -> None:
    spans = [
        FilmSpan(kind="scrimmage", start_seconds=1, end_seconds=4),
        FilmSpan(kind="special_teams", start_seconds=10, end_seconds=16),
        FilmSpan(kind="junk", start_seconds=4, end_seconds=10),
        FilmSpan(kind="uncertain", start_seconds=20, end_seconds=28),
    ]

    assert merge_plays(spans) == [(1, 4), (10, 16)]
    assert [span.kind for span in ready_non_plays(spans, 30)] == ["junk"]
    assert review_ranges(spans) == [(20, 28)]
    assert family_at(spans, 10, 16) == "special_teams"
    assert family_at(spans, 1, 4) == "scrimmage"


def test_a_sure_label_keeps_an_unsure_stretch_off_the_review_pile() -> None:
    covered = [
        FilmSpan(kind="uncertain", start_seconds=0, end_seconds=30),
        FilmSpan(kind="scrimmage", start_seconds=10, end_seconds=20),
        FilmSpan(kind="junk", start_seconds=20, end_seconds=30),
    ]
    sliver = [
        FilmSpan(kind="uncertain", start_seconds=0, end_seconds=10),
        FilmSpan(kind="junk", start_seconds=0, end_seconds=9),
    ]

    assert review_ranges(covered) == [(0, 10)]
    assert review_ranges(sliver) == []


def test_a_second_look_can_keep_a_play_or_leave_it() -> None:
    held = [(1.0, 6.0), (10.0, 16.0)]
    decisions = [
        (FilmSpan(kind="special_teams", start_seconds=1, end_seconds=6),),
        (FilmSpan(kind="uncertain", start_seconds=10, end_seconds=16),),
    ]

    plays, leftovers = resolve_held(held, decisions, [])

    assert [(play.start, play.end, play.family) for play in plays] == [(1.0, 6.0, "special_teams")]
    assert [(span.kind, span.start_seconds, span.end_seconds) for span in leftovers] == [
        ("uncertain", 10.0, 16.0)
    ]


def test_unsure_footage_is_reviewed_after_the_film() -> None:
    def frames(path: Path, start: float, seconds: float) -> list[tuple[float, bytes]]:
        del path, seconds
        return [(start, b"jpeg")]

    def label(samples: list[tuple[float, bytes]]) -> tuple[FilmSpan, ...]:
        start = samples[0][0]
        if start < 1:
            return (FilmSpan(kind="uncertain", start_seconds=0, end_seconds=8),)
        return (FilmSpan(kind="uncertain", start_seconds=30, end_seconds=40),)

    def review(samples: list[tuple[float, bytes]]) -> tuple[FilmSpan, ...]:
        start = samples[0][0]
        if start < 1:
            return (FilmSpan(kind="scrimmage", start_seconds=0, end_seconds=8),)
        return (FilmSpan(kind="junk", start_seconds=30, end_seconds=40),)

    plays: list[tuple[float, str]] = []
    spans: list[tuple[str, float]] = []

    def cut(source: Path, start: float, end: float, dest: Path) -> None:
        del source, start, end
        dest.write_bytes(b"clip")

    scan_film(
        Path("film.mp4"),
        70,
        label,
        frames,
        cut,
        lambda number, start, end, clip, family: plays.append((start, family)),
        lambda kind, start, end: spans.append((kind, start)),
        lambda seconds: None,
        review=review,
    )

    assert plays == [(0.0, "scrimmage")]
    assert spans == [("junk", 30.0)]


def test_windows_run_at_the_same_time() -> None:
    started = threading.Event()
    seen: list[float] = []
    lock = threading.Lock()

    def frames(path: Path, start: float, seconds: float) -> list[tuple[float, bytes]]:
        del path, seconds
        with lock:
            seen.append(start)
            opened = len(seen) >= 2
            if opened:
                started.set()
        if not opened:
            assert started.wait(timeout=2)
        return [(start, b"jpeg")]

    def label(samples: list[tuple[float, bytes]]) -> tuple[FilmSpan, ...]:
        del samples
        return ()

    scan_film(
        Path("film.mp4"),
        70,
        label,
        frames,
        lambda source, start, end, dest: None,
        lambda number, start, end, clip, family: None,
        lambda kind, start, end: None,
        lambda seconds: None,
        workers=2,
    )

    assert sorted(seen) == [0.0, 30.0]


def test_raising_the_worker_limit_lets_waiting_labelers_start() -> None:
    slots = WorkerLimit(1)
    entered = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    active = 0
    peak = 0

    def frames(path: Path, start: float, seconds: float) -> list[tuple[float, bytes]]:
        nonlocal active, peak
        del path, seconds
        with lock:
            active += 1
            peak = max(peak, active)
            if active >= 1:
                entered.set()
        assert release.wait(timeout=2)
        with lock:
            active -= 1
        return [(start, b"jpeg")]

    def label(samples: list[tuple[float, bytes]]) -> tuple[FilmSpan, ...]:
        del samples
        return ()

    def run() -> None:
        scan_film(
            Path("film.mp4"),
            100,
            label,
            frames,
            lambda source, start, end, dest: None,
            lambda number, start, end, clip, family: None,
            lambda kind, start, end: None,
            lambda seconds: None,
            limit=slots,
        )

    job = threading.Thread(target=run)
    job.start()
    assert entered.wait(timeout=2)
    with lock:
        assert peak == 1
    slots.set(MAX_WORKERS)
    for _ in range(100):
        with lock:
            if peak == 3:
                break
        threading.Event().wait(0.02)
    with lock:
        assert peak == 3
    release.set()
    job.join(timeout=2)
    assert not job.is_alive()


def test_resume_starts_at_the_first_unfinished_window() -> None:
    bounds = window_bounds(100)

    assert open_window(bounds, 0, 100) == 0
    assert open_window(bounds, 20, 100) == 0
    assert open_window(bounds, 30, 100) == 1
    assert open_window(bounds, 60, 100) == 2
    assert open_window(bounds, 100, 100) == 2


def test_a_resumed_scan_keeps_plays_from_the_settled_prefix() -> None:
    seen: list[float] = []

    def frames(path: Path, start: float, seconds: float) -> list[tuple[float, bytes]]:
        del path, seconds
        seen.append(start)
        return [(start, b"jpeg")]

    def label(samples: list[tuple[float, bytes]]) -> tuple[FilmSpan, ...]:
        start = samples[0][0]
        if start < 40:
            return (
                FilmSpan(kind="scrimmage", start_seconds=28, end_seconds=36),
                FilmSpan(kind="scrimmage", start_seconds=50, end_seconds=58),
            )
        return (FilmSpan(kind="scrimmage", start_seconds=62, end_seconds=70),)

    plays: list[tuple[int, float, float]] = []

    def cut(source: Path, start: float, end: float, dest: Path) -> None:
        del source, start, end
        dest.write_bytes(b"clip")

    scan_film(
        Path("film.mp4"),
        100,
        label,
        frames,
        cut,
        lambda number, start, end, clip, family: plays.append((number, start, end)),
        lambda kind, start, end: None,
        lambda seconds: None,
        resume_at=30,
        next_number=2,
        already=[(28, 36)],
    )

    assert sorted(seen) == [30.0, 60.0]
    assert plays == [(2, 50.0, 58.0), (3, 62.0, 70.0)]


def test_a_failed_second_look_is_logged_and_left_unsure(caplog: pytest.LogCaptureFixture) -> None:
    def frames(path: Path, start: float, seconds: float) -> list[tuple[float, bytes]]:
        del path, seconds
        return [(start, b"jpeg")]

    def label(samples: list[tuple[float, bytes]]) -> tuple[FilmSpan, ...]:
        del samples
        return (FilmSpan(kind="uncertain", start_seconds=0, end_seconds=8),)

    def review(samples: list[tuple[float, bytes]]) -> tuple[FilmSpan, ...]:
        del samples
        raise RuntimeError("review broke")

    spans: list[tuple[str, float]] = []

    scan_film(
        Path("film.mp4"),
        8,
        label,
        frames,
        lambda source, start, end, dest: None,
        lambda number, start, end, clip, family: None,
        lambda kind, start, end: spans.append((kind, start)),
        lambda seconds: None,
        review=review,
    )

    assert spans == [("uncertain", 0.0)]
    assert "second look at 0.0–8.0 failed" in caplog.text
    assert "review broke" in caplog.text
