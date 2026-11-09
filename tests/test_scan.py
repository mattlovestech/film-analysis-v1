import threading
from pathlib import Path

from film_analysis.agents.scan import (
    merge_plays,
    ready_non_plays,
    ready_plays,
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
        lambda number, start, end, clip: None,
        lambda kind, start, end: None,
        lambda seconds: None,
        workers=2,
    )

    assert sorted(seen) == [0.0, 30.0]
