"""Walk a long film in overlapping windows and commit plays in order."""

import shutil
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from film_analysis.agents.segments import FilmSpan

WINDOW = 40.0
STEP = 30.0
_WORKERS = 4

Frames = Callable[[Path, float, float], list[tuple[float, bytes]]]
Labeler = Callable[[list[tuple[float, bytes]]], tuple[FilmSpan, ...]]
Cutter = Callable[[Path, float, float, Path], None]
PlaySink = Callable[[int, float, float, Path], None]
SpanSink = Callable[[str, float, float], None]
Progress = Callable[[float], None]


def window_bounds(duration: float) -> list[tuple[float, float]]:
    """Return `[start, end)` windows that cover `duration`, overlapping by ten seconds."""

    if duration <= 0:
        return []
    bounds: list[tuple[float, float]] = []
    start = 0.0
    while True:
        end = min(duration, start + WINDOW)
        bounds.append((start, end))
        if end >= duration - 0.05:
            break
        start += STEP
    return bounds


def stable_through(finished: int, bounds: list[tuple[float, float]], duration: float) -> float:
    """Return how far the film is settled once the first `finished` windows have answered."""

    if finished <= 0 or not bounds:
        return 0.0
    if finished >= len(bounds):
        return duration
    return bounds[finished][0]


def merge_plays(spans: list[FilmSpan]) -> list[tuple[float, float]]:
    """Collapse overlapping play labels into one range each."""

    ordered = sorted(
        (
            span
            for span in spans
            if span.kind == "play" and span.end_seconds > span.start_seconds
        ),
        key=lambda span: (span.start_seconds, span.end_seconds),
    )
    merged: list[tuple[float, float]] = []
    for span in ordered:
        start = span.start_seconds
        end = span.end_seconds
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            continue
        merged.append((start, end))
    return merged


def ready_plays(spans: list[FilmSpan], stable: float) -> list[tuple[float, float]]:
    """Plays whose end is inside the settled part of the film."""

    return [(start, end) for start, end in merge_plays(spans) if end <= stable + 0.05]


def ready_non_plays(spans: list[FilmSpan], stable: float) -> list[FilmSpan]:
    """Non-play labels that sit fully inside the settled part and off every play."""

    plays = merge_plays(spans)
    kept: list[FilmSpan] = []
    for span in spans:
        if span.kind == "play" or span.end_seconds > stable + 0.05:
            continue
        if span.end_seconds <= span.start_seconds:
            continue
        if _overlaps_play(span.start_seconds, span.end_seconds, plays):
            continue
        kept.append(span)
    kept.sort(key=lambda span: (span.start_seconds, span.end_seconds, span.kind))
    return kept


def scan_film(
    path: Path,
    duration: float,
    label: Labeler,
    frames: Frames,
    cut: Cutter,
    publish_play: PlaySink,
    publish_span: SpanSink,
    progress: Progress,
    workers: int = _WORKERS,
) -> None:
    """Label the film from start to end. Publish each play once its end has settled."""

    bounds = window_bounds(duration)
    if not bounds:
        progress(duration)
        return

    published_plays = 0
    published_spans: set[tuple[str, float, float]] = set()
    collected: dict[int, tuple[FilmSpan, ...]] = {}
    cuts = Path(tempfile.mkdtemp(prefix="film-cuts-"))
    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            jobs = [
                pool.submit(_label_window, path, start, end, frames, label)
                for start, end in bounds
            ]
            for index, job in enumerate(jobs):
                collected[index] = job.result()
                stable = stable_through(index + 1, bounds, duration)
                spans = [span for group in collected.values() for span in group]
                for start, end in ready_plays(spans, stable)[published_plays:]:
                    published_plays += 1
                    dest = cuts / f"{published_plays}.mp4"
                    cut(path, start, end, dest)
                    publish_play(published_plays, start, end, dest)
                    dest.unlink(missing_ok=True)
                for span in ready_non_plays(spans, stable):
                    key = (span.kind, round(span.start_seconds, 3), round(span.end_seconds, 3))
                    if key in published_spans:
                        continue
                    published_spans.add(key)
                    publish_span(span.kind, span.start_seconds, span.end_seconds)
                progress(stable)
    finally:
        shutil.rmtree(cuts, ignore_errors=True)


def _label_window(
    path: Path,
    start: float,
    end: float,
    frames: Frames,
    label: Labeler,
) -> tuple[FilmSpan, ...]:
    samples = frames(path, start, max(0.1, end - start))
    if not samples:
        return ()
    return tuple(label(samples))


def _overlaps_play(start: float, end: float, plays: list[tuple[float, float]]) -> bool:
    return any(start < play_end and end > play_start for play_start, play_end in plays)
