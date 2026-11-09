"""Walk a long film in overlapping windows and commit plays in order."""

import logging
import shutil
import tempfile
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

from film_analysis.agents.segments import (
    SCRIMMAGE,
    UNCERTAIN,
    FilmSpan,
    is_play_kind,
    play_family,
)

WINDOW = 40.0
STEP = 30.0
DEFAULT_WORKERS = 4
MAX_WORKERS = 16
_MIN_REVIEW = 2.0
_LOG = logging.getLogger("film_analysis")


class WorkerLimit:
    """How many labelers may read a film at once.

    Raising the count lets labelers that are waiting start, so a job already
    in progress speeds up without being restarted.
    """

    def __init__(self, count: int) -> None:
        self._count = clamp_workers(count)
        self._active = 0
        self._waiting = threading.Condition()

    def get(self) -> int:
        with self._waiting:
            return self._count

    def set(self, count: int) -> int:
        with self._waiting:
            self._count = clamp_workers(count)
            self._waiting.notify_all()
            return self._count

    def __enter__(self) -> None:
        with self._waiting:
            while self._active >= self._count:
                self._waiting.wait()
            self._active += 1

    def __exit__(self, *_args: object) -> None:
        with self._waiting:
            self._active -= 1
            self._waiting.notify_all()


def clamp_workers(count: int) -> int:
    """Keep a worker count between one and the most this process will run."""

    return min(MAX_WORKERS, max(1, count))


class FilmWindowError(Exception):
    """One stretch of film could not be read. `cause` is the original failure."""

    def __init__(self, start: float, end: float, cause: BaseException) -> None:
        self.start = start
        self.end = end
        self.cause = cause
        super().__init__(str(cause))


class PlayCut(NamedTuple):
    """One play ready to cut, with scrimmage or special teams as its family."""

    start: float
    end: float
    family: str

Frames = Callable[[Path, float, float], list[tuple[float, bytes]]]
Labeler = Callable[[list[tuple[float, bytes]]], tuple[FilmSpan, ...]]
Cutter = Callable[[Path, float, float, Path], None]
PlaySink = Callable[[int, float, float, Path, str], None]
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


def open_window(bounds: list[tuple[float, float]], resume_at: float, duration: float) -> int:
    """First window that had not finished when the scan reached `resume_at`."""

    if not bounds or resume_at <= 0:
        return 0
    if resume_at >= duration - 0.05:
        return max(0, len(bounds) - 1)
    for index, _span in enumerate(bounds):
        if resume_at < stable_through(index + 1, bounds, duration) - 0.05:
            return index
    return len(bounds)


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
            if is_play_kind(span.kind) and span.end_seconds > span.start_seconds
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
    """Junk labels that sit fully inside the settled part and off every play.

    Uncertain stretches are held until the film has been read, then reviewed.
    """

    plays = merge_plays(spans)
    kept: list[FilmSpan] = []
    for span in spans:
        if is_play_kind(span.kind) or span.kind == UNCERTAIN or span.end_seconds > stable + 0.05:
            continue
        if span.end_seconds <= span.start_seconds:
            continue
        if _overlaps_play(span.start_seconds, span.end_seconds, plays):
            continue
        kept.append(span)
    kept.sort(key=lambda span: (span.start_seconds, span.end_seconds, span.kind))
    return kept


def family_at(spans: list[FilmSpan], start: float, end: float) -> str:
    """Return special teams only when no scrimmage label covers the play."""

    saw_scrimmage = False
    saw_special = False
    for span in spans:
        if span.end_seconds <= start or span.start_seconds >= end or not is_play_kind(span.kind):
            continue
        if play_family(span.kind) == "special_teams":
            saw_special = True
        else:
            saw_scrimmage = True
    if saw_special and not saw_scrimmage:
        return "special_teams"
    return SCRIMMAGE


def review_ranges(spans: list[FilmSpan]) -> list[tuple[float, float]]:
    """Uncertain stretches that no play and no junk label already covers.

    Anything shorter than two seconds is discarded. A neighboring window can
    still keep it when that window is sure it is a play.
    """

    uncertain = _merge_ranges(
        [
            (span.start_seconds, span.end_seconds)
            for span in spans
            if span.kind == UNCERTAIN and span.end_seconds > span.start_seconds
        ]
    )
    blockers = _merge_ranges(
        [
            (span.start_seconds, span.end_seconds)
            for span in spans
            if span.kind != UNCERTAIN and span.end_seconds > span.start_seconds
        ]
    )
    return [
        (start, end)
        for start, end in _subtract(uncertain, blockers)
        if end - start >= _MIN_REVIEW
    ]


def resolve_held(
    held: list[tuple[float, float]],
    decisions: list[tuple[FilmSpan, ...]],
    published: list[tuple[float, float]],
) -> tuple[list[PlayCut], list[FilmSpan]]:
    """Turn a second look into new plays and whatever is still set aside."""

    plays: list[PlayCut] = []
    kept: list[FilmSpan] = []
    taken = list(published)
    for (start, end), decided in zip(held, decisions, strict=True):
        clipped: list[FilmSpan] = []
        for span in decided:
            piece = _clip_span(span, start, end)
            if piece is not None:
                clipped.append(piece)
        for play_start, play_end in merge_plays(clipped):
            for left, right in _subtract([(play_start, play_end)], taken):
                if right - left < _MIN_REVIEW:
                    continue
                plays.append(PlayCut(left, right, family_at(clipped, left, right)))
                taken.append((left, right))
        for span in clipped:
            if is_play_kind(span.kind) or span.kind == UNCERTAIN:
                continue
            if _overlaps_play(span.start_seconds, span.end_seconds, taken):
                continue
            kept.append(span)
        base = [FilmSpan(kind=UNCERTAIN, start_seconds=start, end_seconds=end)]
        for left, right in review_ranges([*base, *clipped]):
            kept.append(FilmSpan(kind=UNCERTAIN, start_seconds=left, end_seconds=right))
    return plays, kept


def scan_film(
    path: Path,
    duration: float,
    label: Labeler,
    frames: Frames,
    cut: Cutter,
    publish_play: PlaySink,
    publish_span: SpanSink,
    progress: Progress,
    workers: int = DEFAULT_WORKERS,
    limit: WorkerLimit | None = None,
    review: Labeler | None = None,
    resume_at: float = 0.0,
    next_number: int = 1,
    already: list[tuple[float, float]] | None = None,
    on_review: Callable[[], None] | None = None,
) -> None:
    """Label the film from start to end. Publish each play once its end has settled.

    `resume_at` is the settled point from an earlier pass. Windows before that
    stay as they were, and new plays continue from `next_number`.
    """

    slots = limit or WorkerLimit(workers)
    bounds = window_bounds(duration)
    if not bounds:
        progress(duration)
        return

    start_index = open_window(bounds, resume_at, duration)
    kept = list(already or [])
    published_plays = next_number - 1
    published_spans: set[tuple[str, float, float]] = set()
    collected: dict[int, tuple[FilmSpan, ...]] = {}
    cuts = Path(tempfile.mkdtemp(prefix="film-cuts-"))
    try:
        pending = bounds[start_index:]
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            jobs = [
                pool.submit(_label_limited, slots, path, start, end, frames, label)
                for start, end in pending
            ]
            for offset, job in enumerate(jobs):
                index = start_index + offset
                collected[index] = job.result()
                stable = stable_through(index + 1, bounds, duration)
                spans = [span for group in collected.values() for span in group]
                published_plays = _publish_ready(
                    path,
                    cuts,
                    cut,
                    publish_play,
                    spans,
                    stable,
                    kept,
                    resume_at,
                    published_plays,
                )
                _emit_new_spans(
                    [
                        span
                        for span in ready_non_plays(spans, stable)
                        if span.end_seconds > resume_at + 0.05
                    ],
                    published_spans,
                    publish_span,
                )
                progress(stable)
        spans = [span for group in collected.values() for span in group]
        if review_ranges(spans) and on_review is not None:
            on_review()
        _review_held(
            path,
            spans,
            frames,
            review,
            cut,
            publish_play,
            publish_span,
            published_plays,
            published_spans,
            cuts,
            slots,
            kept,
        )
    finally:
        shutil.rmtree(cuts, ignore_errors=True)


def _publish_ready(
    path: Path,
    cuts: Path,
    cut: Cutter,
    publish_play: PlaySink,
    spans: list[FilmSpan],
    stable: float,
    kept: list[tuple[float, float]],
    resume_at: float,
    published_plays: int,
) -> int:
    for start, end in ready_plays(spans, stable):
        if end <= resume_at + 0.05 or _overlaps_play(start, end, kept):
            continue
        published_plays += 1
        _emit_play(
            path,
            cuts,
            cut,
            publish_play,
            published_plays,
            start,
            end,
            family_at(spans, start, end),
        )
        kept.append((start, end))
    return published_plays


def _label_limited(
    slots: WorkerLimit,
    path: Path,
    start: float,
    end: float,
    frames: Frames,
    label: Labeler,
) -> tuple[FilmSpan, ...]:
    with slots:
        return _label_window(path, start, end, frames, label)


def _label_window(
    path: Path,
    start: float,
    end: float,
    frames: Frames,
    label: Labeler,
) -> tuple[FilmSpan, ...]:
    try:
        samples = frames(path, start, max(0.1, end - start))
        if not samples:
            return ()
        return tuple(label(samples))
    except Exception as error:
        raise FilmWindowError(start, end, error) from error


def _review_held(
    path: Path,
    spans: list[FilmSpan],
    frames: Frames,
    review: Labeler | None,
    cut: Cutter,
    publish_play: PlaySink,
    publish_span: SpanSink,
    published_plays: int,
    published_spans: set[tuple[str, float, float]],
    cuts: Path,
    slots: WorkerLimit,
    taken: list[tuple[float, float]],
) -> None:
    """Look again at unsure stretches, then keep whatever is still unsure."""

    held = review_ranges(spans)
    if not held:
        return
    if review is None:
        decisions = [() for _ in held]
    else:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            jobs = [
                pool.submit(_review_limited, slots, path, start, end, frames, review)
                for start, end in held
            ]
            decisions = [job.result() for job in jobs]
    plays, leftovers = resolve_held(held, decisions, taken)
    for play in plays:
        published_plays += 1
        _emit_play(
            path,
            cuts,
            cut,
            publish_play,
            published_plays,
            play.start,
            play.end,
            play.family,
        )
    _emit_new_spans(leftovers, published_spans, publish_span)


def _review_limited(
    slots: WorkerLimit,
    path: Path,
    start: float,
    end: float,
    frames: Frames,
    review: Labeler,
) -> tuple[FilmSpan, ...]:
    with slots:
        return _safe_review(path, start, end, frames, review)


def _safe_review(
    path: Path,
    start: float,
    end: float,
    frames: Frames,
    review: Labeler,
) -> tuple[FilmSpan, ...]:
    try:
        return _label_window(path, start, end, frames, review)
    except Exception as error:
        _LOG.exception("second look at %.1f–%.1f failed: %s", start, end, error)
        return (FilmSpan(kind=UNCERTAIN, start_seconds=start, end_seconds=end),)


def _emit_play(
    path: Path,
    cuts: Path,
    cut: Cutter,
    publish_play: PlaySink,
    number: int,
    start: float,
    end: float,
    family: str,
) -> None:
    dest = cuts / f"{number}.mp4"
    cut(path, start, end, dest)
    publish_play(number, start, end, dest, family)
    dest.unlink(missing_ok=True)


def _emit_new_spans(
    spans: list[FilmSpan],
    published: set[tuple[str, float, float]],
    publish_span: SpanSink,
) -> None:
    for span in spans:
        key = (span.kind, round(span.start_seconds, 3), round(span.end_seconds, 3))
        if key in published:
            continue
        published.add(key)
        publish_span(span.kind, span.start_seconds, span.end_seconds)


def _clip_span(span: FilmSpan, start: float, end: float) -> FilmSpan | None:
    left = max(span.start_seconds, start)
    right = min(span.end_seconds, end)
    if right - left < 0.05:
        return None
    return FilmSpan(kind=span.kind, start_seconds=left, end_seconds=right)


def _merge_ranges(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((start, end) for start, end in ranges if end > start)
    merged: list[tuple[float, float]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1] + 0.05:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            continue
        merged.append((start, end))
    return merged


def _subtract(
    ranges: list[tuple[float, float]],
    blockers: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    pending = list(ranges)
    for start, end in blockers:
        nxt: list[tuple[float, float]] = []
        for left, right in pending:
            if end <= left or start >= right:
                nxt.append((left, right))
                continue
            if start > left:
                nxt.append((left, start))
            if end < right:
                nxt.append((end, right))
        pending = nxt
    return [(left, right) for left, right in pending if right > left]


def _overlaps_play(start: float, end: float, plays: list[tuple[float, float]]) -> bool:
    return any(start < play_end and end > play_start for play_start, play_end in plays)
