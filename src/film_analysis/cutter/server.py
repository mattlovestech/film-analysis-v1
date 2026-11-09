"""One film, in order. Labelers and clip cuts run in other containers."""

import json
import os
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from film_analysis.agents.errors import VideoAnalysisError
from film_analysis.agents.scan import (
    MAX_WORKERS,
    WorkerLimit,
    clamp_workers,
    family_at,
    ready_non_plays,
    ready_plays,
    resolve_held,
    review_ranges,
    stable_through,
    window_bounds,
)
from film_analysis.agents.segments import FilmSpan
from film_analysis.storage.catalog import Catalog, CatalogError, catalog_from_env
from film_analysis.web.app import _job_error, _load_dotenv, _progress

app = FastAPI(title="film-cutter", docs_url=None, redoc_url=None, openapi_url=None)
_lock = threading.Lock()
_running = False


class Start(BaseModel):
    project_id: int
    youtube: str | None = None
    worker_url: str
    workers: int | None = Field(default=None, ge=1, le=MAX_WORKERS)


@app.get("/ping")
def ping() -> dict[str, bool]:
    return {"ok": True}


@app.post("/start")
def start(body: Start, authorization: str | None = Header(default=None)) -> dict[str, bool]:
    global _running
    _check_token(authorization)
    token = os.environ.get("FILM_INTERNAL_TOKEN", "").strip()
    with _lock:
        if _running:
            return {"ok": True}
        _running = True
    threading.Thread(
        target=_guard,
        args=(body, token),
        daemon=True,
    ).start()
    return {"ok": True}


def main() -> None:
    import uvicorn

    _load_dotenv()
    port = int(os.environ.get("FILM_ANALYSIS_PORT") or os.environ.get("PORT") or "8000")
    uvicorn.run(app, host="0.0.0.0", port=port)


def _guard(body: Start, token: str) -> None:
    global _running
    catalog = catalog_from_env()
    try:
        _cut_film(catalog, body, token)
    except Exception as error:
        _progress(catalog, body.project_id, "failed", None, 0.0, _job_error(error))
        _mark(catalog, body.project_id, "stopped", running=False)
    finally:
        with _lock:
            _running = False


def _worker_limit(catalog: Catalog, requested: int | None) -> tuple[WorkerLimit, int]:
    """Start from the count this film was given, and remember the stored count."""

    try:
        saved = clamp_workers(catalog.worker_count())
    except CatalogError:
        saved = clamp_workers(requested or 4)
    return WorkerLimit(requested if requested is not None else saved), saved


def _follow_worker_count(
    catalog: Catalog,
    slots: WorkerLimit,
    stop: threading.Event,
    baseline: int,
) -> None:
    """Pick up a worker setting changed while this film is still being read."""

    seen = baseline
    while not stop.wait(1):
        seen = _apply_worker_count(catalog, slots, seen)


def _apply_worker_count(catalog: Catalog, slots: WorkerLimit, seen: int) -> int:
    """Move the labeler cap when the stored count has changed."""

    try:
        current = clamp_workers(catalog.worker_count())
    except CatalogError:
        return seen
    if current == seen:
        return seen
    slots.set(current)
    return current


def _label_task(
    slots: WorkerLimit,
    worker: str,
    token: str,
    project_id: int,
    index: int,
    start: float,
    end: float,
    review: bool,
) -> dict[str, object]:
    name = "review" if review else "label"
    with slots:
        return _task(
            worker,
            token,
            f"{name}-{project_id}-{index}",
            {
                "kind": "label",
                "project_id": project_id,
                "start": start,
                "end": end,
                "review": review,
            },
            900,
        )


def _mark(
    catalog: Catalog,
    project_id: int,
    phase: str,
    *,
    running: bool = True,
    workers: int | None = None,
) -> None:
    try:
        catalog.record_activity(
            project_id,
            phase,
            running=running,
            stack="cloud" if running else None,
            workers=workers if running else None,
        )
    except CatalogError:
        return


def _cut_film(catalog: Catalog, body: Start, token: str) -> None:
    project_id = body.project_id
    _mark(catalog, project_id, "downloading")
    _progress(catalog, project_id, "downloading", None, 0.0, None)
    ingested = _task(
        body.worker_url,
        token,
        f"ingest-{project_id}",
        {"kind": "ingest", "project_id": project_id, "youtube": body.youtube},
        timeout=3600,
    )
    duration = float(ingested["duration"])
    bounds = window_bounds(duration)
    if not bounds:
        _progress(catalog, project_id, "ready", duration, duration, None)
        return

    collected: dict[int, tuple[FilmSpan, ...]] = {}
    kept: list[tuple[float, float]] = []
    published: set[tuple[str, float, float]] = set()
    play_number = 0
    slots, baseline = _worker_limit(catalog, body.workers)
    stop = threading.Event()
    watcher = threading.Thread(
        target=_follow_worker_count,
        args=(catalog, slots, stop, baseline),
        daemon=True,
    )
    watcher.start()
    try:
        _mark(catalog, project_id, "scanning", workers=slots.get())
        _progress(catalog, project_id, "scanning", duration, 0.0, None)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            jobs = [
                pool.submit(
                    _label_task,
                    slots,
                    body.worker_url,
                    token,
                    project_id,
                    index,
                    start,
                    end,
                    False,
                )
                for index, (start, end) in enumerate(bounds)
            ]
            for index, job in enumerate(jobs):
                collected[index] = _spans(job.result())
                stable = stable_through(index + 1, bounds, duration)
                spans = [span for group in collected.values() for span in group]
                for start, end in ready_plays(spans, stable):
                    if _overlaps(start, end, kept):
                        continue
                    play_number += 1
                    _task(
                        body.worker_url,
                        token,
                        f"cut-{project_id}-{play_number}",
                        {
                            "kind": "cut",
                            "project_id": project_id,
                            "play_number": play_number,
                            "start": start,
                            "end": end,
                            "family": family_at(spans, start, end),
                        },
                        900,
                    )
                    kept.append((start, end))
                _emit_spans(
                    catalog,
                    project_id,
                    [span for span in ready_non_plays(spans, stable)],
                    published,
                )
                _progress(catalog, project_id, "scanning", duration, stable, None)

        spans = [span for group in collected.values() for span in group]
        if review_ranges(spans):
            _mark(catalog, project_id, "checking", workers=slots.get())
        _review(catalog, body, token, spans, kept, published, play_number, slots)
        _progress(catalog, project_id, "ready", duration, duration, None)
        _mark(catalog, project_id, "ready", running=False)
    finally:
        stop.set()


def _review(
    catalog: Catalog,
    body: Start,
    token: str,
    spans: list[FilmSpan],
    kept: list[tuple[float, float]],
    published: set[tuple[str, float, float]],
    play_number: int,
    slots: WorkerLimit,
) -> None:
    held = review_ranges(spans)
    if not held:
        return
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        jobs = [
            pool.submit(
                _label_task,
                slots,
                body.worker_url,
                token,
                body.project_id,
                index,
                start,
                end,
                True,
            )
            for index, (start, end) in enumerate(held)
        ]
        decisions = [_spans(job.result()) for job in jobs]
    plays, leftovers = resolve_held(held, decisions, kept)
    for play in plays:
        play_number += 1
        _task(
            body.worker_url,
            token,
            f"cut-{body.project_id}-{play_number}",
            {
                "kind": "cut",
                "project_id": body.project_id,
                "play_number": play_number,
                "start": play.start,
                "end": play.end,
                "family": play.family,
            },
            900,
        )
    _emit_spans(catalog, body.project_id, leftovers, published)


def _emit_spans(
    catalog: Catalog,
    project_id: int,
    spans: list[FilmSpan],
    published: set[tuple[str, float, float]],
) -> None:
    for span in spans:
        key = (span.kind, round(span.start_seconds, 3), round(span.end_seconds, 3))
        if key in published:
            continue
        published.add(key)
        try:
            catalog.add_span(project_id, span.kind, span.start_seconds, span.end_seconds)
        except CatalogError:
            return


def _task(
    worker: str,
    token: str,
    name: str,
    payload: dict[str, object],
    timeout: int,
) -> dict[str, object]:
    request = urllib.request.Request(
        f"{worker}/internal/tasks/{name}/run",
        data=json.dumps(payload).encode(),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise VideoAnalysisError(_detail(detail)) from error
    except urllib.error.URLError as error:
        raise VideoAnalysisError("Could not reach a worker for that film.") from error
    if not isinstance(body, dict):
        raise VideoAnalysisError("A worker returned nothing for that film.")
    return body


def _spans(payload: dict[str, object]) -> tuple[FilmSpan, ...]:
    raw = payload.get("spans") or []
    if not isinstance(raw, list):
        return ()
    spans: list[FilmSpan] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        spans.append(
            FilmSpan(
                kind=str(item.get("kind") or "uncertain"),
                start_seconds=float(item.get("start") or 0),
                end_seconds=float(item.get("end") or 0),
            )
        )
    return tuple(spans)


def _overlaps(start: float, end: float, plays: list[tuple[float, float]]) -> bool:
    return any(start < play_end and end > play_start for play_start, play_end in plays)


def _detail(raw: str) -> str:
    text = " ".join(raw.split())
    if not text:
        return "Could not read that film."
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text[:180]
    if isinstance(parsed, dict) and isinstance(parsed.get("detail"), str):
        return parsed["detail"]
    return text[:180]


def _check_token(authorization: str | None) -> None:
    expected = os.environ.get("FILM_INTERNAL_TOKEN", "").strip()
    if expected and authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Unauthorized.")
