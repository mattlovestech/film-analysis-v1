"""Serve the upload page and run the play-boundary agent."""

import inspect
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from film_analysis.agents.errors import MissingApiKeyError, VideoAnalysisError
from film_analysis.agents.frames import extract_window, video_duration
from film_analysis.agents.play_boundaries import PlayBoundaryAgent, agent_from_env
from film_analysis.agents.scan import scan_film
from film_analysis.agents.segments import FilmSpan, format_play
from film_analysis.agents.youtube import (
    FilmInfo,
    canonical_youtube_url,
    download_youtube,
    thumbnail_url,
    youtube_details,
)
from film_analysis.storage.catalog import (
    Catalog,
    CatalogError,
    MemoryCatalog,
    ProjectDetail,
    catalog_from_env,
)
from film_analysis.web.playback import cut_clip

_PAGE = Path(__file__).with_name("index.html").read_text(encoding="utf-8")
_DEFAULT_MAX_BYTES = 500 * 1024 * 1024
_ALLOWED_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
_VIDEO = File(None)
_YOUTUBE = Form("")
_Downloader = Callable[[str, Path], Path]
_Preparer = Callable[[Path], Path]
_Cutter = Callable[[Path, float, float, Path], None]
_Duration = Callable[[Path], float]
_Frames = Callable[[Path, float, float], list[tuple[float, bytes]]]
_Details = Callable[[str], FilmInfo | None]
_MIME_SUFFIX = {
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
    "video/x-matroska": ".mkv",
    "video/avi": ".avi",
    "video/x-msvideo": ".avi",
}


def create_app(
    agent: PlayBoundaryAgent | None = None,
    max_bytes: int | None = None,
    downloader: _Downloader | None = None,
    prepare: _Preparer | None = None,
    cut: _Cutter | None = None,
    store: Catalog | None = None,
    duration: _Duration | None = None,
    frames: _Frames | None = None,
    workers: int | None = None,
    details: _Details | None = None,
) -> FastAPI:
    """Build the upload app. Pass an agent in tests; otherwise read it from the environment."""

    limit = _DEFAULT_MAX_BYTES if max_bytes is None else max_bytes
    fetch = downloader or download_youtube
    running: set[int] = set()
    activity: dict[int, str] = {}
    running_lock = threading.Lock()
    split = cut or cut_clip
    probe = duration or video_duration
    sample = frames or extract_window
    pool = 4 if workers is None else workers
    lookup = youtube_details if details is None else details
    if store is None:
        storage = tempfile.TemporaryDirectory(prefix="film-analysis-")
        catalog: Catalog = MemoryCatalog(Path(storage.name))
    else:
        storage = None
        catalog = store
    app = FastAPI(title="film-analysis", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.storage = storage

    @app.get("/")
    def home() -> HTMLResponse:
        return _page()

    @app.get("/analyses")
    def analyses_page() -> HTMLResponse:
        return _page()

    @app.get("/activity")
    def activity_page() -> HTMLResponse:
        return _page()

    @app.get("/projects/{project_id}")
    def project_page(project_id: int) -> HTMLResponse:
        del project_id
        return _page()

    @app.get("/api/projects")
    def projects() -> list[dict[str, object]]:
        try:
            saved = catalog.list_projects()
        except CatalogError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        return [
            {
                "project": item.project,
                "name": item.name,
                "play_count": item.play_count,
                "thumbnail": thumbnail_url(item.youtube_url),
                "status": item.status,
                "downloaded": item.downloaded_bytes,
                "total_bytes": item.total_bytes,
                "created": item.created_at,
                "youtube": item.youtube_url is not None,
                "running": _running(running, running_lock, item.project),
            }
            for item in saved
        ]

    @app.get("/api/activity")
    def activity_feed() -> dict[str, object]:
        try:
            saved = catalog.list_projects()
        except CatalogError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        with running_lock:
            phases = dict(activity)
            active = set(running)
        jobs: list[dict[str, object]] = []
        for item in saved:
            working = item.project in active
            if not working and item.status not in {"downloading", "scanning"}:
                continue
            try:
                detail = catalog.get_project(item.project)
            except CatalogError as error:
                raise HTTPException(status_code=502, detail=str(error)) from error
            if detail is None:
                continue
            phase = phases.get(item.project, "starting") if working else "stopped"
            jobs.append(_activity_job(detail, phase=phase, running=working))
        return {"workers": pool, "jobs": jobs}

    @app.get("/api/projects/{project_id}")
    def project(project_id: int) -> dict[str, object]:
        try:
            saved = catalog.get_project(project_id)
        except CatalogError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        if saved is None:
            raise HTTPException(status_code=404, detail="That analysis is no longer available.")
        return _project_payload(saved, running=_running(running, running_lock, project_id))

    @app.post("/api/projects/{project_id}/retry")
    def retry(project_id: int) -> dict[str, object]:
        try:
            saved = catalog.get_project(project_id)
        except CatalogError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        if saved is None:
            raise HTTPException(status_code=404, detail="That analysis is no longer available.")
        if not saved.youtube_url:
            raise HTTPException(status_code=400, detail="Upload that film again to retry it.")
        if saved.status not in {"downloading", "failed"}:
            raise HTTPException(status_code=400, detail="That download has already finished.")
        if not _claim(running, running_lock, project_id):
            raise HTTPException(status_code=409, detail="That download is already running.")
        try:
            chosen = agent if agent is not None else agent_from_env()
        except MissingApiKeyError as error:
            _release(running, running_lock, project_id)
            raise HTTPException(status_code=503, detail=str(error)) from error
        try:
            catalog.set_progress(
                project_id,
                status="downloading",
                duration_seconds=None,
                scanned_seconds=0.0,
                error=None,
            )
            catalog.set_download(project_id, 0, None)
        except CatalogError as error:
            _release(running, running_lock, project_id)
            raise HTTPException(status_code=502, detail=str(error)) from error
        _start_film(
            running,
            activity,
            running_lock,
            catalog=catalog,
            project_id=project_id,
            agent=chosen,
            fetch=fetch,
            split=split,
            probe=probe,
            sample=sample,
            workers=pool,
            youtube=saved.youtube_url,
            upload_path=None,
            details=lookup,
        )
        detail = catalog.get_project(project_id)
        if detail is None:
            raise HTTPException(status_code=502, detail="Could not save that analysis.")
        return _project_payload(detail, running=True)

    @app.get("/projects/{project_id}/plays/{play_number}")
    def play(project_id: int, play_number: int) -> FileResponse:
        path = catalog.clip_file(project_id, play_number)
        if path is None:
            raise HTTPException(status_code=404, detail="That play is no longer available.")
        return FileResponse(path, media_type="video/mp4", content_disposition_type="inline")

    @app.post("/analyze")
    async def analyze(
        video: UploadFile | None = _VIDEO,
        youtube: str = _YOUTUBE,
    ) -> dict[str, object]:
        has_file = _has_upload(video)
        link = youtube.strip()
        if has_file and link:
            raise HTTPException(status_code=400, detail="Use a file or a YouTube link.")
        if not has_file and not link:
            raise HTTPException(
                status_code=400,
                detail="Upload a video or paste a YouTube link.",
            )
        canonical = _youtube_url(link) if link else None
        upload_path: Path | None = None
        if video is not None and has_file:
            upload_path = await _save_upload(video, _suffix(video), limit)
        try:
            chosen = agent if agent is not None else agent_from_env()
        except MissingApiKeyError as error:
            if upload_path is not None:
                upload_path.unlink(missing_ok=True)
            raise HTTPException(status_code=503, detail=str(error)) from error

        if canonical is not None:
            name = canonical
        else:
            assert video is not None
            name = Path(video.filename or "film").name
        try:
            project_id = catalog.create_project(name, canonical)
        except CatalogError as error:
            if upload_path is not None:
                upload_path.unlink(missing_ok=True)
            raise HTTPException(status_code=502, detail=str(error)) from error

        _claim(running, running_lock, project_id)
        _start_film(
            running,
            activity,
            running_lock,
            catalog=catalog,
            project_id=project_id,
            agent=chosen,
            fetch=fetch,
            split=split,
            probe=probe,
            sample=sample,
            workers=pool,
            youtube=canonical,
            upload_path=upload_path,
            details=lookup,
        )
        detail = catalog.get_project(project_id)
        if detail is None:
            raise HTTPException(status_code=502, detail="Could not save that analysis.")
        return _project_payload(detail, running=True)

    return app


def main() -> None:
    """Run the upload page. Set FILM_ANALYSIS_HOST to publish it beyond localhost."""

    import uvicorn

    _load_dotenv()
    host = os.environ.get("FILM_ANALYSIS_HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(create_app(store=catalog_from_env()), host=host, port=port)


def _page() -> HTMLResponse:
    return HTMLResponse(_PAGE, headers={"Cache-Control": "no-store"})


def _claim(running: set[int], lock: threading.Lock, project_id: int) -> bool:
    with lock:
        if project_id in running:
            return False
        running.add(project_id)
        return True


def _release(running: set[int], lock: threading.Lock, project_id: int) -> None:
    with lock:
        running.discard(project_id)


def _running(running: set[int], lock: threading.Lock, project_id: int) -> bool:
    with lock:
        return project_id in running


def _note(activity: dict[int, str], lock: threading.Lock, project_id: int, phase: str) -> None:
    with lock:
        activity[project_id] = phase


def _clear(activity: dict[int, str], lock: threading.Lock, project_id: int) -> None:
    with lock:
        activity.pop(project_id, None)


def _start_film(
    running: set[int],
    activity: dict[int, str],
    lock: threading.Lock,
    **kwargs: object,
) -> None:
    project_id = int(kwargs["project_id"])
    _note(activity, lock, project_id, "starting")

    def run() -> None:
        try:
            _watch_film(activity, lock, **kwargs)  # type: ignore[arg-type]
        finally:
            _clear(activity, lock, project_id)
            _release(running, lock, project_id)

    threading.Thread(target=run, daemon=True).start()


def _watch_film(
    activity: dict[int, str],
    lock: threading.Lock,
    catalog: Catalog,
    project_id: int,
    agent: PlayBoundaryAgent,
    fetch: _Downloader,
    split: _Cutter,
    probe: _Duration,
    sample: _Frames,
    workers: int,
    youtube: str | None,
    upload_path: Path | None,
    details: _Details,
) -> None:
    path: Path | None = upload_path
    owned: Path | None = None
    length: float | None = None
    scanned = 0.0
    try:
        _progress(catalog, project_id, "downloading", None, 0.0, None)
        if youtube is not None:
            try:
                info = details(youtube)
            except Exception:
                info = None
            _remember_film(catalog, project_id, info)
            _note(activity, lock, project_id, "downloading")
            owned = Path(tempfile.mkdtemp(prefix="film-download-"))
            path = _fetch(fetch, youtube, owned, _download_reporter(catalog, project_id))
        if path is None:
            raise VideoAnalysisError("Could not read that film.")
        _note(activity, lock, project_id, "saving")
        catalog.save_source(project_id, path)
        length = probe(path)

        def publish_play(number: int, start: float, end: float, clip: Path) -> None:
            catalog.add_play(project_id, number, start, end, clip)

        def publish_span(kind: str, start: float, end: float) -> None:
            catalog.add_span(project_id, kind, start, end)

        def progress(seconds: float) -> None:
            nonlocal scanned
            scanned = seconds
            _progress(catalog, project_id, "scanning", length, seconds, None)

        def label(samples: list[tuple[float, bytes]]) -> tuple[FilmSpan, ...]:
            return agent.label_window(samples)

        _note(activity, lock, project_id, "scanning")
        _progress(catalog, project_id, "scanning", length, 0.0, None)
        scan_film(
            path,
            length,
            label,
            sample,
            split,
            publish_play,
            publish_span,
            progress,
            workers=workers,
        )
        _progress(catalog, project_id, "ready", length, length, None)
    except (Exception, SystemExit) as error:
        _progress(catalog, project_id, "failed", length, scanned, _job_error(error))
    finally:
        if upload_path is not None:
            upload_path.unlink(missing_ok=True)
        if owned is not None:
            shutil.rmtree(owned, ignore_errors=True)


def _download_reporter(catalog: Catalog, project_id: int) -> Callable[[int, int | None], None]:
    last = 0.0

    def report(done: int, total: int | None) -> None:
        nonlocal last
        now = time.monotonic()
        finished = total is not None and total > 0 and done >= total
        if not finished and now - last < 0.4:
            return
        last = now
        try:
            catalog.set_download(project_id, done, total)
        except CatalogError:
            return

    return report


def _fetch(
    fetch: _Downloader,
    url: str,
    directory: Path,
    progress: Callable[[int, int | None], None],
) -> Path:
    try:
        signature = inspect.signature(fetch)
    except (TypeError, ValueError):
        return fetch(url, directory)
    positional = [
        param
        for param in signature.parameters.values()
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
    ]
    variadic = any(
        param.kind is inspect.Parameter.VAR_POSITIONAL for param in signature.parameters.values()
    )
    if variadic or len(positional) >= 3:
        return fetch(url, directory, progress)  # type: ignore[call-arg]
    return fetch(url, directory)


def _remember_film(catalog: Catalog, project_id: int, info: FilmInfo | None) -> None:
    if info is None:
        return
    try:
        catalog.describe_film(
            project_id,
            name=info.title or None,
            channel=info.channel or None,
            duration_seconds=info.duration_seconds,
        )
    except CatalogError:
        return


def _progress(
    catalog: Catalog,
    project_id: int,
    status: str,
    duration_seconds: float | None,
    scanned_seconds: float,
    error: str | None,
) -> None:
    try:
        catalog.set_progress(
            project_id,
            status=status,
            duration_seconds=duration_seconds,
            scanned_seconds=scanned_seconds,
            error=error,
        )
    except CatalogError:
        return


def _job_error(error: Exception) -> str:
    if isinstance(error, SystemExit):
        text = "Could not fetch that YouTube video."
    elif isinstance(error, (VideoAnalysisError, MissingApiKeyError, CatalogError)):
        text = str(error)
    else:
        text = "Could not read that film."
    text = " ".join(text.split())
    if len(text) > 180:
        text = text[:177] + "..."
    return text or "Could not read that film."


def _activity_job(project: ProjectDetail, *, phase: str, running: bool) -> dict[str, object]:
    return {
        "project": project.project,
        "name": project.name,
        "phase": phase,
        "running": running,
        "status": project.status,
        "downloaded": project.downloaded_bytes,
        "total_bytes": project.total_bytes,
        "scanned": project.scanned_seconds,
        "duration": project.duration_seconds,
        "error": project.error,
        "play_count": len(project.plays),
    }


def _project_payload(project: ProjectDetail, *, running: bool = False) -> dict[str, object]:
    return {
        "running": running,
        "project": project.project,
        "name": project.name,
        "status": project.status,
        "duration": project.duration_seconds,
        "scanned": project.scanned_seconds,
        "error": project.error,
        "channel": project.channel,
        "youtube_url": project.youtube_url,
        "thumbnail": thumbnail_url(project.youtube_url),
        "downloaded": project.downloaded_bytes,
        "total_bytes": project.total_bytes,
        "plays": [
            {
                "play_number": play.play_number,
                "label": format_play(play.play_number, play.start_seconds, play.end_seconds),
                "start": play.start_seconds,
                "end": play.end_seconds,
                "video": f"/projects/{project.project}/plays/{play.play_number}",
            }
            for play in project.plays
        ],
    }


def _load_dotenv() -> None:
    """Load a local .env without overriding variables already set in the shell."""

    candidates = (Path.cwd() / ".env", Path(__file__).resolve().parents[3] / ".env")
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _has_upload(video: UploadFile | None) -> bool:
    return video is not None and bool((video.filename or "").strip())


def _youtube_url(value: str) -> str:
    try:
        return canonical_youtube_url(value)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def _suffix(video: UploadFile) -> str:
    suffix = Path(video.filename or "").suffix.lower()
    if suffix in _ALLOWED_SUFFIXES:
        return suffix
    mime = (video.content_type or "").split(";", maxsplit=1)[0].strip().lower()
    mapped = _MIME_SUFFIX.get(mime)
    if mapped is not None:
        return mapped
    raise HTTPException(status_code=400, detail="Upload a video file.")


async def _save_upload(video: UploadFile, suffix: str, max_bytes: int) -> Path:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        path = Path(handle.name)
        size = 0
        try:
            while chunk := await video.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(status_code=413, detail="That video is too large.")
                handle.write(chunk)
            if size == 0:
                raise HTTPException(status_code=400, detail="That video is empty.")
        except Exception:
            path.unlink(missing_ok=True)
            raise
    return path
