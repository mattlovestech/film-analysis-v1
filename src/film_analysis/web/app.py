"""Serve the upload page and run the play-boundary agent."""

import asyncio
import contextlib
import inspect
import logging
import os
import shutil
import tempfile
import threading
import traceback
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from film_analysis.agents.errors import MissingApiKeyError, VideoAnalysisError
from film_analysis.agents.frames import extract_window, video_duration
from film_analysis.agents.play_boundaries import PlayBoundaryAgent, agent_from_env, bind_read_note
from film_analysis.agents.scan import (
    MAX_WORKERS,
    FilmWindowError,
    WorkerLimit,
    clamp_workers,
    scan_film,
)
from film_analysis.agents.segments import FilmSpan, format_play
from film_analysis.agents.youtube import (
    FilmInfo,
    canonical_youtube_url,
    detach_progress,
    download_youtube,
    thumbnail_url,
    youtube_details,
)
from film_analysis.domain.models import PlayType
from film_analysis.storage.catalog import (
    TEAM_IMAGE_KINDS,
    ActivityStep,
    Catalog,
    CatalogError,
    FilmTeams,
    MemoryCatalog,
    PlayRecord,
    ProjectDetail,
    ProjectSummary,
    ReadNote,
    SpanRecord,
    catalog_from_env,
)
from film_analysis.web.cloud import cloud_target
from film_analysis.web.playback import cut_clip
from film_analysis.web.report import scouting_pdf, untagged_plays
from film_analysis.web.teams import lineup_moment, make_portrait, split_lineup, suggest_team_names

_PAGE = Path(__file__).with_name("index.html")
_DEFAULT_MAX_BYTES = 500 * 1024 * 1024
_ALLOWED_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
_VIDEO = File(None)
_YOUTUBE = Form("")
_SAVED = Form("")
_Downloader = Callable[[str, Path], Path]
_Preparer = Callable[[Path], Path]
_Cutter = Callable[[Path, float, float, Path], None]
_Duration = Callable[[Path], float]
_Frames = Callable[[Path, float, float], list[tuple[float, bytes]]]
_Details = Callable[[str], FilmInfo | None]


class _ReviewDecision(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    decision: str

    @field_validator("decision")
    @classmethod
    def _known_decision(cls, value: str) -> str:
        if value not in {"scrimmage", "special_teams", "junk"}:
            raise ValueError("decision must be scrimmage, special_teams, or junk")
        return value

    @model_validator(mode="after")
    def _end_after_start(self) -> "_ReviewDecision":
        if self.end <= self.start:
            raise ValueError("end must be after start")
        return self


_SPECIAL_PLAY_TYPES = {
    PlayType.PUNT.value,
    PlayType.KICKOFF.value,
    PlayType.FIELD_GOAL.value,
    PlayType.EXTRA_POINT.value,
}
_PLAY_LABELS = {item.value for item in PlayType} | {"scrimmage", "special_teams", "junk"}


class _WorkerSetting(BaseModel):
    workers: int | None = Field(default=None, ge=1, le=MAX_WORKERS)
    place: str | None = None

    @field_validator("place")
    @classmethod
    def _known_place(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value not in {"local", "cloud"}:
            raise ValueError("place must be local or cloud")
        return value

    @model_validator(mode="after")
    def _has_change(self) -> "_WorkerSetting":
        if self.workers is None and self.place is None:
            raise ValueError("Choose where jobs run, or how many workers.")
        return self


class _TeamNames(BaseModel):
    left_name: str = ""
    right_name: str = ""
    mine: str | None = None

    @field_validator("left_name", "right_name")
    @classmethod
    def _name(cls, value: str) -> str:
        text = " ".join(value.split())
        if len(text) > 40:
            raise ValueError("Use a team name of 40 characters or fewer.")
        return text

    @field_validator("mine")
    @classmethod
    def _side(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        if value not in {"left", "right"}:
            raise ValueError("Mark which side is your team.")
        return value


class _ArchiveFlag(BaseModel):
    archived: bool


class _PlayLabel(BaseModel):
    play_type: str | None = None
    reviewed: bool | None = None
    unit: str | None = None

    @field_validator("play_type")
    @classmethod
    def _known_type(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value not in _PLAY_LABELS:
            raise ValueError("play type must be a known play")
        return value

    @field_validator("unit")
    @classmethod
    def _known_unit(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value not in {"offense", "defense"}:
            raise ValueError("Mark the play as offense or defense.")
        return value

    @model_validator(mode="after")
    def _has_change(self) -> "_PlayLabel":
        if self.play_type is None and self.reviewed is None and self.unit is None:
            raise ValueError("Choose a play type, mark the review, or tag offense or defense.")
        return self


def _stored_play_label(play_type: str) -> tuple[str, str | None]:
    """Return the family and the specific type stored for a user's choice."""

    if play_type in {"scrimmage", "special_teams"}:
        return play_type, None
    if play_type in _SPECIAL_PLAY_TYPES:
        return "special_teams", play_type
    return "scrimmage", play_type


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
    lineup: Callable[[Path, float], tuple[bytes, bytes]] | None = None,
    portrait: Callable[[bytes, str], tuple[bytes, str]] | None = None,
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
    draw_lineup = split_lineup if lineup is None else lineup
    draw_portrait = make_portrait if portrait is None else portrait
    if store is None:
        storage = tempfile.TemporaryDirectory(prefix="film-analysis-")
        catalog: Catalog = MemoryCatalog(Path(storage.name))
    else:
        storage = None
        catalog = store
    _stop_orphans(catalog)
    slots = WorkerLimit(pool)
    if workers is None:
        with contextlib.suppress(CatalogError):
            slots.set(catalog.worker_count())
    app = FastAPI(title="film-analysis", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.storage = storage

    @app.get("/ping")
    def ping() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/")
    def home() -> HTMLResponse:
        return _page()

    @app.get("/analyses")
    def analyses_page() -> HTMLResponse:
        return _page()

    @app.get("/projects")
    def projects_page() -> HTMLResponse:
        return _page()

    @app.get("/activity")
    def activity_page() -> HTMLResponse:
        return _page()

    @app.get("/settings")
    def settings_page() -> HTMLResponse:
        return _page()

    @app.api_route("/via-cloud/{path:path}", methods=["GET", "POST", "HEAD"])
    async def via_cloud(path: str, request: Request) -> StreamingResponse:
        """Send one request from this computer's page to the Cloudflare site."""

        body = await request.body()
        forwarded = {
            key: value
            for key, value in request.headers.items()
            if key.lower() in {"content-type", "range", "accept", "user-agent"}
        }
        status, headers, upstream = await asyncio.to_thread(
            _open_cloud,
            request.method,
            path,
            request.url.query,
            forwarded,
            body,
        )

        def chunks() -> Iterator[bytes]:
            try:
                while True:
                    block = upstream.read(256 * 1024)
                    if not block:
                        break
                    yield block
            finally:
                upstream.close()

        return StreamingResponse(chunks(), status_code=status, headers=headers)

    @app.get("/projects/{project_id}")
    def project_page(project_id: int) -> HTMLResponse:
        del project_id
        return _page()

    @app.get("/play/{uid}")
    def play_page(uid: str) -> HTMLResponse:
        del uid
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
                "error": item.error,
                "scanned": item.scanned_seconds,
                "review_count": item.review_count,
                "duration": item.duration_seconds,
                "archived": item.archived,
            }
            for item in saved
        ]

    @app.get("/api/activity")
    def activity_feed(request: Request) -> dict[str, object]:
        try:
            saved = catalog.list_activity()
            projects = catalog.list_projects()
        except CatalogError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        by_id = {item.project: item for item in projects}
        with running_lock:
            active = set(running)
        seen = {item.project for item in saved}
        dated: list[tuple[str, dict[str, object]]] = []
        for item in saved:
            summary = by_id.get(item.project)
            if summary is None:
                continue
            if summary.archived and not (item.running or item.project in active):
                continue
            phase, working = _activity_state(item.phase, item.running, item.project in active)
            dated.append(
                (
                    item.updated_at,
                    _activity_job(
                        summary,
                        phase=phase,
                        running=working,
                        updated=item.updated_at,
                        steps=item.steps,
                    ),
                )
            )
        for summary in projects:
            if summary.archived or summary.project in seen or summary.status != "ready":
                continue
            stamp = summary.created_at or ""
            dated.append(
                (stamp, _activity_job(summary, phase="ready", running=False, updated=stamp))
            )
        dated.sort(key=lambda pair: pair[0], reverse=True)
        return {**_run_payload(catalog, slots, request), "jobs": [job for _, job in dated]}

    @app.get("/api/settings")
    def read_settings(request: Request) -> dict[str, object]:
        return _run_payload(catalog, slots, request)

    @app.post("/api/settings")
    def update_settings(body: _WorkerSetting, request: Request) -> dict[str, object]:
        here = _request_place(request)
        if body.place is not None and body.place != here:
            detail = (
                "Open the Cloudflare site to run jobs there."
                if body.place == "cloud"
                else "Local runs on your computer. This site runs on Cloudflare."
            )
            raise HTTPException(status_code=400, detail=detail)
        try:
            if body.workers is not None:
                catalog.set_worker_count(body.workers)
                slots.set(body.workers)
        except CatalogError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        return _run_payload(catalog, slots, request)

    @app.get("/api/projects/{project_id}")
    def project(project_id: int) -> dict[str, object]:
        try:
            saved = catalog.get_project(project_id)
        except CatalogError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        if saved is None:
            raise HTTPException(status_code=404, detail="That analysis is no longer available.")
        return _project_payload(
            saved,
            running=_running(running, running_lock, project_id),
            steps=_saved_steps(catalog, project_id),
            catalog=catalog,
        )

    @app.post("/api/projects/{project_id}/teams/stills")
    def team_stills(project_id: int) -> dict[str, object]:
        saved = _require_project(catalog, project_id)
        _require_identified(saved)
        _store_lineup(catalog, saved, draw_lineup)
        return _project_payload(
            saved,
            running=_running(running, running_lock, project_id),
            steps=_saved_steps(catalog, project_id),
            catalog=catalog,
        )

    @app.post("/api/projects/{project_id}/teams")
    def save_teams(project_id: int, body: _TeamNames) -> dict[str, object]:
        saved = _require_project(catalog, project_id)
        _require_identified(saved)
        try:
            needs_lineup = catalog.team_image(project_id, "left") is None
        except CatalogError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        if needs_lineup:
            _store_lineup(catalog, saved, draw_lineup)
        named = _named_sides(catalog, saved, body)
        _store_portraits(catalog, project_id, named, draw_portrait)
        try:
            catalog.set_film_teams(project_id, named)
        except CatalogError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        return _project_payload(
            saved,
            running=_running(running, running_lock, project_id),
            steps=_saved_steps(catalog, project_id),
            catalog=catalog,
        )

    @app.get("/projects/{project_id}/teams/{kind}")
    def team_picture(project_id: int, kind: str) -> FileResponse:
        if kind not in TEAM_IMAGE_KINDS:
            raise HTTPException(status_code=404, detail="That picture is not available.")
        try:
            path = catalog.team_image(project_id, kind)
        except CatalogError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        if path is None or not path.is_file():
            raise HTTPException(status_code=404, detail="That picture is not available.")
        media = "image/svg+xml" if path.suffix == ".svg" else "image/jpeg"
        return FileResponse(path, media_type=media)

    @app.get("/api/projects/{project_id}/report.pdf")
    def scouting_report(project_id: int) -> Response:
        saved = _require_project(catalog, project_id)
        if saved.status != "ready" or not saved.plays:
            raise HTTPException(
                status_code=400,
                detail="The scouting report waits until the plays are identified.",
            )
        waiting = untagged_plays(saved.plays)
        if waiting:
            noun = "play" if waiting == 1 else "plays"
            raise HTTPException(
                status_code=400,
                detail=f"Tag {waiting} {noun} as offense or defense before the scouting report.",
            )
        try:
            teams = catalog.film_teams(project_id)
        except CatalogError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        pdf = scouting_pdf(saved.name, teams, _ordered_plays(saved))
        return Response(
            content=pdf,
            media_type="application/pdf",
            headers={"Content-Disposition": 'attachment; filename="scouting-report.pdf"'},
        )

    @app.post("/api/projects/{project_id}/archive")
    def archive_project(project_id: int, body: _ArchiveFlag) -> dict[str, object]:
        try:
            catalog.set_archived(project_id, body.archived)
        except CatalogError as error:
            missing = "not found" in str(error).lower()
            detail = "That analysis is no longer available." if missing else str(error)
            raise HTTPException(status_code=404 if missing else 502, detail=detail) from error
        return {"project": project_id, "archived": body.archived}

    @app.post("/api/projects/{project_id}/retry")
    def retry(project_id: int, request: Request) -> dict[str, object]:
        try:
            saved = catalog.get_project(project_id)
        except CatalogError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        if saved is None:
            raise HTTPException(status_code=404, detail="That analysis is no longer available.")
        if not saved.youtube_url and catalog.source_file(project_id) is None:
            raise HTTPException(status_code=400, detail="Upload that film again to retry it.")
        if saved.status not in {"downloading", "scanning", "failed"}:
            raise HTTPException(status_code=400, detail="That film has already finished.")
        cloud = _cloud_choice(request, catalog)
        if cloud is None and not _claim(running, running_lock, project_id):
            raise HTTPException(status_code=409, detail="That download is already running.")
        try:
            chosen = agent if agent is not None else agent_from_env()
        except MissingApiKeyError as error:
            if cloud is None:
                _release(running, running_lock, project_id)
            raise HTTPException(status_code=503, detail=str(error)) from error
        resume_scan = _can_resume_scan(catalog, project_id, probe)
        try:
            if resume_scan:
                catalog.set_progress(
                    project_id,
                    status="scanning",
                    duration_seconds=saved.duration_seconds,
                    scanned_seconds=saved.scanned_seconds,
                    error=None,
                )
            else:
                catalog.set_progress(
                    project_id,
                    status="downloading",
                    duration_seconds=None,
                    scanned_seconds=0.0,
                    error=None,
                )
                catalog.set_download(project_id, 0, None)
        except CatalogError as error:
            if cloud is None:
                _release(running, running_lock, project_id)
            raise HTTPException(status_code=502, detail=str(error)) from error
        if cloud is not None:
            return _dispatch_cloud(request, catalog, project_id, saved.youtube_url, cloud, slots)
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
            limit=slots,
            youtube=saved.youtube_url,
            upload_path=None,
            details=lookup,
            resume_scan=resume_scan,
            resume_at=saved.scanned_seconds if resume_scan else 0.0,
        )
        detail = catalog.get_project(project_id)
        if detail is None:
            raise HTTPException(status_code=502, detail="Could not save that analysis.")
        return _project_payload(detail, running=True, catalog=catalog)

    @app.get("/projects/{project_id}/source")
    def source(project_id: int) -> FileResponse:
        path = catalog.source_file(project_id)
        if path is None:
            raise HTTPException(status_code=404, detail="That film is no longer available.")
        return FileResponse(path, media_type="video/mp4", content_disposition_type="inline")

    @app.post("/api/projects/{project_id}/reviews")
    def resolve_review(project_id: int, decision: _ReviewDecision) -> dict[str, object]:
        try:
            saved = catalog.get_project(project_id)
        except CatalogError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        if saved is None:
            raise HTTPException(status_code=404, detail="That analysis is no longer available.")
        span = _matching_review(saved.spans, decision.start, decision.end)
        if span is None:
            raise HTTPException(status_code=404, detail="That stretch was already marked.")
        if decision.decision != "junk":
            film = catalog.source_file(project_id)
            if film is None:
                raise HTTPException(status_code=400, detail="That film is no longer available.")
            number = 1 + max((play.play_number for play in saved.plays), default=0)
            folder = Path(tempfile.mkdtemp(prefix="film-review-"))
            dest = folder / "clip.mp4"
            try:
                split(film, span.start_seconds, span.end_seconds, dest)
                catalog.add_play(
                    project_id,
                    number,
                    span.start_seconds,
                    span.end_seconds,
                    dest,
                    decision.decision,
                )
            except (VideoAnalysisError, CatalogError) as error:
                raise HTTPException(status_code=400, detail=str(error)) from error
            finally:
                shutil.rmtree(folder, ignore_errors=True)
        try:
            catalog.drop_span(project_id, span.start_seconds, span.end_seconds)
            saved = catalog.get_project(project_id)
        except CatalogError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        if saved is None:
            raise HTTPException(status_code=502, detail="Could not save that analysis.")
        return _project_payload(
            saved,
            running=_running(running, running_lock, project_id),
            steps=_saved_steps(catalog, project_id),
            catalog=catalog,
        )

    @app.get("/api/plays/{uid}")
    def play_detail(uid: str) -> dict[str, object]:
        parsed = _parse_play_uid(uid)
        if parsed is None:
            raise HTTPException(status_code=404, detail="That play is no longer available.")
        project_id, play_number = parsed
        try:
            saved = catalog.get_project(project_id)
        except CatalogError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        if saved is None:
            raise HTTPException(status_code=404, detail="That play is no longer available.")
        payload = _play_payload(saved, play_number)
        if payload is None:
            raise HTTPException(status_code=404, detail="That play is no longer available.")
        return payload

    @app.post("/api/plays/{uid}")
    def relabel_play(uid: str, choice: _PlayLabel) -> dict[str, object]:
        parsed = _parse_play_uid(uid)
        if parsed is None:
            raise HTTPException(status_code=404, detail="That play is no longer available.")
        project_id, play_number = parsed
        try:
            saved = catalog.get_project(project_id)
        except CatalogError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        if saved is None or not any(play.play_number == play_number for play in saved.plays):
            raise HTTPException(status_code=404, detail="That play is no longer available.")
        try:
            if choice.play_type == "junk":
                catalog.drop_play(project_id, play_number)
                return {
                    "removed": True,
                    "project": project_id,
                    "film": f"/projects/{project_id}",
                }
            if choice.play_type is not None:
                family, play_type = _stored_play_label(choice.play_type)
                catalog.set_play_type(project_id, play_number, play_type, family)
            elif choice.unit is not None:
                catalog.set_play_unit(project_id, play_number, choice.unit)
            elif choice.reviewed is not None:
                catalog.set_reviewed(project_id, play_number, choice.reviewed)
            saved = catalog.get_project(project_id)
        except CatalogError as error:
            status = 404 if str(error) == "That play is no longer available." else 502
            raise HTTPException(status_code=status, detail=str(error)) from error
        if saved is None:
            raise HTTPException(status_code=502, detail="Could not save that play.")
        payload = _play_payload(saved, play_number)
        if payload is None:
            raise HTTPException(status_code=404, detail="That play is no longer available.")
        return payload

    @app.get("/projects/{project_id}/plays/{play_number}")
    def play(project_id: int, play_number: int) -> FileResponse:
        path = catalog.clip_file(project_id, play_number)
        if path is None:
            raise HTTPException(status_code=404, detail="That play is no longer available.")
        return FileResponse(path, media_type="video/mp4", content_disposition_type="inline")

    @app.get("/api/downloads")
    def downloads() -> list[dict[str, object]]:
        try:
            saved = catalog.list_saved_films()
        except CatalogError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        return [{"project": item.project, "name": item.name} for item in saved]

    @app.post("/analyze")
    async def analyze(
        request: Request,
        video: UploadFile | None = _VIDEO,
        youtube: str = _YOUTUBE,
        saved: str = _SAVED,
    ) -> dict[str, object]:
        has_file = _has_upload(video)
        link = youtube.strip()
        saved_id = _saved_id(saved)
        sources = sum((has_file, bool(link), saved_id is not None))
        if sources > 1:
            raise HTTPException(
                status_code=400,
                detail="Use a file, a YouTube link, or a downloaded film.",
            )
        if sources == 0:
            raise HTTPException(
                status_code=400,
                detail="Upload a video, paste a YouTube link, or choose a downloaded film.",
            )
        if saved_id is not None:
            return _analyze_saved(
                catalog,
                saved_id,
                agent=agent,
                fetch=fetch,
                split=split,
                probe=probe,
                sample=sample,
                workers=pool,
                limit=slots,
                running=running,
                activity=activity,
                running_lock=running_lock,
                details=lookup,
                cloud=_cloud_choice(request, catalog),
                request=request,
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

        cloud = _cloud_choice(request, catalog)
        if cloud is not None:
            try:
                try:
                    if upload_path is not None:
                        catalog.save_source(project_id, upload_path)
                except CatalogError as error:
                    _progress(catalog, project_id, "failed", None, 0.0, str(error))
                    raise HTTPException(status_code=502, detail=str(error)) from error
            finally:
                if upload_path is not None:
                    upload_path.unlink(missing_ok=True)
            return _dispatch_cloud(request, catalog, project_id, canonical, cloud, slots)

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
            limit=slots,
            youtube=canonical,
            upload_path=upload_path,
            details=lookup,
        )
        detail = catalog.get_project(project_id)
        if detail is None:
            raise HTTPException(status_code=502, detail="Could not save that analysis.")
        return _project_payload(detail, running=True, catalog=catalog)

    return app


def main() -> None:
    """Run the upload page. Set FILM_ANALYSIS_HOST to publish it beyond localhost."""

    import uvicorn

    _load_dotenv()
    host = os.environ.get("FILM_ANALYSIS_HOST", "127.0.0.1")
    port = int(os.environ.get("FILM_ANALYSIS_PORT") or os.environ.get("PORT") or "8000")
    uvicorn.run(create_app(store=catalog_from_env()), host=host, port=port)


_CLOUD_ORIGIN = "https://film-analysis.broad-night-d47d.workers.dev"


def _open_cloud(
    method: str,
    path: str,
    query: str,
    headers: dict[str, str],
    body: bytes,
) -> tuple[int, dict[str, str], object]:
    url = f"{_CLOUD_ORIGIN}/{path.lstrip('/')}"
    if query:
        url = f"{url}?{query}"
    outgoing = urllib.request.Request(url, data=body or None, headers=headers, method=method)
    try:
        upstream = urllib.request.urlopen(outgoing, timeout=300)
    except urllib.error.HTTPError as error:
        upstream = error
    kept = {}
    for key in ("Content-Type", "Content-Range", "Accept-Ranges"):
        value = upstream.headers.get(key)
        if value:
            kept[key] = value
    return upstream.status, kept, upstream


def _page() -> HTMLResponse:
    return HTMLResponse(_PAGE.read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})


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


def _note(
    activity: dict[int, str],
    lock: threading.Lock,
    catalog: Catalog,
    project_id: int,
    phase: str,
) -> None:
    with lock:
        activity[project_id] = phase
    _save_activity(catalog, project_id, phase, running=True, stack="local")


def _save_activity(
    catalog: Catalog,
    project_id: int,
    phase: str,
    *,
    running: bool,
    stack: str | None = None,
    workers: int | None = None,
) -> None:
    try:
        catalog.record_activity(
            project_id,
            phase,
            running=running,
            stack=stack,
            workers=workers,
        )
    except CatalogError:
        return


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
    catalog: Catalog = kwargs["catalog"]  # type: ignore[assignment]
    _note(activity, lock, catalog, project_id, "starting")

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
    resume_scan: bool = False,
    resume_at: float = 0.0,
    limit: WorkerLimit | None = None,
) -> None:
    path: Path | None = upload_path
    owned: Path | None = None
    length: float | None = None
    scanned = resume_at
    phase = "reading" if resume_scan else "downloading"
    try:
        if resume_scan:
            path = catalog.source_file(project_id)
            if path is not None:
                try:
                    length = probe(path)
                except VideoAnalysisError:
                    path = None
            if path is None or length is None or length <= 0:
                if youtube is None and upload_path is None:
                    raise VideoAnalysisError("That film is no longer available.")
                resume_scan = False
                resume_at = 0.0
                scanned = 0.0
                phase = "downloading"
        if not resume_scan:
            path = upload_path
            phase = "downloading"
            _progress(catalog, project_id, "downloading", None, 0.0, None)
            scanned = 0.0
            if youtube is not None:
                try:
                    info = details(youtube)
                except Exception:
                    info = None
                _remember_film(catalog, project_id, info)
                _note(activity, lock, catalog, project_id, "downloading")
                owned = Path(tempfile.mkdtemp(prefix="film-download-"))
                path = _fetch(fetch, youtube, owned, _download_reporter(catalog, project_id))
            if path is None:
                raise VideoAnalysisError("Could not read that film.")
            phase = "saving"
            _note(activity, lock, catalog, project_id, "saving")
            catalog.save_source(project_id, path)
            length = probe(path)
            resume_at = 0.0
            phase = "reading"

        saved = catalog.get_project(project_id)
        already = [
            (play.start_seconds, play.end_seconds) for play in (saved.plays if saved else ())
        ]
        plays = saved.plays if saved else ()
        next_number = 1 + max((play.play_number for play in plays), default=0)

        def publish_play(
            number: int,
            start: float,
            end: float,
            clip: Path,
            family: str,
        ) -> None:
            catalog.add_play(project_id, number, start, end, clip, family)

        def publish_span(kind: str, start: float, end: float) -> None:
            catalog.add_span(project_id, kind, start, end)

        def progress(seconds: float) -> None:
            nonlocal scanned
            scanned = seconds
            _progress(catalog, project_id, "scanning", length, seconds, None)

        def label(samples: list[tuple[float, bytes]]) -> tuple[FilmSpan, ...]:
            bind_read_note(remember_read)
            try:
                return agent.label_window(samples)
            finally:
                bind_read_note(None)

        def review_window(samples: list[tuple[float, bytes]]) -> tuple[FilmSpan, ...]:
            watch = getattr(agent, "review_window", None)
            if watch is None:
                return ()
            bind_read_note(remember_read)
            try:
                return watch(samples)
            finally:
                bind_read_note(None)

        review = review_window if getattr(agent, "review_window", None) else None

        phase = "reading"
        _note(activity, lock, catalog, project_id, "scanning")
        _progress(catalog, project_id, "scanning", length, resume_at, None)
        if resume_at <= 0:
            with contextlib.suppress(CatalogError):
                catalog.clear_reads(project_id)

        def mark_review() -> None:
            _note(activity, lock, catalog, project_id, "checking")

        def remember_read(
            start: float,
            end: float,
            frames: int,
            nbytes: int,
            reason: str,
            reply: str,
        ) -> None:
            detail = _read_detail(frames, nbytes, reason, reply)
            try:
                already = catalog.list_reads(project_id)
                if any(
                    note.start_seconds == start
                    and note.end_seconds == end
                    and note.detail == detail
                    for note in already
                ):
                    return
                catalog.add_read(
                    project_id,
                    ReadNote(start, end, frames, nbytes, detail),
                )
            except CatalogError:
                return

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
            limit=limit,
            review=review,
            resume_at=resume_at,
            next_number=next_number,
            already=already,
            on_review=mark_review,
        )
        _progress(catalog, project_id, "ready", length, length, None)
        _save_activity(catalog, project_id, "ready", running=False)
    except (Exception, SystemExit) as error:
        _log_failure(project_id, phase, scanned, length, error)
        _progress(
            catalog,
            project_id,
            "failed",
            length,
            scanned,
            _job_error(error, phase=phase, scanned=scanned, duration=length),
        )
        _save_activity(catalog, project_id, "stopped", running=False)
    finally:
        if upload_path is not None:
            upload_path.unlink(missing_ok=True)
        if owned is not None:
            shutil.rmtree(owned, ignore_errors=True)


def _can_resume_scan(catalog: Catalog, project_id: int, probe: _Duration) -> bool:
    """The download finished when the film file is already stored and readable."""

    path = catalog.source_file(project_id)
    if path is None:
        return False
    try:
        return probe(path) > 0
    except VideoAnalysisError:
        return False


def _download_reporter(catalog: Catalog, project_id: int) -> Callable[[int, int | None], None]:
    def send(done: int, total: int | None) -> None:
        try:
            catalog.set_download(project_id, done, total)
        except CatalogError:
            return

    return detach_progress(send)


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


_INTERRUPTED = "The server stopped while this was running."


def _request_place(request: Request) -> str:
    """Cloud is the public site. Local is this computer, and only this computer."""

    return "cloud" if request.headers.get("x-film-delegate") == "1" else "local"


def _cloud_choice(request: Request, catalog: Catalog) -> tuple[str, str] | None:
    """Set when the browser called the Cloudflare worker, so the cutter should start."""

    del catalog
    if _request_place(request) != "cloud":
        return None
    return cloud_target(request.headers.get("x-film-worker-url")) or ("", "")


def _run_payload(catalog: Catalog, slots: WorkerLimit, request: Request) -> dict[str, object]:
    place = _request_place(request)
    try:
        workers = catalog.worker_count()
    except CatalogError:
        workers = slots.get()
    return {
        "workers": workers,
        "place": place,
        "cloud": place == "cloud",
    }


def _chosen_workers(catalog: Catalog, slots: WorkerLimit) -> int:
    try:
        return clamp_workers(catalog.worker_count())
    except CatalogError:
        return slots.get()


def _dispatch_cloud(
    request: Request,
    catalog: Catalog,
    project_id: int,
    youtube: str | None,
    cloud: tuple[str, str],
    slots: WorkerLimit,
) -> JSONResponse | dict[str, object]:
    """Tell the Cloudflare worker to start the cutter for this film."""

    del cloud
    workers = _chosen_workers(catalog, slots)
    return _delegate(catalog, project_id, youtube, workers)


def _delegate(catalog: Catalog, project_id: int, youtube: str | None, workers: int) -> JSONResponse:
    """Tell the Worker to start the cutter. The website does not call it itself."""

    _save_activity(
        catalog,
        project_id,
        "starting",
        running=True,
        stack="cloud",
        workers=workers,
    )
    detail = catalog.get_project(project_id)
    if detail is None:
        raise HTTPException(status_code=502, detail="Could not save that analysis.")
    return JSONResponse(
        _project_payload(detail, running=True, catalog=catalog),
        headers={
            "x-film-start": str(project_id),
            "x-film-youtube": youtube or "",
            "x-film-workers": str(workers),
        },
    )


def _stop_orphans(catalog: Catalog) -> None:
    """A restart leaves in-progress films with no error. Record why they stopped.

    In the cloud the cutter keeps running after this process sleeps, so those
    films are still going.
    """

    if os.environ.get("FILM_INTERNAL_TOKEN"):
        return
    try:
        saved = catalog.list_projects()
    except CatalogError:
        return
    for item in saved:
        if item.status not in {"downloading", "scanning"}:
            continue
        try:
            detail = catalog.get_project(item.project)
        except CatalogError:
            continue
        if detail is None:
            continue
        try:
            catalog.set_progress(
                item.project,
                status=detail.status,
                duration_seconds=detail.duration_seconds,
                scanned_seconds=detail.scanned_seconds,
                error=detail.error or _INTERRUPTED,
            )
        except CatalogError:
            continue
    _remember_stopped_activity(catalog, saved)


def _remember_stopped_activity(catalog: Catalog, saved: list[object]) -> None:
    """Copy in-progress films onto the activity table and close ones this process left behind."""

    try:
        current = {row.project: row for row in catalog.list_activity()}
    except CatalogError:
        return
    for row in current.values():
        if not row.running:
            continue
        try:
            catalog.record_activity(row.project, "stopped", running=False)
        except CatalogError:
            continue
        current[row.project] = row
    for item in saved:
        project = getattr(item, "project", None)
        status = getattr(item, "status", None)
        if not isinstance(project, int) or project in current:
            continue
        if status not in {"downloading", "scanning", "failed"}:
            continue
        phase = "failed" if status == "failed" else "stopped"
        try:
            catalog.record_activity(project, phase, running=False)
        except CatalogError:
            continue


_LOG = logging.getLogger("film_analysis")
_ERROR_LIMIT = 2000


def _log_failure(
    project_id: int,
    phase: str,
    scanned: float,
    duration: float | None,
    error: BaseException,
) -> None:
    length = "unknown" if duration is None else f"{duration:.1f}s"
    _LOG.exception(
        "film %s failed while %s (scanned %.1fs of %s): %s",
        project_id,
        phase,
        scanned,
        length,
        error,
    )


def _job_error(
    error: BaseException,
    *,
    phase: str,
    scanned: float,
    duration: float | None,
) -> str:
    lines = [_failure_place(error, phase, scanned, duration), *_report_lines(error)]
    text = "\n".join(line for line in lines if line)
    if len(text) > _ERROR_LIMIT:
        text = text[: _ERROR_LIMIT - 3] + "..."
    return text or "Could not read that film."


def _report_lines(error: BaseException) -> list[str]:
    lines: list[str] = []
    for index, item in enumerate(_cause_chain(error)):
        prefix = "" if index == 0 else "caused by "
        lines.append(prefix + _error_detail(item))
        lines.extend(f"at {frame}" for frame in _app_frames(item))
    return lines


def _cause_chain(error: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while isinstance(current, BaseException) and id(current) not in seen:
        seen.add(id(current))
        if not isinstance(current, FilmWindowError):
            chain.append(current)
        cause = current.__cause__
        current = cause if isinstance(cause, BaseException) else None
    return chain


def _app_frames(error: BaseException) -> list[str]:
    if error.__traceback__ is None:
        return []
    frames: list[str] = []
    for frame in traceback.extract_tb(error.__traceback__):
        path = frame.filename.replace("\\", "/")
        marker = "film_analysis/"
        if marker not in path:
            continue
        short = path[path.rfind(marker):]
        frames.append(f"{short}:{frame.name}:{frame.lineno}")
    return frames[-4:]


def _failure_place(
    error: BaseException,
    phase: str,
    scanned: float,
    duration: float | None,
) -> str:
    if isinstance(error, FilmWindowError):
        span = f"Stopped while reading {_clock(error.start)}–{_clock(error.end)}"
        if duration is not None and duration > 0:
            return f"{span} of {_clock(duration)}."
        return f"{span}."
    if phase == "saving":
        return "Stopped while saving the film."
    if phase == "reading":
        if duration is not None and duration > 0:
            return f"Stopped while reading at {_clock(scanned)} of {_clock(duration)}."
        return f"Stopped while reading at {_clock(scanned)}."
    return "Stopped while downloading."


def _error_detail(error: BaseException) -> str:
    if isinstance(error, SystemExit):
        code = error.code
        extra = f" exit {code}" if code not in (None, 0, "") else ""
        return f"SystemExit:{extra} Could not fetch that YouTube video."
    text = str(error).strip()
    name = type(error).__name__
    if not text:
        return name
    if text.startswith(f"{name}:"):
        return text
    return f"{name}: {text}"


def _clock(seconds: float) -> str:
    whole = max(0, round(seconds))
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


_LIVE_PHASES = frozenset({"starting", "downloading", "saving", "scanning", "checking"})


def _activity_state(phase: str, running: bool, active: bool) -> tuple[str, bool]:
    """Keep a finished film complete. A live phase that is no longer running has stopped."""

    if phase == "ready":
        return "ready", False
    if (running or active) and phase in _LIVE_PHASES:
        return phase, True
    return "stopped", False


def _activity_job(
    project: ProjectDetail | ProjectSummary,
    *,
    phase: str,
    running: bool,
    updated: str,
    steps: tuple[ActivityStep, ...] = (),
) -> dict[str, object]:
    play_count = (
        project.play_count if isinstance(project, ProjectSummary) else len(project.plays)
    )
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
        "youtube": project.youtube_url is not None,
        "play_count": play_count,
        "review_count": _review_count(project),
        "thumbnail": thumbnail_url(project.youtube_url),
        "created": getattr(project, "created_at", None),
        "updated": updated,
        "steps": _step_times(steps),
    }


def _review_count(project: ProjectDetail | ProjectSummary) -> int:
    counted = getattr(project, "review_count", None)
    if isinstance(counted, int):
        return counted
    return sum(1 for span in getattr(project, "spans", ()) if span.kind == "uncertain")


def _play_uid(project_id: int, play_number: int) -> str:
    return f"{project_id}-{play_number}"


def _parse_play_uid(uid: str) -> tuple[int, int] | None:
    project, separator, number = uid.partition("-")
    if separator != "-" or not project.isdigit() or not number.isdigit():
        return None
    project_id = int(project)
    play_number = int(number)
    if (
        project_id < 1
        or play_number < 1
        or str(project_id) != project
        or str(play_number) != number
    ):
        return None
    return project_id, play_number


def _ordered_plays(project: ProjectDetail) -> list[PlayRecord]:
    return sorted(project.plays, key=lambda play: (play.start_seconds, play.play_number))


def _play_summary(project_id: int, play: PlayRecord) -> dict[str, object]:
    return {
        "uid": _play_uid(project_id, play.play_number),
        "play_number": play.play_number,
        "label": format_play(play.play_number, play.start_seconds, play.end_seconds),
        "start": play.start_seconds,
        "end": play.end_seconds,
        "family": play.family,
        "play_type": play.play_type,
        "unit": play.unit,
        "reviewed": play.reviewed,
        "video": f"/projects/{project_id}/plays/{play.play_number}",
    }


def _play_payload(project: ProjectDetail, play_number: int) -> dict[str, object] | None:
    ordered = _ordered_plays(project)
    index = next(
        (position for position, play in enumerate(ordered) if play.play_number == play_number),
        None,
    )
    if index is None:
        return None
    play = ordered[index]
    previous_play = ordered[index - 1] if index else None
    next_play = ordered[index + 1] if index + 1 < len(ordered) else None
    return {
        **_play_summary(project.project, play),
        "project": project.project,
        "name": project.name,
        "channel": project.channel,
        "position": index + 1,
        "count": len(ordered),
        "film": f"/projects/{project.project}",
        "previous": (
            None
            if previous_play is None
            else _play_uid(project.project, previous_play.play_number)
        ),
        "next": None if next_play is None else _play_uid(project.project, next_play.play_number),
        "tracker": [
            {
                "uid": _play_uid(project.project, item.play_number),
                "position": place,
                "reviewed": item.reviewed,
            }
            for place, item in enumerate(ordered, start=1)
        ],
    }


def _saved_steps(catalog: Catalog, project_id: int) -> tuple[ActivityStep, ...]:
    try:
        row = catalog.get_activity(project_id)
    except CatalogError:
        return ()
    if row is None:
        return ()
    return row.steps


def _step_times(steps: tuple[ActivityStep, ...]) -> list[dict[str, object]]:
    now = datetime.now(UTC)
    timed: list[dict[str, object]] = []
    for step in steps:
        started = _stamp(step.started_at)
        ended = now if step.ended_at is None else _stamp(step.ended_at)
        seconds = 0.0
        if started is not None and ended is not None:
            seconds = max(0.0, (ended - started).total_seconds())
        timed.append(
            {
                "name": step.name,
                "seconds": round(seconds, 1),
                "running": step.ended_at is None,
                "stack": step.stack,
                "workers": step.workers,
            }
        )
    return timed


def _stamp(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _require_project(catalog: Catalog, project_id: int) -> ProjectDetail:
    try:
        saved = catalog.get_project(project_id)
    except CatalogError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    if saved is None:
        raise HTTPException(status_code=404, detail="That analysis is no longer available.")
    return saved


def _require_identified(project: ProjectDetail) -> None:
    if project.status != "ready" or not project.plays:
        raise HTTPException(
            status_code=400,
            detail="Name the teams after the plays are identified.",
        )


def _lineup_video(catalog: Catalog, project: ProjectDetail) -> tuple[Path, float]:
    try:
        video = catalog.source_file(project.project)
    except CatalogError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    if video is not None and video.is_file():
        return video, lineup_moment(project.plays)
    clip = catalog.clip_file(project.project, project.plays[0].play_number)
    if clip is None:
        raise HTTPException(status_code=400, detail="Could not read a lineup from that film.")
    length = max(0.0, project.plays[0].end_seconds - project.plays[0].start_seconds)
    return clip, min(0.6, length / 2)


def _store_lineup(
    catalog: Catalog,
    project: ProjectDetail,
    draw: Callable[[Path, float], tuple[bytes, bytes]],
) -> None:
    video, at = _lineup_video(catalog, project)
    try:
        left, right = draw(video, at)
    except (VideoAnalysisError, OSError) as error:
        raise HTTPException(
            status_code=400,
            detail="Could not read a lineup from that film.",
        ) from error
    if not left or not right:
        raise HTTPException(status_code=400, detail="Could not read a lineup from that film.")
    try:
        catalog.save_team_image(project.project, "left", left, ".jpg")
        catalog.save_team_image(project.project, "right", right, ".jpg")
    except CatalogError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


def _named_sides(catalog: Catalog, project: ProjectDetail, body: _TeamNames) -> FilmTeams:
    suggested = suggest_team_names(
        project.name,
        _still_bytes(catalog, project.project, "left"),
        _still_bytes(catalog, project.project, "right"),
    )
    return FilmTeams(
        body.left_name or suggested[0],
        body.right_name or suggested[1],
        body.mine,
    )


def _still_bytes(catalog: Catalog, project_id: int, kind: str) -> bytes | None:
    try:
        path = catalog.team_image(project_id, kind)
    except CatalogError:
        return None
    if path is None or not path.is_file():
        return None
    try:
        return path.read_bytes()
    except OSError:
        return None


def _store_portraits(
    catalog: Catalog,
    project_id: int,
    teams: FilmTeams,
    draw: Callable[[bytes, str], tuple[bytes, str]],
) -> None:
    try:
        for side, name in (("left", teams.left_name), ("right", teams.right_name)):
            still = catalog.team_image(project_id, side)
            if still is None or not still.is_file():
                raise HTTPException(
                    status_code=400,
                    detail="Could not read a lineup from that film.",
                )
            image, suffix = draw(still.read_bytes(), name)
            catalog.save_team_image(project_id, f"{side}-player", image, suffix)
    except CatalogError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


def _teams_view(catalog: Catalog, project_id: int) -> dict[str, object] | None:
    try:
        saved = catalog.film_teams(project_id)
        pictures = {kind: catalog.team_image(project_id, kind) for kind in TEAM_IMAGE_KINDS}
    except CatalogError:
        return None

    def url(kind: str) -> str | None:
        path = pictures[kind]
        if path is None or not path.is_file():
            return None
        return f"/projects/{project_id}/teams/{kind}?v={int(path.stat().st_mtime)}"

    if saved is None and url("left") is None and url("right") is None:
        return None
    try:
        project = catalog.get_project(project_id)
    except CatalogError:
        project = None
    suggested = suggest_team_names(
        "" if project is None else project.name,
        _still_bytes(catalog, project_id, "left"),
        _still_bytes(catalog, project_id, "right"),
    )
    left_name = suggested[0] if saved is None or not saved.left_name else saved.left_name
    right_name = suggested[1] if saved is None or not saved.right_name else saved.right_name
    return {
        "left_name": left_name,
        "right_name": right_name,
        "mine": None if saved is None else saved.mine,
        "left_still": url("left"),
        "right_still": url("right"),
        "left_player": url("left-player"),
        "right_player": url("right-player"),
    }


def _project_payload(
    project: ProjectDetail,
    *,
    running: bool = False,
    steps: tuple[ActivityStep, ...] = (),
    catalog: Catalog | None = None,
) -> dict[str, object]:
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
        "steps": _step_times(steps),
        "teams": None if catalog is None else _teams_view(catalog, project.project),
        "plays": [_play_summary(project.project, play) for play in _ordered_plays(project)],
        "reads": [] if catalog is None else _read_views(catalog, project.project),
        "reviews": [
            {
                "start": span.start_seconds,
                "end": span.end_seconds,
                "video": f"/projects/{project.project}/source",
            }
            for span in sorted(
                (span for span in project.spans if span.kind == "uncertain"),
                key=lambda span: (span.start_seconds, span.end_seconds),
            )
        ],
    }


def _read_views(catalog: Catalog, project_id: int) -> list[dict[str, object]]:
    try:
        notes = catalog.list_reads(project_id)
    except CatalogError:
        return []
    return [
        {
            "start": note.start_seconds,
            "end": note.end_seconds,
            "detail": note.detail,
        }
        for note in notes
    ]


def _read_detail(frames: int, nbytes: int, reason: str, reply: str) -> str:
    quote = " ".join(reply.split())
    if len(quote) > 180:
        quote = quote[:177] + "..."
    text = f"{frames} frames, {_byte_size(nbytes)}. {reason.rstrip('.')}."
    if quote:
        text = f"{text} Reply: {quote}"
    if len(text) > 500:
        return text[:497] + "..."
    return text


def _byte_size(nbytes: int) -> str:
    if nbytes >= 1_000_000:
        return f"{nbytes / 1_000_000:.1f} MB"
    if nbytes >= 1000:
        return f"{round(nbytes / 1000)} KB"
    return f"{nbytes} B"


def _matching_review(
    spans: tuple[SpanRecord, ...],
    start: float,
    end: float,
) -> SpanRecord | None:
    return next(
        (
            span
            for span in spans
            if span.kind == "uncertain"
            and abs(span.start_seconds - start) < 0.05
            and abs(span.end_seconds - end) < 0.05
        ),
        None,
    )


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


def _saved_id(value: str) -> int | None:
    raw = value.strip()
    if not raw:
        return None
    try:
        project_id = int(raw)
    except ValueError:
        raise HTTPException(status_code=400, detail="Choose a downloaded film.") from None
    if project_id < 1:
        raise HTTPException(status_code=400, detail="Choose a downloaded film.")
    return project_id


def _analyze_saved(
    catalog: Catalog,
    source_id: int,
    *,
    agent: PlayBoundaryAgent | None,
    fetch: _Downloader,
    split: _Cutter,
    probe: _Duration,
    sample: _Frames,
    workers: int,
    limit: WorkerLimit,
    running: set[int],
    activity: dict[int, str],
    running_lock: threading.Lock,
    details: _Details,
    cloud: tuple[str, str] | None,
    request: Request,
) -> dict[str, object] | JSONResponse:
    """Start a new analysis from a film that is already stored."""

    try:
        origin = catalog.get_project(source_id)
    except CatalogError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    if origin is None:
        raise HTTPException(status_code=404, detail="That film is no longer available.")
    try:
        chosen = agent if agent is not None else agent_from_env()
    except MissingApiKeyError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    try:
        project_id = catalog.create_project(origin.name, origin.youtube_url)
        catalog.describe_film(
            project_id,
            name=origin.name,
            channel=origin.channel,
            duration_seconds=origin.duration_seconds,
        )
        copied = catalog.copy_source(source_id, project_id)
    except CatalogError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    if not copied:
        _progress(
            catalog,
            project_id,
            "failed",
            None,
            0.0,
            "That film is no longer available.",
        )
        raise HTTPException(status_code=400, detail="That film is no longer available.")
    if cloud is not None:
        return _dispatch_cloud(request, catalog, project_id, None, cloud, limit)
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
        workers=workers,
        limit=limit,
        youtube=None,
        upload_path=None,
        details=details,
        resume_scan=True,
        resume_at=0.0,
    )
    detail = catalog.get_project(project_id)
    if detail is None:
        raise HTTPException(status_code=502, detail="Could not save that analysis.")
    return _project_payload(detail, running=True, catalog=catalog)


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
