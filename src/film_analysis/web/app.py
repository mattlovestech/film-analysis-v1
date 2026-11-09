"""Serve the upload page and run the play-boundary agent."""

import asyncio
import contextlib
import csv
import inspect
import io
import json
import logging
import os
import queue
import shutil
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from pydantic import BaseModel, Field, field_validator, model_validator
from starlette.background import BackgroundTask

from film_analysis.chart.call import SPECIAL_CALLS, ChartedPlay, chart_play
from film_analysis.chart.formation import (
    FORMATIONS,
    FormationRead,
    formation_label,
    read_formation,
)
from film_analysis.domain.models import PlayType
from film_analysis.ingest.cost import read_cost
from film_analysis.ingest.errors import MissingApiKeyError, VideoAnalysisError
from film_analysis.ingest.frames import extract_window, sample_frames, video_duration
from film_analysis.ingest.youtube import (
    FilmInfo,
    canonical_youtube_url,
    detach_progress,
    download_youtube,
    thumbnail_url,
    youtube_details,
)
from film_analysis.learning import propose_guidance, publish_guidance
from film_analysis.plays.boundaries import PlayBoundaryAgent, agent_from_env, bind_read_note
from film_analysis.plays.scan import (
    MAX_WORKERS,
    FilmWindowError,
    WorkerLimit,
    clamp_workers,
    processing_percent,
    scan_film,
)
from film_analysis.plays.segments import FilmSpan, format_clock, format_play
from film_analysis.scout.report import (
    chart_kinds,
    live_snapshot,
    play_reads,
    report_sections,
    scouting_blocks,
    scouting_csv,
    scouting_pdf,
    tendency_lines,
)
from film_analysis.storage.catalog import (
    TEAM_IMAGE_KINDS,
    ActivityRecord,
    ActivityStep,
    Catalog,
    CatalogError,
    CoachEdit,
    FilmTeams,
    MemoryCatalog,
    PlayRecord,
    ProjectDetail,
    ProjectSummary,
    ReadNote,
    ReportAnalysis,
    SpanRecord,
    catalog_from_env,
)
from film_analysis.teams.players import usable_player_jpeg
from film_analysis.teams.uniforms import (
    Uniform,
    action_moments,
    choose_action_shot,
    color_words,
    decode_uniforms,
    detect_uniforms,
    dominant_colors,
    encode_uniforms,
    focus_player,
    label_home_away,
    lineup_moment,
    play_frames,
    player_action_frames,
    player_frames,
    player_photo,
    portrait_uniform,
    reference_for_model,
    split_lineup,
    still_frame,
    suggest_team_names,
    team_cards,
    uniform_colors,
    uniform_line,
)
from film_analysis.web.cloud import cloud_target
from film_analysis.web.playback import cut_clip

_FORMATION_KEYS = frozenset(FORMATIONS)
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
_Still = Callable[[Path, float], bytes]
_poster_lock = threading.Lock()
_poster_wait: dict[tuple[int, int], float] = {}


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
    PlayType.KICK_RETURN.value,
    PlayType.PUNT_RETURN.value,
    PlayType.FIELD_GOAL.value,
    PlayType.EXTRA_POINT.value,
}
_PLAY_LABELS = {item.value for item in PlayType} | {"scrimmage", "special_teams", "junk"}


class _WorkerSetting(BaseModel):
    workers: int | None = Field(default=None, ge=1, le=MAX_WORKERS)
    place: str | None = None
    formations: bool | None = None

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
        if self.workers is None and self.place is None and self.formations is None:
            raise ValueError("Choose where jobs run, or how many workers.")
        return self


class _TeamNames(BaseModel):
    left_name: str = ""
    right_name: str = ""
    mine: str | None = None
    redraw: bool = True

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


class _ReportChoice(BaseModel):
    sections: list[str] = Field(default_factory=list)
    restart: bool = False
    plays: list[int] = Field(default_factory=list)


_REPORT_BODY = Body(default=None)


class _ArchiveFlag(BaseModel):
    archived: bool


class _PlayLabel(BaseModel):
    play_type: str | None = None
    reviewed: bool | None = None
    unit: str | None = None
    discarded: bool | None = None
    formation: str | None = None
    clear_formation: bool = False

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

    @field_validator("formation")
    @classmethod
    def _known_formation(cls, value: str | None) -> str | None:
        if value is None:
            return None
        key = value.strip().lower().replace("-", "_").replace(" ", "_")
        if key not in _FORMATION_KEYS:
            raise ValueError("Choose a known formation.")
        return key

    @model_validator(mode="after")
    def _has_change(self) -> "_PlayLabel":
        if (
            self.play_type is None
            and self.reviewed is None
            and self.unit is None
            and self.discarded is None
            and self.formation is None
            and not self.clear_formation
        ):
            raise ValueError(
                "Choose a play type, mark the review, tag offense or defense, "
                "set a formation, or discard the play."
            )
        return self


class _PlayTrim(BaseModel):
    start: float
    end: float

    @model_validator(mode="after")
    def _ordered(self) -> "_PlayTrim":
        if self.start < 0:
            raise ValueError("A play cannot start before the film.")
        if self.end - self.start < 0.5:
            raise ValueError("Keep at least half a second of the play.")
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


def _film_player(*_args: object, **_kwargs: object) -> tuple[bytes, str]:
    """Player cards are frames from the film. Nothing is generated."""

    return b"", ""


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
    portrait: Callable[..., tuple[bytes, str]] | None = None,
    chart: (
        Callable[[Sequence[bytes], Uniform, Uniform, str], ChartedPlay | str | None] | None
    ) = None,
    formations: (
        Callable[..., FormationRead | str | None] | None
    ) = None,
    still: _Still | None = None,
    learner: Callable[[str, Sequence[CoachEdit]], str] | None = None,
) -> FastAPI:
    """Build the upload app. Pass an agent in tests; otherwise read it from the environment."""

    limit = _DEFAULT_MAX_BYTES if max_bytes is None else max_bytes
    fetch = downloader or download_youtube
    running: set[int] = set()
    activity: dict[int, str] = {}
    report_choice: dict[int, frozenset[str]] = {}
    report_events: dict[int, queue.Queue[str | None]] = {}
    running_lock = threading.Lock()
    split = cut or cut_clip
    probe = duration or video_duration
    sample = frames or extract_window
    pool = 4 if workers is None else workers
    lookup = youtube_details if details is None else details
    draw_lineup = split_lineup if lineup is None else lineup
    draw_portrait = _film_player if portrait is None else portrait
    read_play = chart_play if chart is None else chart
    read_formation_play = read_formation if formations is None else formations
    grab = still_frame if still is None else still
    if store is None:
        storage = tempfile.TemporaryDirectory(prefix="film-analysis-")
        catalog: Catalog = MemoryCatalog(Path(storage.name))
    else:
        storage = None
        catalog = store
    _stop_orphans(catalog)
    with contextlib.suppress(CatalogError):
        publish_guidance(catalog.coach_guidance())
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

    @app.get("/projects/{project_id}/review")
    @app.get("/projects/{project_id}/select-teams")
    @app.get("/projects/{project_id}/generate-report")
    def project_screen(project_id: int) -> HTMLResponse:
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
        listed: list[dict[str, object]] = []
        for item in saved:
            phase, report_done, report_total = _list_report_progress(
                catalog, running, running_lock, item
            )
            listed.append(
            {
                "project": item.project,
                "name": item.name,
                "play_count": item.play_count,
                "thumbnail": _thumbnail_for(item, catalog),
                "status": item.status,
                "downloaded": item.downloaded_bytes,
                "total_bytes": item.total_bytes,
                "created": item.created_at,
                "youtube": item.youtube_url is not None,
                "running": _running(running, running_lock, item.project),
                "phase": phase,
                "report_done": report_done,
                "report_total": report_total,
                "error": item.error,
                "scanned": item.scanned_seconds,
                "review_count": item.review_count,
                "duration": item.duration_seconds,
                "archived": item.archived,
                "retries": item.retries,
                "retries_done": item.retries_done,
                "processing": processing_percent(
                    item.duration_seconds,
                    item.scanned_seconds,
                    item.retries,
                    item.retries_done,
                ),
            }
            )
        return listed

    @app.get("/api/activity")
    def activity_feed(request: Request) -> dict[str, object]:
        # Each read is its own database round trip. Run them together, not one after another.
        with ThreadPoolExecutor(max_workers=5) as reads:
            saved_call = reads.submit(catalog.list_activity)
            projects_call = reads.submit(catalog.list_projects)
            costs_call = reads.submit(catalog.play_read_costs)
            read_spend_call = reads.submit(catalog.read_spend)
            report_spend_call = reads.submit(catalog.report_spend)
            try:
                saved = saved_call.result()
                projects = projects_call.result()
            except CatalogError as error:
                raise HTTPException(status_code=502, detail=str(error)) from error
            try:
                play_costs = costs_call.result()
                spend = _merged_spend(read_spend_call.result(), report_spend_call.result())
            except CatalogError:
                play_costs = {}
                spend = {}
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
                        play_costs=play_costs.get(item.project, []),
                        spend=spend.get(item.project),
                        catalog=catalog,
                    ),
                )
            )
        for summary in projects:
            if summary.archived or summary.project in seen or summary.status != "ready":
                continue
            stamp = summary.created_at or ""
            dated.append(
                (
                    stamp,
                    _activity_job(
                        summary,
                        phase="ready",
                        running=False,
                        updated=stamp,
                        play_costs=play_costs.get(summary.project, []),
                        spend=spend.get(summary.project),
                        catalog=catalog,
                    ),
                )
            )
        dated.sort(key=lambda pair: pair[0], reverse=True)
        return {**_run_payload(catalog, slots, request), "jobs": [job for _, job in dated]}

    @app.get("/api/settings")
    def read_settings(request: Request) -> dict[str, object]:
        return _run_payload(catalog, slots, request)

    @app.get("/api/learning")
    def read_learning() -> dict[str, object]:
        """Coach corrections, and the notes the next cut and tag already follow."""

        try:
            edits = catalog.list_coach_edits()
            guidance = catalog.coach_guidance()
        except CatalogError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        return {
            "guidance": guidance,
            "pending": sum(1 for edit in edits if not edit.learned),
            "edits": [_edit_view(edit) for edit in edits],
        }

    @app.post("/api/learning")
    def learn_from_coaches() -> dict[str, object]:
        """Send new coach corrections to the model and store the notes it writes."""

        try:
            pending = [edit for edit in catalog.list_coach_edits(limit=200) if not edit.learned]
            guidance = catalog.coach_guidance()
        except CatalogError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        batch = pending[:40]
        if not batch:
            raise HTTPException(status_code=400, detail="No new coach edits to learn from.")
        if learner is None and not os.environ.get("ANTHROPIC_API_KEY", "").strip():
            raise HTTPException(
                status_code=400,
                detail="Add an API key before sending coach edits.",
            )
        ask = learner or propose_guidance
        try:
            notes = ask(guidance, batch)
        except Exception as error:
            raise HTTPException(
                status_code=502,
                detail="Could not learn from those edits.",
            ) from error
        if not str(notes).strip():
            raise HTTPException(status_code=502, detail="Could not learn from those edits.")
        try:
            catalog.set_coach_guidance(str(notes).strip())
            catalog.mark_coach_edits_learned([edit.edit_id for edit in batch])
        except CatalogError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        publish_guidance(str(notes).strip())
        return {"guidance": str(notes).strip(), "learned": len(batch)}

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
            if body.formations is not None:
                catalog.set_formation_reads(body.formations)
        except CatalogError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        return _run_payload(catalog, slots, request)

    @app.get("/api/teams/reset")
    def team_reset_status() -> dict[str, object]:
        return _team_reset_view()

    @app.post("/api/teams/reset")
    def team_reset_all(failed: bool = False) -> dict[str, object]:
        started = _start_team_reset(
            catalog,
            draw_lineup,
            draw_portrait,
            failed_only=failed,
        )
        if not started:
            current = _team_reset_view()
            if current.get("running"):
                return current
            raise HTTPException(status_code=400, detail="No ready films need new team photos.")
        return _team_reset_view()

    @app.get("/api/projects/{project_id}")
    def project(project_id: int) -> dict[str, object]:
        try:
            page = _load_project_page(catalog, project_id)
        except CatalogError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        saved = page[0]
        if saved is None:
            raise HTTPException(status_code=404, detail="That analysis is no longer available.")
        refreshing = _schedule_portrait_refresh(catalog, saved, draw_lineup, draw_portrait)
        payload = _project_payload(
            saved,
            running=_running(running, running_lock, project_id),
            steps=page[1],
            catalog=catalog,
            film_teams=page[2],
            read_notes=page[3],
            prepared=True,
            still=grab,
            make_poster=True,
        )
        teams = payload.get("teams")
        if refreshing and isinstance(teams, dict):
            teams["refreshing"] = True
        return payload

    @app.post("/api/projects/{project_id}/teams/stills")
    def team_stills(
        project_id: int,
        again: bool = False,
        side: str | None = None,
        wait: bool = True,
    ) -> dict[str, object]:
        saved = _require_project(catalog, project_id)
        _require_identified(saved)
        if side is not None and side not in {"left", "right"}:
            raise HTTPException(status_code=400, detail="Choose the left or right team.")
        _start_team_build(
            catalog,
            saved,
            draw_lineup,
            draw_portrait,
            force=again,
            side=side,
            wait=wait,
        )
        refreshed = catalog.get_project(project_id) or saved
        payload = _project_payload(
            refreshed,
            running=_running(running, running_lock, project_id),
            steps=_saved_steps(catalog, project_id),
            catalog=catalog,
        )
        teams = payload.get("teams")
        progress = _team_progress_view(project_id)
        if isinstance(teams, dict) and progress is not None:
            teams["building"] = bool(progress.get("running"))
            teams["progress"] = progress
        return payload

    @app.post("/api/projects/{project_id}/teams")
    def save_teams(project_id: int, body: _TeamNames) -> dict[str, object]:
        saved = _require_project(catalog, project_id)
        _require_identified(saved)
        try:
            before = catalog.film_teams(project_id)
            needs_lineup = catalog.team_image(project_id, "left") is None
        except CatalogError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        if needs_lineup:
            _store_lineup(catalog, saved, draw_lineup)
            _remember_team_names(catalog, saved)
        named = _named_sides(catalog, saved, body)
        switching = (
            before is not None
            and before.mine in {"left", "right"}
            and named.mine in {"left", "right"}
            and before.mine != named.mine
        )
        if switching and _running(running, running_lock, project_id):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Wait until this report finishes reading, "
                    "then switch the team being scouted."
                ),
            )
        if body.redraw or not _players_saved(catalog, project_id):
            _store_portraits(catalog, project_id, named)
        try:
            catalog.set_film_teams(project_id, named)
            if switching:
                catalog.flip_team_perspective(project_id)
        except CatalogError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        return _project_payload(
            _require_project(catalog, project_id),
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

    @app.get("/api/projects/{project_id}/report")
    def preview_scouting_report(project_id: int, sections: str = "") -> dict[str, object]:
        """The report from plays already marked, before a PDF is made."""

        saved, teams, plays = _report_film(catalog, project_id)
        chosen = _report_sections(sections)
        return _report_payload(saved, teams, plays, chosen)

    @app.post("/api/projects/{project_id}/report")
    def generate_scouting_report(
        project_id: int,
        body: _ReportChoice | None = _REPORT_BODY,
    ) -> dict[str, object]:
        saved, teams, _plays = _report_film(catalog, project_id)
        try:
            chosen = report_sections(None if body is None or not body.sections else body.sections)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        restart = bool(body and body.restart)
        only = _report_play_numbers(body)
        if not _claim(running, running_lock, project_id):
            raise HTTPException(status_code=409, detail="That report is already reading.")
        report_choice[project_id] = chosen
        try:
            for _event in _live_lines(
                catalog,
                saved,
                teams,
                chosen,
                read_play,
                read_formation_play,
                slots,
                restart=restart,
                only=only,
            ):
                pass
        finally:
            _release(running, running_lock, project_id)
        saved = _require_project(catalog, project_id)
        return _report_payload(saved, teams, _report_plays(saved), chosen)

    @app.post("/api/projects/{project_id}/report/live")
    def live_scouting_report(
        project_id: int,
        body: _ReportChoice | None = _REPORT_BODY,
    ) -> StreamingResponse:
        """Start charting in the background and stream progress until it finishes."""

        saved, teams, _plays = _report_film(catalog, project_id)
        try:
            chosen = report_sections(None if body is None or not body.sections else body.sections)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        restart = bool(body and body.restart)
        only = _report_play_numbers(body)
        already = _running(running, running_lock, project_id)
        if already and restart:
            raise HTTPException(status_code=409, detail="That report is already reading.")
        if not already:
            if not _claim(running, running_lock, project_id):
                raise HTTPException(status_code=409, detail="That report is already reading.")
            report_choice[project_id] = chosen
            events: queue.Queue[str | None] = queue.Queue()
            report_events[project_id] = events
            _start_report(
                running,
                running_lock,
                report_events,
                events=events,
                catalog=catalog,
                saved=saved,
                teams=teams,
                chosen=chosen,
                reader=read_play,
                formations=read_formation_play,
                slots=slots,
                restart=restart,
                only=only,
            )
        follow = report_choice.get(project_id, chosen)

        def generate() -> Iterator[str]:
            yield from _follow_report(
                catalog,
                project_id,
                teams,
                follow,
                running,
                running_lock,
                report_events,
                slots,
            )

        return StreamingResponse(
            generate(),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/projects/{project_id}/report/analysis")
    def scouting_report_analysis(project_id: int, sections: str = "") -> dict[str, object]:
        """The scouting-analysis table for a film, plus the live report so far."""

        saved, teams, plays = _report_film(catalog, project_id)
        stored = report_choice.get(project_id)
        chosen = _report_sections(sections) or stored or report_sections(None)
        try:
            kept = {play.play_number for play in plays}
            rows = [
                row
                for row in catalog.list_report_analysis(project_id)
                if row.play_number in kept
            ]
            activity = catalog.get_activity(project_id)
        except CatalogError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        workers = None
        if activity is not None and activity.steps:
            workers = activity.steps[-1].workers
        done = sum(1 for row in rows if row.status in {"ready", "missed"})
        kept_reads = sum(1 for row in rows if row.status == "ready")
        missed = sum(1 for row in rows if row.status == "missed")
        unfinished = sum(1 for row in rows if row.status in {"pending", "reading", "missed"})
        live = _running(running, running_lock, project_id) or bool(
            activity and activity.running and activity.phase == "charting"
        )
        return {
            "project": project_id,
            "running": live,
            "phase": None if activity is None else activity.phase,
            "done": done,
            "kept": kept_reads,
            "missed": missed,
            "total": len(rows)
            or len([play for play in plays if _play_kind(play) in chart_kinds(chosen)]),
            "workers": workers if workers is not None else slots.get(),
            "sections": sorted(chosen),
            "can_resume": (not live) and unfinished > 0,
            "analysis": [
                _analysis_view(row, project_id, _play_by_number(saved, row.play_number))
                for row in rows
            ],
            "snapshot": live_snapshot(teams, plays, chosen),
        }

    @app.get("/api/projects/{project_id}/report.pdf")
    def scouting_report(project_id: int, sections: str = "") -> Response:
        saved, teams, plays = _report_film(catalog, project_id)
        chosen = _report_sections(sections)
        pdf = scouting_pdf(saved.name, teams, plays, chosen)
        return Response(
            content=pdf,
            media_type="application/pdf",
            headers={"Content-Disposition": 'attachment; filename="scouting-report.pdf"'},
        )

    @app.get("/api/projects/{project_id}/report.csv")
    def scouting_sheet(project_id: int, sections: str = "") -> Response:
        saved, teams, plays = _report_film(catalog, project_id)
        chosen = _report_sections(sections)
        sheet = scouting_csv(saved.name, teams, plays, chosen)
        return Response(
            content=sheet,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="scouting-report.csv"'},
        )

    @app.get("/api/projects/{project_id}/plays.zip")
    def export_plays(project_id: int) -> FileResponse:
        """Zip every kept play clip so a team manager can take the cut film."""

        try:
            saved = catalog.get_project(project_id)
        except CatalogError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        if saved is None:
            raise HTTPException(status_code=404, detail="That analysis is no longer available.")
        plays = _report_plays(saved)
        if not plays:
            raise HTTPException(status_code=400, detail="No plays to export.")
        packed = _pack_play_clips(catalog, saved, plays)
        return FileResponse(
            packed,
            media_type="application/zip",
            filename=_export_filename(saved.name, "plays.zip"),
            background=BackgroundTask(_remove_temp, packed),
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
            lineup=draw_lineup,
            portrait=draw_portrait,
            still=grab,
            resume_scan=resume_scan,
            resume_at=saved.scanned_seconds if resume_scan else 0.0,
        )
        detail = catalog.get_project(project_id)
        if detail is None:
            raise HTTPException(status_code=502, detail="Could not save that analysis.")
        return _project_payload(detail, running=True, catalog=catalog)

    @app.get("/projects/{project_id}/poster", response_model=None)
    def project_poster(project_id: int) -> FileResponse | RedirectResponse:
        """Serve a saved still. Missing posters fall back to YouTube art."""

        try:
            path = catalog.poster_file(project_id)
        except CatalogError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        if path is not None:
            return FileResponse(
                path,
                media_type="image/jpeg",
                headers={"Cache-Control": "public, max-age=86400"},
            )
        try:
            saved = catalog.get_project(project_id)
        except CatalogError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        if saved is None or saved.status != "ready" or not saved.plays:
            raise HTTPException(status_code=404, detail="That picture is not available.")
        fallback = thumbnail_url(saved.youtube_url)
        if fallback:
            return RedirectResponse(
                fallback,
                status_code=302,
                headers={"Cache-Control": "public, max-age=300"},
            )
        raise HTTPException(status_code=404, detail="That picture is not available.")

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
        marked = {
            "scrimmage": "Marked as a scrimmage play.",
            "special_teams": "Marked as special teams.",
            "junk": "Marked as not a play.",
        }.get(decision.decision, "Marked.")
        _remember_edit(
            catalog,
            saved,
            None,
            "mark",
            f"Uncertain stretch {span.start_seconds:.1f}s–{span.end_seconds:.1f}s.",
            marked,
        )
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
        payload = _play_payload(saved, play_number, catalog)
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
        if saved is None:
            raise HTTPException(status_code=404, detail="That play is no longer available.")
        play = next((item for item in saved.plays if item.play_number == play_number), None)
        if play is None:
            raise HTTPException(status_code=404, detail="That play is no longer available.")
        try:
            if choice.play_type == "junk":
                _remember_edit(
                    catalog,
                    saved,
                    play.play_number,
                    "drop",
                    _play_edit_line(play),
                    "Removed. A coach marked this as not a play.",
                )
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
            elif choice.clear_formation:
                catalog.set_play_formation(project_id, play_number, "")
            elif choice.formation is not None:
                catalog.set_play_formation(project_id, play_number, choice.formation)
            elif choice.reviewed is not None:
                catalog.set_reviewed(project_id, play_number, choice.reviewed)
            elif choice.discarded is not None:
                catalog.set_discarded(project_id, play_number, choice.discarded)
                _remember_edit(
                    catalog,
                    saved,
                    play.play_number,
                    "discard",
                    (
                        "Left out of the scouting report."
                        if play.discarded
                        else "In the scouting report."
                    ),
                    (
                        "Left out of the scouting report."
                        if choice.discarded
                        else "Put back in the scouting report."
                    ),
                )
            saved = catalog.get_project(project_id)
            updated = None if saved is None else _play_by_number(saved, play_number)
            if saved is not None and choice.play_type not in {None, "junk"} and updated is not None:
                _remember_edit(
                    catalog,
                    saved,
                    play.play_number,
                    "retag",
                    f"Tagged {_tag_line(play)}.",
                    f"Tagged {_tag_line(updated)}.",
                )
            elif saved is not None and (choice.formation is not None or choice.clear_formation):
                before = formation_label(play.formation) or "Not set"
                after = (
                    "Not set"
                    if choice.clear_formation or updated is None or not updated.formation
                    else formation_label(updated.formation) or updated.formation
                )
                _remember_edit(
                    catalog,
                    saved,
                    play.play_number,
                    "formation",
                    f"Formation {before}.",
                    f"Formation {after}.",
                )
            elif saved is not None and choice.unit is not None:
                _remember_edit(
                    catalog,
                    saved,
                    play.play_number,
                    "unit",
                    f"Your team was on {play.unit or 'an untagged side'}.",
                    f"Your team is on {choice.unit}.",
                )
        except CatalogError as error:
            status = 404 if str(error) == "That play is no longer available." else 502
            raise HTTPException(status_code=status, detail=str(error)) from error
        if saved is None:
            raise HTTPException(status_code=502, detail="Could not save that play.")
        payload = _play_payload(saved, play_number, catalog)
        if payload is None:
            raise HTTPException(status_code=404, detail="That play is no longer available.")
        return payload

    @app.post("/api/plays/{uid}/formation")
    def detect_play_formation(uid: str) -> dict[str, object]:
        """Plot both teams and guess the formation. A person can still override it."""

        parsed = _parse_play_uid(uid)
        if parsed is None:
            raise HTTPException(status_code=404, detail="That play is no longer available.")
        project_id, play_number = parsed
        try:
            saved = catalog.get_project(project_id)
            teams = catalog.film_teams(project_id)
        except CatalogError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        if saved is None:
            raise HTTPException(status_code=404, detail="That play is no longer available.")
        if teams is None or teams.mine not in {"left", "right"}:
            raise HTTPException(
                status_code=400,
                detail="Label your team before reading formations.",
            )
        play = _play_by_number(saved, play_number)
        if play is None:
            raise HTTPException(status_code=404, detail="That play is no longer available.")
        if play.family == "special_teams":
            raise HTTPException(status_code=400, detail="Formations are for scrimmage plays.")
        call = (play.play_type or "").strip().lower()
        if call not in {"run", "pass"}:
            raise HTTPException(
                status_code=400,
                detail="Chart the play as a run or pass before reading the formation.",
            )
        mine, other = _team_uniforms(teams)
        _chart_formation_play(
            catalog,
            saved,
            play,
            mine,
            other,
            read_formation_play,
            force=True,
        )
        saved = _require_project(catalog, project_id)
        payload = _play_payload(saved, play_number, catalog)
        if payload is None:
            raise HTTPException(status_code=404, detail="That play is no longer available.")
        return payload

    @app.post("/api/plays/{uid}/trim")
    def trim_play(uid: str, choice: _PlayTrim) -> dict[str, object]:
        """Recut the play. The window may reach past the cut when the film was cut too soon."""

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
        play = next((item for item in saved.plays if item.play_number == play_number), None)
        if play is None:
            raise HTTPException(status_code=404, detail="That play is no longer available.")
        start = round(choice.start, 3)
        end = round(choice.end, 3)
        earliest, latest = _trim_room(saved, play)
        if start < earliest - 0.05 or end > latest + 0.05:
            raise HTTPException(
                status_code=400,
                detail="Keep the new length beside this play.",
            )
        start = max(earliest, start)
        end = min(latest, end)
        if end - start < 0.5:
            raise HTTPException(
                status_code=400,
                detail="Keep at least half a second of the play.",
            )
        if abs(start - play.start_seconds) < 0.05 and abs(end - play.end_seconds) < 0.05:
            payload = _play_payload(saved, play_number, catalog)
            if payload is None:
                raise HTTPException(status_code=404, detail="That play is no longer available.")
            return payload
        film = catalog.source_file(project_id)
        clip = catalog.clip_file(project_id, play_number)
        extends = start < play.start_seconds - 0.05 or end > play.end_seconds + 0.05
        if film is None and clip is None:
            raise HTTPException(status_code=400, detail="That film is no longer available.")
        if extends and film is None:
            raise HTTPException(
                status_code=400,
                detail="The full film is needed to lengthen this play.",
            )
        folder = Path(tempfile.mkdtemp(prefix="film-trim-"))
        dest = folder / "clip.mp4"
        try:
            if film is not None:
                split(film, start, end, dest)
            else:
                assert clip is not None
                split(clip, start - play.start_seconds, end - play.start_seconds, dest)
            catalog.set_play_bounds(project_id, play_number, start, end, dest)
            _drop_play_read(catalog, project_id, play_number)
            saved = catalog.get_project(project_id)
        except (VideoAnalysisError, CatalogError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        finally:
            shutil.rmtree(folder, ignore_errors=True)
        if saved is None:
            raise HTTPException(status_code=502, detail="Could not save that play.")
        longer = (end - start) > (play.end_seconds - play.start_seconds) + 0.2
        shorter = (end - start) < (play.end_seconds - play.start_seconds) - 0.2
        verb = "Lengthened" if longer else "Shortened" if shorter else "Moved"
        _remember_edit(
            catalog,
            saved,
            play.play_number,
            "trim",
            f"Cut from {play.start_seconds:.1f}s to {play.end_seconds:.1f}s.",
            f"{verb} to {start:.1f}s–{end:.1f}s.",
        )
        payload = _play_payload(saved, play_number, catalog)
        if payload is None:
            raise HTTPException(status_code=404, detail="That play is no longer available.")
        return payload

    @app.get("/projects/{project_id}/plays/{play_number}")
    def play(project_id: int, play_number: int) -> FileResponse:
        path = catalog.clip_file(project_id, play_number)
        if path is None:
            raise HTTPException(status_code=404, detail="That play is no longer available.")
        return FileResponse(path, media_type="video/mp4", content_disposition_type="inline")

    @app.get("/projects/{project_id}/plays/{play_number}/formation.jpg")
    def formation_plot(project_id: int, play_number: int) -> FileResponse:
        path = catalog.play_image(project_id, play_number, "formation")
        if path is None:
            raise HTTPException(status_code=404, detail="That formation plot is not ready.")
        return FileResponse(path, media_type="image/jpeg", content_disposition_type="inline")

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
                lineup=draw_lineup,
                portrait=draw_portrait,
                still=grab,
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
            lineup=draw_lineup,
            portrait=draw_portrait,
            still=grab,
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


def _start_report(
    running: set[int],
    lock: threading.Lock,
    report_events: dict[int, queue.Queue[str | None]],
    **kwargs: object,
) -> None:
    """Chart the scouting report on a daemon thread so the page can leave and return."""

    project_id = int(kwargs["saved"].project)  # type: ignore[attr-defined]
    events: queue.Queue[str | None] = kwargs.pop("events")  # type: ignore[assignment]

    def run() -> None:
        try:
            for event in _live_lines(**kwargs):  # type: ignore[arg-type]
                events.put(event)
        except Exception:
            _LOG.exception("scouting report %s failed", project_id)
            with contextlib.suppress(CatalogError):
                catalog = kwargs["catalog"]
                catalog.record_activity(project_id, "stopped", running=False)  # type: ignore[union-attr]
        finally:
            events.put(None)
            with lock:
                if report_events.get(project_id) is events:
                    report_events.pop(project_id, None)
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
    lineup: Callable[[Path, float], tuple[bytes, bytes]] = split_lineup,
    portrait: Callable[..., tuple[bytes, str]] = _film_player,
    still: _Still = still_frame,
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

        retries = 0
        retries_done = 0

        def progress(seconds: float) -> None:
            nonlocal scanned
            scanned = seconds
            _progress(
                catalog,
                project_id,
                "scanning",
                length,
                seconds,
                None,
                retries=retries,
                retries_done=retries_done,
            )

        def on_counts(found: int, finished: int) -> None:
            nonlocal retries, retries_done
            retries = found
            retries_done = finished
            _progress(
                catalog,
                project_id,
                "scanning",
                length,
                scanned,
                None,
                retries=retries,
                retries_done=retries_done,
            )

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
        _progress(
            catalog,
            project_id,
            "scanning",
            length,
            resume_at,
            None,
            retries=0 if resume_at <= 0 else None,
            retries_done=0 if resume_at <= 0 else None,
        )
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
            usage: object = None,
        ) -> None:
            priced = _priced_read(usage)
            if priced is not None and not reason:
                try:
                    catalog.add_read(
                        project_id,
                        ReadNote(start, end, frames, nbytes, "", *priced),
                    )
                except CatalogError:
                    return
                return
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
            on_counts=on_counts,
            on_cut=lambda: _store_action_poster(catalog, project_id, path, still, force=True),
        )
        _store_action_poster(catalog, project_id, path, still, force=True)
        start_team_analysis(catalog, project_id, lineup, portrait)
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
    retries: int | None = None,
    retries_done: int | None = None,
) -> None:
    try:
        catalog.set_progress(
            project_id,
            status=status,
            duration_seconds=duration_seconds,
            scanned_seconds=scanned_seconds,
            error=error,
            retries=retries,
            retries_done=retries_done,
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
        formations_on = catalog.formation_reads()
    except CatalogError:
        workers = slots.get()
        formations_on = True
    return {
        "workers": workers,
        "place": place,
        "cloud": place == "cloud",
        "formations": formations_on,
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


_LIVE_PHASES = frozenset(
    {"starting", "downloading", "saving", "scanning", "checking", "charting"}
)


def _activity_state(phase: str, running: bool, active: bool) -> tuple[str, bool]:
    """Keep a finished film complete. A live phase that is no longer running has stopped."""

    if phase == "ready":
        return "ready", False
    if (running or active) and phase in _LIVE_PHASES:
        return phase, True
    return "stopped", False


def _thumbnail_for(
    project: ProjectDetail | ProjectSummary,
    catalog: Catalog | None = None,
    still: _Still = still_frame,
    *,
    make: bool = False,
) -> str | None:
    """YouTube art until the first cut has saved an action still from the film.

    A finished film's list row points at the poster without probing storage, so
    a long library stays fast. While the second look is still running, the
    saved still replaces the YouTube image on the film page and in the list.
    """

    count = project.play_count if isinstance(project, ProjectSummary) else len(project.plays)
    youtube = thumbnail_url(project.youtube_url)
    if count <= 0:
        return youtube
    poster = f"/projects/{project.project}/poster"
    if project.status == "ready":
        if catalog is None:
            return youtube
        if make:
            try:
                if catalog.poster_file(project.project) is None:
                    _store_action_poster(catalog, project.project, None, still)
                if catalog.poster_file(project.project) is not None:
                    return poster
            except CatalogError:
                return youtube
            return youtube
        return poster
    if catalog is None:
        return youtube
    try:
        if catalog.poster_file(project.project) is not None:
            return poster
    except CatalogError:
        return youtube
    return youtube


def _store_action_poster(
    catalog: Catalog,
    project_id: int,
    video: Path | None = None,
    grab: _Still = still_frame,
    *,
    force: bool = False,
) -> None:
    """Save one action frame from the film. A failed attempt waits before trying again."""

    key = (id(catalog), project_id)
    now = time.monotonic()
    if not force and _poster_wait.get(key, 0) > now:
        return
    try:
        with _poster_lock:
            if catalog.poster_file(project_id) is not None:
                return
            saved = catalog.get_project(project_id)
            if saved is None or not saved.plays:
                return
            jpeg = _capture_poster(catalog, saved, grab, video)
            if jpeg is None:
                _poster_wait[key] = time.monotonic() + 30
                return
            catalog.save_poster(project_id, jpeg)
            _poster_wait.pop(key, None)
    except (CatalogError, VideoAnalysisError, OSError):
        _poster_wait[key] = time.monotonic() + 30


def _capture_poster(
    catalog: Catalog,
    project: ProjectDetail,
    grab: _Still,
    video: Path | None,
) -> bytes | None:
    """Frames from the full film when it is local, otherwise from the play clips."""

    picks = action_moments(project.plays)
    if not picks:
        return None
    use_film = video is not None and video.is_file()
    by_number = {play.play_number: play for play in project.plays}
    frames: list[bytes] = []
    for number, offset in picks:
        play = by_number.get(number)
        if play is None:
            continue
        if use_film:
            target: Path | None = video
            at = play.start_seconds + offset
        else:
            try:
                target = catalog.clip_file(project.project, number)
            except CatalogError:
                continue
            at = offset
        if target is None:
            continue
        try:
            frames.append(grab(target, at))
        except (VideoAnalysisError, OSError):
            continue
    return choose_action_shot(frames)


def _activity_job(
    project: ProjectDetail | ProjectSummary,
    *,
    phase: str,
    running: bool,
    updated: str,
    steps: tuple[ActivityStep, ...] = (),
    play_costs: list[tuple[int, float]] | None = None,
    spend: tuple[float, int, int, str] | None = None,
    catalog: Catalog | None = None,
) -> dict[str, object]:
    cost, incoming, outgoing, model = spend or (0.0, 0, 0, "")
    play_count = (
        project.play_count if isinstance(project, ProjectSummary) else len(project.plays)
    )
    report_done, report_total = (
        _report_counts(catalog, project.project) if phase == "charting" else (0, 0)
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
        "retries": project.retries,
        "retries_done": project.retries_done,
        "processing": processing_percent(
            project.duration_seconds,
            project.scanned_seconds,
            project.retries,
            project.retries_done,
        ),
        "error": project.error,
        "youtube": project.youtube_url is not None,
        "play_count": play_count,
        "report_done": report_done,
        "report_total": report_total,
        "review_count": _review_count(project),
        "thumbnail": _thumbnail_for(project, catalog),
        "created": getattr(project, "created_at", None),
        "updated": updated,
        "steps": _step_times(steps),
        "play_costs": [
            {"play": number, "usd": round(usd, 6)} for number, usd in play_costs or []
        ],
        "read_cost": round(float(cost), 6),
        "read_model": model,
        "input_tokens": incoming,
        "output_tokens": outgoing,
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


def _report_plays(project: ProjectDetail) -> list[PlayRecord]:
    """Plays the scouting report should read. Discarded plays stay on the film."""

    return [play for play in _ordered_plays(project) if not play.discarded]


def _export_filename(title: str, suffix: str) -> str:
    """A safe attachment name from the film title."""

    cleaned = "".join(ch if ch.isalnum() or ch in "-_ " else " " for ch in (title or "film"))
    stem = "-".join(part for part in cleaned.split() if part) or "film"
    return f"{stem[:72]}-{suffix}"


def _remove_temp(path: Path) -> None:
    path.unlink(missing_ok=True)


def _pack_play_clips(catalog: Catalog, project: ProjectDetail, plays: list[PlayRecord]) -> Path:
    """Write kept clips and a small sheet into one zip file."""

    descriptor, name = tempfile.mkstemp(prefix="plays-", suffix=".zip")
    os.close(descriptor)
    packed = Path(name)
    sheet = io.StringIO()
    writer = csv.writer(sheet)
    writer.writerow(["play", "start", "end", "family", "call", "unit", "formation", "file"])
    packed_any = False
    try:
        with zipfile.ZipFile(packed, "w", compression=zipfile.ZIP_STORED) as archive:
            for play in plays:
                clip = catalog.clip_file(project.project, play.play_number)
                if clip is None or not clip.is_file():
                    continue
                name = f"play-{play.play_number:03d}.mp4"
                archive.write(clip, arcname=name)
                writer.writerow(
                    [
                        play.play_number,
                        format_clock(play.start_seconds),
                        format_clock(play.end_seconds),
                        play.family or "",
                        play.play_type or "",
                        play.unit or "",
                        play.formation or "",
                        name,
                    ]
                )
                packed_any = True
            if not packed_any:
                raise HTTPException(status_code=400, detail="No play clips are ready to export.")
            archive.writestr("plays.csv", sheet.getvalue())
    except Exception:
        packed.unlink(missing_ok=True)
        raise
    return packed


def _remember_edit(
    catalog: Catalog,
    project: ProjectDetail,
    play_number: int | None,
    action: str,
    before: str,
    after: str,
) -> None:
    if before == after:
        return
    with contextlib.suppress(CatalogError):
        catalog.add_coach_edit(project.project, play_number, project.name, action, before, after)


def _tag_line(play: PlayRecord) -> str:
    side = f", {play.unit}" if play.unit else ""
    return f"{play.play_type or play.family}{side}"


def _play_edit_line(play: PlayRecord) -> str:
    return (
        f"Play {play.play_number}, {_tag_line(play)}, "
        f"{play.start_seconds:.1f}s–{play.end_seconds:.1f}s."
    )


def _edit_view(edit: CoachEdit) -> dict[str, object]:
    return {
        "id": edit.edit_id,
        "project": edit.project_id,
        "play": edit.play_number,
        "film": edit.film,
        "action": edit.action,
        "before": edit.before,
        "after": edit.after,
        "at": edit.at,
        "learned": edit.learned,
    }


_TRIM_PAD = 12.0


def _trim_room(project: ProjectDetail, play: PlayRecord) -> tuple[float, float]:
    """How far a play can be pulled open around its current cut.

    Twelve seconds on either side, stopping at the film edges and at the
    next or previous play so two clips do not cover the same moment.
    """

    ordered = _ordered_plays(project)
    index = next(
        (place for place, item in enumerate(ordered) if item.play_number == play.play_number),
        None,
    )
    previous_end = 0.0 if index in {None, 0} else ordered[index - 1].end_seconds
    next_start = (
        None if index is None or index + 1 >= len(ordered) else ordered[index + 1].start_seconds
    )
    earliest = min(play.start_seconds, max(previous_end, play.start_seconds - _TRIM_PAD, 0.0))
    latest = play.end_seconds + _TRIM_PAD
    if project.duration_seconds and project.duration_seconds > play.end_seconds:
        latest = min(latest, project.duration_seconds)
    if next_start is not None:
        latest = min(latest, next_start)
    latest = max(latest, play.end_seconds)
    return round(earliest, 3), round(latest, 3)


def _play_summary(
    project: ProjectDetail,
    play: PlayRecord,
    catalog: Catalog | None = None,
) -> dict[str, object]:
    named = formation_label(play.formation) or None
    room_start, room_end = _trim_room(project, play)
    has_plot = False
    if catalog is not None:
        with contextlib.suppress(CatalogError):
            path = catalog.play_image(project.project, play.play_number, "formation")
            has_plot = path is not None
    return {
        "uid": _play_uid(project.project, play.play_number),
        "play_number": play.play_number,
        "label": format_play(play.play_number, play.start_seconds, play.end_seconds),
        "start": play.start_seconds,
        "end": play.end_seconds,
        "room_start": room_start,
        "room_end": room_end,
        "family": play.family,
        "play_type": play.play_type,
        "unit": play.unit,
        "formation": named,
        "formation_key": play.formation,
        "formation_plot": (
            f"/projects/{project.project}/plays/{play.play_number}/formation.jpg"
            if has_plot
            else None
        ),
        "formation_choices": [
            {"key": key, "label": label} for key, label in FORMATIONS.items()
        ],
        "reviewed": play.reviewed,
        "discarded": play.discarded,
        "video": f"/projects/{project.project}/plays/{play.play_number}",
    }


def _play_payload(
    project: ProjectDetail,
    play_number: int,
    catalog: Catalog | None = None,
) -> dict[str, object] | None:
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
        **_play_summary(project, play, catalog),
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


def _chart_play(
    catalog: Catalog,
    project: ProjectDetail,
    play: PlayRecord,
    mine: Uniform,
    other: Uniform,
    reader: Callable[[Sequence[bytes], Uniform, Uniform, str], ChartedPlay | str | None],
    kinds: frozenset[str] | None,
) -> tuple[int, int, str]:
    """Add the call. Returns tokens and model used for this read."""

    kind = "special_teams" if play.family == "special_teams" else "scrimmage"
    if kinds is not None and kind not in kinds:
        return 0, 0, ""
    allowed = SPECIAL_CALLS if kind == "special_teams" else {"run", "pass"}
    needs_call = play.play_type not in allowed
    needs_side = kind == "scrimmage" and play.unit not in {"offense", "defense"}
    if not needs_call and not needs_side:
        return 0, 0, ""
    call, side, incoming, outgoing, model = _chart_answer(
        reader(_scrimmage_frames(catalog, project, play), mine, other, kind)
    )
    try:
        if needs_call and call in allowed:
            catalog.set_play_type(project.project, play.play_number, call, kind)
        if needs_side and side in {"offense", "defense"}:
            catalog.set_play_unit(project.project, play.play_number, side, mark_reviewed=False)
    except CatalogError:
        return incoming, outgoing, model
    return incoming, outgoing, model


def _chart_answer(value: object) -> tuple[str | None, str | None, int, int, str]:
    if isinstance(value, str):
        return value, None, 0, 0, ""
    call = getattr(value, "call", None)
    side = getattr(value, "side", None)
    incoming = int(getattr(value, "input_tokens", 0) or 0)
    outgoing = int(getattr(value, "output_tokens", 0) or 0)
    model = str(getattr(value, "model", "") or "")
    if not isinstance(call, str):
        return None, None, incoming, outgoing, model
    if side not in {"offense", "defense"}:
        side = None
    return call, side, incoming, outgoing, model


def _chart_formation_play(
    catalog: Catalog,
    project: ProjectDetail,
    play: PlayRecord,
    mine: Uniform,
    other: Uniform,
    reader: Callable[..., FormationRead | str | None],
    *,
    force: bool = False,
) -> tuple[int, int, str]:
    """Plot the snap and name the formation after the run/pass call is known.

    Both offenses are read, so each team's report can group its snaps by
    formation. The side tells whose uniform is on offense. The scouted team's
    jersey is drawn primary on the plot.
    """

    call = (play.play_type or "").strip().lower()
    if play.family == "special_teams" or call not in {"run", "pass"}:
        return 0, 0, ""
    if not force and play.unit not in {"offense", "defense"}:
        return 0, 0, ""
    has_plot = False
    has_layout = False
    with contextlib.suppress(CatalogError):
        has_plot = catalog.play_image(project.project, play.play_number, "formation") is not None
        has_layout = catalog.play_layout(project.project, play.play_number) is not None
    # Skip only when the name, plot, and layout are already on the play.
    if not force and play.formation and has_plot and has_layout:
        return 0, 0, ""
    frames = _scrimmage_frames(catalog, project, play, snap=True)
    offense, defense = (other, mine) if play.unit != "offense" else (mine, other)
    # Scouted team is ``mine`` — emphasize their jersey on the graph.
    focus = "offense" if play.unit == "offense" else "defense"
    answer = _call_formation_reader(reader, frames, offense, defense, call, focus=focus)
    named, incoming, outgoing, model, plot, layout = _formation_answer(answer)
    try:
        if plot:
            catalog.save_play_image(project.project, play.play_number, "formation", plot)
        if layout:
            catalog.save_play_layout(project.project, play.play_number, layout)
        if named in FORMATIONS:
            catalog.set_play_formation(project.project, play.play_number, named)
        elif (force or not play.formation) and not named:
            # Keep an empty guess so the UI shows the plot for a human pick.
            catalog.set_play_formation(project.project, play.play_number, "")
    except CatalogError:
        return incoming, outgoing, model
    return incoming, outgoing, model


def _call_formation_reader(
    reader: Callable[..., FormationRead | str | None],
    frames: Sequence[bytes],
    offense: Uniform,
    defense: Uniform,
    call: str,
    *,
    focus: str = "offense",
) -> FormationRead | str | None:
    try:
        signature = inspect.signature(reader)
    except (TypeError, ValueError):
        signature = None
    if signature is None:
        return reader(frames, offense, defense)
    params = signature.parameters
    if "focus" in params:
        return reader(frames, offense, defense, call, focus=focus)
    if len(params) >= 4:
        return reader(frames, offense, defense, call)
    return reader(frames, offense, defense)


def _formation_answer(
    value: object,
) -> tuple[str | None, int, int, str, bytes, dict[str, object] | None]:
    if isinstance(value, str):
        return value, 0, 0, "", b"", None
    named = getattr(value, "formation", None)
    incoming = int(getattr(value, "input_tokens", 0) or 0)
    outgoing = int(getattr(value, "output_tokens", 0) or 0)
    model = str(getattr(value, "model", "") or "")
    plot = getattr(value, "plot", b"")
    layout = getattr(value, "layout", None)
    if not isinstance(plot, (bytes, bytearray)):
        plot = b""
    if not isinstance(layout, dict):
        layout = None
    if not isinstance(named, str):
        return None, incoming, outgoing, model, bytes(plot), layout
    return named, incoming, outgoing, model, bytes(plot), layout


def _team_uniforms(teams: FilmTeams) -> tuple[Uniform, Uniform]:
    """The user's uniform first, then the other team's."""

    pair = decode_uniforms(teams.uniforms)
    if pair is None:
        pair = (
            Uniform(teams.left_name or "Home", "white", "white", "white"),
            Uniform(teams.right_name or "Away", "dark blue", "white", "dark blue"),
        )
    # Name each uniform for its team so the reader can match the scorebug.
    left = replace(pair[0], name=teams.left_name or pair[0].name)
    right = replace(pair[1], name=teams.right_name or pair[1].name)
    if teams.mine == "right":
        return right, left
    return left, right


def _scrimmage_frames(
    catalog: Catalog, project: ProjectDetail, play: PlayRecord, *, snap: bool = False
) -> list[bytes]:
    """Frames for one play. With `snap`, four larger frames from the first few
    seconds, where the offense is still in its formation."""

    try:
        clip = catalog.clip_file(project.project, play.play_number)
    except CatalogError:
        return []
    if clip is None or not clip.is_file():
        return []
    length = max(0.5, play.end_seconds - play.start_seconds)
    try:
        # Six frames spread from the snap to the whistle. One every two seconds
        # left short plays with two or three frames, and none under two seconds.
        if snap:
            # Formations need the players large enough to tell a back from a slot.
            return sample_frames(clip, min(length, 2.5), 4, width=1280)
        return sample_frames(clip, length, 6)
    except (VideoAnalysisError, OSError):
        return []


def _report_play_numbers(body: _ReportChoice | None) -> list[int]:
    """Play numbers named for another try. An empty list means every unfinished play."""

    if body is None:
        return []
    numbers: list[int] = []
    for number in body.plays:
        if number > 0 and number not in numbers:
            numbers.append(number)
    return numbers


def _drop_play_read(catalog: Catalog, project_id: int, play_number: int) -> None:
    """Forget one play's report row after its clip changes. The other plays stay."""

    try:
        saved = {row.play_number for row in catalog.list_report_analysis(project_id)}
    except CatalogError:
        return
    if play_number not in saved:
        return
    with contextlib.suppress(CatalogError):
        catalog.set_report_analysis(project_id, ReportAnalysis(play_number, "pending"))


def _seed_report_plays(
    catalog: Catalog,
    project_id: int,
    ordered: list[PlayRecord],
    *,
    restart: bool,
    only: Sequence[int] | None = None,
) -> list[PlayRecord]:
    """Seed the analysis table and return the plays that still need charting.

    A restart reads every play again. Otherwise a finished play stays, and a
    named play is tried again without clearing the others.
    """

    if restart:
        catalog.clear_report_analysis(project_id)
        for play in ordered:
            catalog.set_report_analysis(
                project_id, ReportAnalysis(play.play_number, "pending")
            )
        return list(ordered)

    wanted = {number for number in (only or []) if number > 0}
    existing = {
        row.play_number: row for row in catalog.list_report_analysis(project_id)
    }
    todo: list[PlayRecord] = []
    for play in ordered:
        if wanted and play.play_number not in wanted:
            continue
        row = existing.get(play.play_number)
        if row is not None and row.status == "ready" and play.play_number not in wanted:
            continue
        catalog.set_report_analysis(
            project_id, ReportAnalysis(play.play_number, "pending")
        )
        todo.append(play)
    return todo


def _live_lines(
    catalog: Catalog,
    saved: ProjectDetail,
    teams: FilmTeams,
    chosen: frozenset[str],
    reader: Callable[[Sequence[bytes], Uniform, Uniform, str], ChartedPlay | str | None],
    formations: Callable[..., FormationRead | str | None],
    slots: WorkerLimit,
    *,
    restart: bool = False,
    only: Sequence[int] | None = None,
) -> Iterator[str]:
    """Write the scouting-analysis table, chart plays in parallel, and stream the rows."""

    kinds = chart_kinds(chosen)
    project_id = saved.project
    ordered = [play for play in _report_plays(saved) if _play_kind(play) in kinds]
    workers = slots.get()
    try:
        todo = _seed_report_plays(
            catalog, project_id, ordered, restart=restart, only=only
        )
        catalog.record_activity(
            project_id, "charting", running=True, stack="local", workers=workers
        )
    except CatalogError:
        todo = list(ordered)
    total = len(ordered)
    current = saved
    done = total - len(todo)
    yield _live_event(catalog, current, teams, chosen, done, total, workers)
    if not todo:
        with contextlib.suppress(CatalogError):
            catalog.record_activity(project_id, "ready", running=False)
        yield json.dumps(_report_payload(current, teams, _report_plays(current), chosen)) + "\n"
        return

    mine, other = _team_uniforms(teams)
    updates: queue.Queue[BaseException | None] = queue.Queue()
    remaining = len(todo)

    def work(play: PlayRecord) -> None:
        with slots:
            with contextlib.suppress(CatalogError):
                catalog.set_report_analysis(
                    project_id, ReportAnalysis(play.play_number, "reading")
                )
            updates.put(None)
            try:
                _analyze_report_play(
                    catalog,
                    project_id,
                    play,
                    mine,
                    other,
                    reader,
                    formations,
                    kinds,
                    chosen,
                )
            except Exception as error:
                with contextlib.suppress(CatalogError):
                    catalog.set_report_analysis(
                        project_id,
                        ReportAnalysis(
                            play.play_number, "missed", detail=str(error)[:200]
                        ),
                    )
            updates.put(None)

    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            jobs = [pool.submit(work, play) for play in todo]
            finished = 0
            while finished < remaining:
                error = updates.get()
                if error is not None:
                    raise error
                rows = []
                with contextlib.suppress(CatalogError):
                    rows = catalog.list_report_analysis(project_id)
                finished = sum(
                    1
                    for row in rows
                    if row.status in {"ready", "missed"}
                    and any(play.play_number == row.play_number for play in todo)
                )
                current = _require_project(catalog, project_id)
                yield _live_event(
                    catalog,
                    current,
                    teams,
                    chosen,
                    sum(
                        1
                        for row in rows
                        if row.status in {"ready", "missed"}
                        and any(play.play_number == row.play_number for play in ordered)
                    ),
                    total,
                    slots.get(),
                )
            for job in jobs:
                job.result()
        with contextlib.suppress(CatalogError):
            catalog.record_activity(project_id, "ready", running=False)
        current = _require_project(catalog, project_id)
        yield json.dumps(_report_payload(current, teams, _report_plays(current), chosen)) + "\n"
    except Exception:
        with contextlib.suppress(CatalogError):
            catalog.record_activity(project_id, "stopped", running=False)
        raise


def _follow_report(
    catalog: Catalog,
    project_id: int,
    teams: FilmTeams,
    chosen: frozenset[str],
    running: set[int],
    lock: threading.Lock,
    report_events: dict[int, queue.Queue[str | None]],
    slots: WorkerLimit,
) -> Iterator[str]:
    """Stream progress from the durable chart job until it finishes."""

    with lock:
        events = report_events.get(project_id)
    if events is not None:
        while True:
            event = events.get()
            if event is None:
                return
            yield event

    last = ""
    for _ in range(100):
        try:
            current = _require_project(catalog, project_id)
            kept = {play.play_number for play in _report_plays(current)}
            rows = [
                row
                for row in catalog.list_report_analysis(project_id)
                if row.play_number in kept
            ]
            activity = catalog.get_activity(project_id)
        except (CatalogError, HTTPException):
            return
        done = sum(1 for row in rows if row.status in {"ready", "missed"})
        total = len(rows) or len(
            [
                play
                for play in _report_plays(current)
                if _play_kind(play) in chart_kinds(chosen)
            ]
        )
        workers = slots.get()
        if activity is not None and activity.steps:
            step_workers = activity.steps[-1].workers
            if step_workers is not None:
                workers = step_workers
        event = _live_event(catalog, current, teams, chosen, done, total, workers)
        if event != last:
            yield event
            last = event
        live = _running(running, lock, project_id) or bool(
            activity and activity.running and activity.phase == "charting"
        )
        unfinished = any(row.status in {"pending", "reading"} for row in rows)
        if not live and not unfinished:
            yield json.dumps(
                _report_payload(current, teams, _report_plays(current), chosen)
            ) + "\n"
            return
        if not live:
            yield json.dumps(
                _report_payload(current, teams, _report_plays(current), chosen)
            ) + "\n"
            return
        time.sleep(0.1)


def _analyze_report_play(
    catalog: Catalog,
    project_id: int,
    play: PlayRecord,
    mine: Uniform,
    other: Uniform,
    reader: Callable[[Sequence[bytes], Uniform, Uniform, str], ChartedPlay | str | None],
    formations: Callable[..., FormationRead | str | None],
    kinds: frozenset[str],
    chosen: frozenset[str],
    *,
    phase: str = "all",
) -> None:
    """Chart one play. ``phase`` is call, formation, or both."""

    try:
        project = _require_project(catalog, project_id)
        latest = _play_by_number(project, play.play_number) or play
        incoming = 0
        outgoing = 0
        model = ""
        if phase in {"all", "call"}:
            more_in, more_out, model = _chart_play(
                catalog, project, latest, mine, other, reader, kinds
            )
            incoming += more_in
            outgoing += more_out
            project = _require_project(catalog, project_id)
            latest = _play_by_number(project, play.play_number) or latest
        want_formation = "formations" in chosen
        if not want_formation:
            with contextlib.suppress(CatalogError):
                want_formation = catalog.formation_reads()
        if want_formation and phase in {"all", "formation"}:
            more_in, more_out, formed_model = _chart_formation_play(
                catalog,
                project,
                latest,
                mine,
                other,
                formations,
                force=phase == "formation",
            )
            incoming += more_in
            outgoing += more_out
            model = model or formed_model
        project = _require_project(catalog, project_id)
        latest = _play_by_number(project, play.play_number) or latest
        prior = None
        with contextlib.suppress(CatalogError):
            prior = next(
                (
                    row
                    for row in catalog.list_report_analysis(project_id)
                    if row.play_number == play.play_number
                ),
                None,
            )
        cost = read_cost(model, incoming, outgoing) + (prior.cost_usd if prior else 0.0)
        learned = bool(latest.play_type or latest.unit)
        has_plot = False
        with contextlib.suppress(CatalogError):
            has_plot = (
                catalog.play_image(project_id, play.play_number, "formation") is not None
            )
        detail = ""
        status = "ready" if learned else "missed"
        if learned and "formations" in chosen and phase in {"all", "formation"}:
            call = (latest.play_type or "").strip().lower()
            if (
                latest.family != "special_teams"
                and call in {"run", "pass"}
                and not has_plot
            ):
                status = "missed"
                detail = "Could not plot the starting formation."
        if not learned:
            detail = "Could not tell whether the opponent was on offense or defense."
        catalog.set_report_analysis(
            project_id,
            ReportAnalysis(
                play.play_number,
                status,
                call=latest.play_type or "",
                side=latest.unit or "",
                formation=latest.formation or "",
                input_tokens=incoming
                + (prior.input_tokens if prior and phase == "formation" else 0),
                output_tokens=outgoing
                + (prior.output_tokens if prior and phase == "formation" else 0),
                cost_usd=cost,
                model=model or (prior.model if prior else ""),
                detail=detail,
            ),
        )
    except Exception as error:
        with contextlib.suppress(CatalogError):
            catalog.set_report_analysis(
                project_id,
                ReportAnalysis(play.play_number, "missed", detail=str(error)[:200]),
            )


def _live_event(
    catalog: Catalog,
    project: ProjectDetail,
    teams: FilmTeams,
    chosen: frozenset[str],
    done: int,
    total: int,
    workers: int = 1,
) -> str:
    kept_plays = _report_plays(project)
    kept = {play.play_number for play in kept_plays}
    snapshot = live_snapshot(teams, kept_plays, chosen)
    snapshot["done"] = done
    snapshot["total"] = total
    snapshot["workers"] = workers
    analysis: list[dict[str, object]] = []
    with contextlib.suppress(CatalogError):
        by_number = {play.play_number: play for play in kept_plays}
        analysis = [
            _analysis_view(row, project.project, by_number.get(row.play_number))
            for row in catalog.list_report_analysis(project.project)
            if row.play_number in kept
        ]
    snapshot["analysis"] = analysis
    return json.dumps(snapshot) + "\n"


def _analysis_view(
    note: ReportAnalysis,
    project_id: int | None = None,
    play: PlayRecord | None = None,
) -> dict[str, object]:
    number = note.play_number
    return {
        "play": number,
        "uid": _play_uid(project_id, number) if project_id is not None else None,
        "start": None if play is None else play.start_seconds,
        "end": None if play is None else play.end_seconds,
        "status": note.status,
        "call": note.call,
        "side": note.side,
        "formation": formation_label(note.formation) or note.formation,
        "input_tokens": note.input_tokens,
        "output_tokens": note.output_tokens,
        "cost_usd": round(note.cost_usd, 6),
        "model": note.model,
        "detail": note.detail,
    }


def _merged_spend(
    film: dict[int, tuple[float, int, int, str]],
    report: dict[int, tuple[float, int, int, str]],
) -> dict[int, tuple[float, int, int, str]]:
    merged = dict(film)
    for project_id, (cost, incoming, outgoing, model) in report.items():
        if project_id not in merged:
            merged[project_id] = (cost, incoming, outgoing, model)
            continue
        prior_cost, prior_in, prior_out, prior_model = merged[project_id]
        models = [name for name in (prior_model, model) if name]
        unique: list[str] = []
        for name in models:
            for part in name.split(", "):
                if part and part not in unique:
                    unique.append(part)
        merged[project_id] = (
            prior_cost + cost,
            prior_in + incoming,
            prior_out + outgoing,
            ", ".join(unique),
        )
    return merged


def _play_kind(play: PlayRecord) -> str:
    return "special_teams" if play.family == "special_teams" else "scrimmage"


def _play_by_number(project: ProjectDetail, play_number: int) -> PlayRecord | None:
    for play in project.plays:
        if play.play_number == play_number:
            return play
    return None


def _report_payload(
    saved: ProjectDetail,
    teams: FilmTeams,
    plays: list[PlayRecord],
    chosen: frozenset[str] | None,
) -> dict[str, object]:
    sections = report_sections(chosen)
    reads = play_reads(plays)
    blocks = scouting_blocks(saved.name, teams, plays, sections)
    tendencies = " ".join(tendency_lines(teams, plays)) if "tendencies" in sections else ""
    return {
        "project": saved.project,
        "plays": len(reads),
        "ready": True,
        "tendencies": tendencies,
        "sections": sorted(sections),
        "blocks": [{"kind": kind, "text": text} for kind, text in blocks],
    }


def _report_sections(raw: str) -> frozenset[str] | None:
    names = [part.strip() for part in raw.split(",") if part.strip()]
    if not names:
        return None
    try:
        return report_sections(names)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def _report_film(
    catalog: Catalog, project_id: int
) -> tuple[ProjectDetail, FilmTeams, list[PlayRecord]]:
    """Load a finished film and the chart stored on each play."""

    saved = _require_project(catalog, project_id)
    if saved.status != "ready" or not saved.plays:
        raise HTTPException(
            status_code=400,
            detail="The scouting report waits until the plays are identified.",
        )
    try:
        teams = catalog.film_teams(project_id)
    except CatalogError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    if teams is None or teams.mine not in {"left", "right"}:
        raise HTTPException(
            status_code=400,
            detail="Label your team before the scouting report.",
        )
    return saved, teams, _report_plays(saved)


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


def start_team_analysis(
    catalog: Catalog,
    project_id: int,
    lineup: Callable[[Path, float], tuple[bytes, bytes]] = split_lineup,
    portrait: Callable[..., tuple[bytes, str]] = _film_player,
) -> None:
    """Name the two teams once the plays exist. A miss here leaves the plays in place."""

    try:
        saved = catalog.get_project(project_id)
    except CatalogError:
        _LOG.warning("team analysis could not open film %s", project_id, exc_info=True)
        return
    if saved is None or not saved.plays:
        return
    try:
        run_team_analysis(catalog, saved, lineup, portrait)
    except Exception:
        _LOG.warning("team analysis did not finish for film %s", project_id, exc_info=True)


def run_team_analysis(
    catalog: Catalog,
    project: ProjectDetail,
    lineup: Callable[[Path, float], tuple[bytes, bytes]],
    portrait: Callable[..., tuple[bytes, str]],
    force: bool = False,
    side: str | None = None,
    attempt: int = 1,
) -> None:
    """Read the lineup, name both sides, and keep one player from each team in view."""

    if not project.plays:
        return
    project_id = project.project
    missing_lineup = _still_bytes(catalog, project_id, "left") is None or _still_bytes(
        catalog, project_id, "right"
    ) is None
    _set_team_progress(
        project_id,
        step="read",
        percent=8,
        detail="Reading the film for both uniforms.",
        attempt=attempt,
        event="Opening the film to read both uniforms.",
        plays=len(project.plays),
        sides={
            "left": {"status": "reading", "label": "Reading film"},
            "right": {"status": "reading", "label": "Reading film"},
        },
    )
    # Re-grab the lineup when regenerating so each side can get a fresh film photo.
    if missing_lineup or force:
        _store_lineup(catalog, project, lineup, attempt=attempt)
    _remember_team_names(catalog, project)
    redraw = (
        force
        or _players_missing(catalog, project_id)
        or _portraits_need_replacement(catalog, project_id)
    )
    try:
        named = catalog.film_teams(project_id)
    except CatalogError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    if named is None:
        return
    sides = (side,) if side in {"left", "right"} else ("left", "right")
    if redraw:
        _store_portraits(
            catalog,
            project_id,
            named,
            keep_current=not force,
            sides=sides,
            attempt=attempt,
        )
    elif _references_missing(catalog, project_id):
        _set_team_progress(
            project_id,
            step="save",
            percent=90,
            detail="Keeping the photos already found.",
            attempt=attempt,
        )
        _store_references(catalog, project_id, named)
    _set_team_progress(
        project_id,
        step="save",
        percent=100,
        detail="Team cards are ready.",
        attempt=attempt,
        running=False,
    )


def _store_lineup(
    catalog: Catalog,
    project: ProjectDetail,
    draw: Callable[[Path, float], tuple[bytes, bytes]],
    *,
    attempt: int = 1,
) -> None:
    project_id = project.project
    _set_team_progress(
        project_id,
        step="read",
        percent=14,
        detail="Pulling a wide look at both sides of the ball.",
        attempt=attempt,
        event="Sampling a lineup moment from the film.",
        plays=len(project.plays),
        sides={
            "left": {"status": "reading", "label": "Sampling lineup"},
            "right": {"status": "reading", "label": "Sampling lineup"},
        },
    )
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
        _set_team_progress(
            project_id,
            step="read",
            percent=22,
            detail="Uniforms are on the cards. Looking for players next.",
            attempt=attempt,
            event="Saved the lineup stills for both teams.",
            sides={
                "left": {"status": "ready", "label": "Lineup saved"},
                "right": {"status": "ready", "label": "Lineup saved"},
            },
        )
    except CatalogError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


def _remember_team_names(catalog: Catalog, project: ProjectDetail) -> None:
    try:
        saved = catalog.film_teams(project.project)
    except CatalogError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    pair = _detected_uniforms(catalog, project)
    previous = None if saved is None else decode_uniforms(saved.uniforms)
    guessed = saved is not None and previous is not None and _names_from_detection(saved, previous)
    if saved is not None and not _placeholder_names(saved) and not guessed:
        left_name, right_name, mine = saved.left_name, saved.right_name, saved.mine
    else:
        # Stable side ids stay left/right. Friendly names default to Home/Away.
        labeled = label_home_away(*pair)
        left_name, right_name = labeled[0].name, labeled[1].name
        mine = None if saved is None else saved.mine
    try:
        catalog.set_film_teams(
            project.project,
            FilmTeams(left_name, right_name, mine, encode_uniforms(*pair)),
        )
    except CatalogError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


def _detected_uniforms(catalog: Catalog, project: ProjectDetail) -> tuple[Uniform, Uniform]:
    video, _at = _lineup_video(catalog, project)
    return detect_uniforms(
        play_frames(video, project.plays),
        project.name,
        _still_bytes(catalog, project.project, "left"),
        _still_bytes(catalog, project.project, "right"),
    )


def _remember_uniforms(catalog: Catalog, project: ProjectDetail) -> None:
    _remember_team_names(catalog, project)


def _placeholder_names(teams: FilmTeams) -> bool:
    return {teams.left_name.casefold(), teams.right_name.casefold()} <= {"", "home", "away"}


def _names_from_detection(teams: FilmTeams, found: tuple[Uniform, Uniform]) -> bool:
    """True when the saved names are still the ones read off the uniforms."""

    return (
        teams.left_name.casefold() == found[0].name.casefold()
        and teams.right_name.casefold() == found[1].name.casefold()
    )


def _named_sides(catalog: Catalog, project: ProjectDetail, body: _TeamNames) -> FilmTeams:
    try:
        existing = catalog.film_teams(project.project)
    except CatalogError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    suggested = suggest_team_names(project.name)
    uniforms = "" if existing is None else existing.uniforms
    left = body.left_name or (
        existing.left_name if existing and existing.left_name else suggested[0]
    )
    right = body.right_name or (
        existing.right_name if existing and existing.right_name else suggested[1]
    )
    return FilmTeams(left, right, body.mine, uniforms)


def _players_missing(catalog: Catalog, project_id: int) -> bool:
    return _still_bytes(catalog, project_id, "left-player") is None or _still_bytes(
        catalog, project_id, "right-player"
    ) is None


def _players_saved(catalog: Catalog, project_id: int) -> bool:
    return _still_bytes(catalog, project_id, "left-player") is not None and _still_bytes(
        catalog, project_id, "right-player"
    ) is not None


def _load_project_page(
    catalog: Catalog, project_id: int
) -> tuple[ProjectDetail | None, tuple[ActivityStep, ...], FilmTeams | None, list[ReadNote]]:
    """Read the film, its steps, teams, and notes in one round of requests."""

    errors: list[CatalogError] = []

    def run(fn: Callable[[], object]) -> object:
        try:
            return fn()
        except CatalogError as error:
            errors.append(error)
            return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        project_call = pool.submit(run, lambda: catalog.get_project(project_id))
        activity_call = pool.submit(run, lambda: catalog.get_activity(project_id))
        teams_call = pool.submit(run, lambda: catalog.film_teams(project_id))
        reads_call = pool.submit(run, lambda: catalog.list_reads(project_id))
        project = project_call.result()
        activity = activity_call.result()
        teams = teams_call.result()
        reads = reads_call.result()
    if errors:
        raise errors[0]
    steps = activity.steps if isinstance(activity, ActivityRecord) else ()
    notes = (
        [note for note in reads if isinstance(note, ReadNote)] if isinstance(reads, list) else []
    )
    saved = project if isinstance(project, ProjectDetail) else None
    named = teams if isinstance(teams, FilmTeams) else None
    return saved, steps, named, notes


_portrait_refreshing: set[int] = set()
_portrait_refresh_lock = threading.Lock()
_team_builds: dict[int, dict[str, object]] = {}
_team_build_lock = threading.Lock()


def _team_progress_view(project_id: int) -> dict[str, object] | None:
    with _team_build_lock:
        current = _team_builds.get(project_id)
        if current is None:
            return None
        view = dict(current)
        started = float(view.get("started_at") or 0)
        if started:
            view["elapsed_ms"] = max(0, int((time.time() - started) * 1000))
        return view


def _set_team_progress(
    project_id: int,
    *,
    step: str,
    percent: int,
    detail: str,
    attempt: int = 1,
    running: bool = True,
    error: str | None = None,
    event: str | None = None,
    frames: int | None = None,
    plays: int | None = None,
    sides: dict[str, dict[str, str]] | None = None,
) -> None:
    with _team_build_lock:
        previous = _team_builds.get(project_id) or {}
        started = float(previous.get("started_at") or time.time())
        events = [dict(row) for row in (previous.get("events") or []) if isinstance(row, dict)]
        note = (event or detail or "").strip()
        if note:
            stamp = time.strftime("%H:%M:%S")
            if not events or events[-1].get("text") != note:
                events.append({"at": stamp, "text": note})
            events = events[-10:]
        side_state: dict[str, dict[str, str]] = {
            key: dict(value)
            for key, value in dict(previous.get("sides") or {}).items()
            if isinstance(value, dict)
        }
        if sides:
            for key, value in sides.items():
                side_state[key] = {**side_state.get(key, {}), **value}
        _team_builds[project_id] = {
            "step": step,
            "percent": max(0, min(100, percent)),
            "detail": detail,
            "attempt": max(1, attempt),
            "running": running,
            "error": error,
            "started_at": started,
            "elapsed_ms": max(0, int((time.time() - started) * 1000)),
            "frames": int(frames if frames is not None else previous.get("frames") or 0),
            "plays": int(plays if plays is not None else previous.get("plays") or 0),
            "sides": side_state,
            "events": events,
            "steps": [
                {"id": name, "label": label}
                for name, label in (
                    ("read", "Read film"),
                    ("find", "Find players"),
                    ("save", "Save cards"),
                )
            ],
        }


def _next_team_attempt(project_id: int) -> int:
    with _team_build_lock:
        current = _team_builds.get(project_id) or {}
        attempt = int(current.get("attempt") or 0) + 1
        return attempt


def _start_team_build(
    catalog: Catalog,
    project: ProjectDetail,
    lineup: Callable[[Path, float], tuple[bytes, bytes]],
    portrait: Callable[..., tuple[bytes, str]],
    *,
    force: bool = False,
    side: str | None = None,
    wait: bool = True,
) -> None:
    """Build the team cards now, or in the background so the page can show progress."""

    project_id = project.project
    with _team_build_lock:
        current = _team_builds.get(project_id)
        if not wait and current is not None and current.get("running"):
            return
    attempt = _next_team_attempt(project_id)
    play_count = len(project.plays)
    _set_team_progress(
        project_id,
        step="read",
        percent=4,
        detail="Starting the team cards.",
        attempt=attempt,
        event="Queued a fresh team-card pass.",
        plays=play_count,
        sides={
            "left": {"status": "waiting", "label": "Waiting"},
            "right": {"status": "waiting", "label": "Waiting"},
        },
    )

    def work() -> None:
        try:
            if force:
                _reset_team_photos(catalog, project_id, side=side)
            run_team_analysis(
                catalog,
                project,
                lineup,
                portrait,
                force=force,
                side=side,
                attempt=attempt,
            )
            _set_team_progress(
                project_id,
                step="save",
                percent=100,
                detail="Team cards are ready.",
                attempt=attempt,
                running=False,
            )
        except HTTPException as error:
            detail = (
                error.detail
                if isinstance(error.detail, str)
                else "Could not build the team cards."
            )
            _set_team_progress(
                project_id,
                step="save",
                percent=100,
                detail=detail,
                attempt=attempt,
                running=False,
                error=detail,
            )
        except Exception as error:
            _LOG.warning("team cards failed for film %s", project_id, exc_info=True)
            detail = str(error).strip()[:180] or "Could not build the team cards."
            _set_team_progress(
                project_id,
                step="save",
                percent=100,
                detail=detail,
                attempt=attempt,
                running=False,
                error=detail,
            )

    if wait:
        work()
        return
    threading.Thread(target=work, name=f"team-cards-{project_id}", daemon=True).start()


_team_reset: dict[str, object] = {
    "running": False,
    "attempt": 0,
    "total": 0,
    "done": 0,
    "failed": 0,
    "percent": 0,
    "detail": "",
    "current": None,
    "films": [],
}
_team_reset_lock = threading.Lock()


def _team_reset_view() -> dict[str, object]:
    with _team_reset_lock:
        return {
            "running": bool(_team_reset["running"]),
            "attempt": int(_team_reset["attempt"] or 0),
            "total": int(_team_reset["total"] or 0),
            "done": int(_team_reset["done"] or 0),
            "failed": int(_team_reset["failed"] or 0),
            "percent": int(_team_reset["percent"] or 0),
            "detail": str(_team_reset["detail"] or ""),
            "current": _team_reset["current"],
            "films": [dict(row) for row in (_team_reset["films"] or [])],  # type: ignore[union-attr]
            "steps": [
                {"id": "queue", "label": "Queue films"},
                {"id": "reset", "label": "Reset photos"},
                {"id": "rebuild", "label": "Rebuild cards"},
            ],
        }


def _set_team_reset(**fields: object) -> None:
    with _team_reset_lock:
        _team_reset.update(fields)


def _update_team_reset_film(project_id: int, **fields: object) -> None:
    with _team_reset_lock:
        films = list(_team_reset["films"] or [])  # type: ignore[arg-type]
        for index, row in enumerate(films):
            if int(row["project"]) == project_id:  # type: ignore[index]
                films[index] = {**row, **fields}  # type: ignore[arg-type]
                break
        _team_reset["films"] = films


def _team_reset_error_detail(error: BaseException) -> str:
    if isinstance(error, HTTPException):
        return (
            error.detail
            if isinstance(error.detail, str)
            else "Could not rebuild the team cards."
        )
    text = str(error).strip()
    return text[:180] if text else "Could not rebuild the team cards."


def _mirror_team_reset_film(
    project_id: int,
    name: str,
    index: int,
    total: int,
    stop: threading.Event,
) -> None:
    """Copy live card-build progress onto the global reset film row."""

    while not stop.wait(0.35):
        progress = _team_progress_view(project_id) or {}
        if not progress:
            continue
        build_pct = max(0, min(100, int(progress.get("percent") or 0)))
        detail = str(progress.get("detail") or "Finding a player from each team.")
        film_pct = 15 + round(build_pct * 0.8)
        overall = (
            round(((index + (build_pct / 100.0)) / total) * 100) if total else 100
        )
        _update_team_reset_film(
            project_id,
            status="running",
            detail=detail,
            percent=film_pct,
            step=str(progress.get("step") or "find"),
        )
        _set_team_reset(
            detail=f"{name}: {detail}",
            percent=max(0, min(99, overall)),
        )


def _ready_team_films(catalog: Catalog) -> list[ProjectDetail]:
    try:
        rows = catalog.list_projects()
    except CatalogError:
        return []
    films: list[ProjectDetail] = []
    for row in rows:
        if row.status != "ready" or row.play_count < 1 or row.archived:
            continue
        try:
            project = catalog.get_project(row.project)
        except CatalogError:
            continue
        if project is None or not project.plays:
            continue
        films.append(project)
    return films


def _start_team_reset(
    catalog: Catalog,
    lineup: Callable[[Path, float], tuple[bytes, bytes]],
    portrait: Callable[..., tuple[bytes, str]],
    *,
    failed_only: bool = False,
) -> bool:
    """Clear and rebuild team photos for every ready film, with live progress."""

    with _team_reset_lock:
        if _team_reset["running"]:
            return False
        attempt = int(_team_reset["attempt"] or 0) + 1
    films = _ready_team_films(catalog)
    if failed_only:
        with _team_reset_lock:
            failed_ids = {
                int(row["project"])  # type: ignore[index]
                for row in (_team_reset["films"] or [])  # type: ignore[union-attr]
                if row.get("status") == "failed"  # type: ignore[union-attr]
            }
        films = [film for film in films if film.project in failed_ids]
    if not films:
        return False
    queue = [
        {
            "project": film.project,
            "name": film.name,
            "status": "queued",
            "detail": "Waiting.",
            "percent": 0,
            "step": "queue",
        }
        for film in films
    ]
    _set_team_reset(
        running=True,
        attempt=attempt,
        total=len(films),
        done=0,
        failed=0,
        percent=0,
        detail=f"Resetting team photos for {len(films)} films.",
        current=None,
        films=queue,
    )

    def work() -> None:
        done = 0
        failed = 0
        total = len(films)
        for index, film in enumerate(films):
            _set_team_reset(
                current={"project": film.project, "name": film.name},
                detail=f"Resetting {film.name}.",
                percent=round((index / total) * 100) if total else 100,
            )
            _update_team_reset_film(
                film.project,
                status="running",
                detail="Clearing the old team photos.",
                percent=5,
                step="reset",
            )
            stop = threading.Event()
            mirror = threading.Thread(
                target=_mirror_team_reset_film,
                args=(film.project, film.name, index, total, stop),
                name=f"team-reset-mirror-{film.project}",
                daemon=True,
            )
            try:
                _reset_team_photos(catalog, film.project)
                _update_team_reset_film(
                    film.project,
                    status="running",
                    detail="Finding a player from each team.",
                    percent=20,
                    step="rebuild",
                )
                mirror.start()
                run_team_analysis(
                    catalog,
                    film,
                    lineup,
                    portrait,
                    force=True,
                    attempt=attempt,
                )
                progress = _team_progress_view(film.project) or {}
                if progress.get("error"):
                    raise RuntimeError(str(progress["error"]))
                done += 1
                _update_team_reset_film(
                    film.project,
                    status="ready",
                    detail="Team cards are ready.",
                    percent=100,
                    step="rebuild",
                )
            except Exception as error:
                failed += 1
                detail = _team_reset_error_detail(error)
                _LOG.warning(
                    "team photo reset failed for film %s",
                    film.project,
                    exc_info=True,
                )
                _update_team_reset_film(
                    film.project,
                    status="failed",
                    detail=detail,
                    percent=100,
                    step="rebuild",
                )
            finally:
                stop.set()
                if mirror.ident is not None:
                    mirror.join(timeout=1.0)
            _set_team_reset(
                done=done,
                failed=failed,
                percent=round(((index + 1) / total) * 100) if total else 100,
            )
        detail = (
            f"Reset {done} of {total} films."
            if failed == 0
            else f"Reset {done} of {total} films. {failed} need another try."
        )
        _set_team_reset(
            running=False,
            current=None,
            detail=detail,
            percent=100,
            done=done,
            failed=failed,
        )

    threading.Thread(target=work, name="team-photo-reset", daemon=True).start()
    return True


def _reset_team_photos(
    catalog: Catalog,
    project_id: int,
    *,
    side: str | None = None,
) -> None:
    """Remove stored player photos so the next build starts clean."""

    sides = (side,) if side in {"left", "right"} else ("left", "right")
    kinds: list[str] = []
    for chosen in sides:
        kinds.extend(
            (
                f"{chosen}-player",
                f"{chosen}-reference",
                chosen,
            )
        )
    for kind in kinds:
        try:
            catalog.clear_team_image(project_id, kind)
        except CatalogError:
            continue
    _clear_crops_checked(catalog, project_id)


def _schedule_portrait_refresh(
    catalog: Catalog,
    project: ProjectDetail,
    lineup: Callable[[Path, float], tuple[bytes, bytes]],
    portrait: Callable[..., tuple[bytes, str]],
) -> bool:
    """Replace an old player picture after the page has started to load."""

    project_id = project.project
    with _portrait_refresh_lock:
        if project_id in _portrait_refreshing:
            return True
        _portrait_refreshing.add(project_id)
    finished = threading.Event()

    def work() -> None:
        try:
            _replace_old_portraits(catalog, project, lineup, portrait)
        except Exception:
            logging.getLogger(__name__).warning(
                "Could not refresh player photos for project %s",
                project_id,
                exc_info=True,
            )
        finally:
            with _portrait_refresh_lock:
                _portrait_refreshing.discard(project_id)
            finished.set()

    threading.Thread(target=work, name=f"player-photos-{project_id}", daemon=True).start()
    finished.wait(0.05)
    with _portrait_refresh_lock:
        return project_id in _portrait_refreshing


def _replace_old_portraits(
    catalog: Catalog,
    project: ProjectDetail,
    lineup: Callable[[Path, float], tuple[bytes, bytes]],
    portrait: Callable[..., tuple[bytes, str]],
) -> ProjectDetail:
    """Swap 768 by 1280 generated players for a film shot, or one 512 by 768 player."""

    if not _portraits_need_replacement(catalog, project.project):
        return project
    try:
        run_team_analysis(catalog, project, lineup, portrait)
    except HTTPException as error:
        logging.getLogger(__name__).warning(
            "Could not replace the old player pictures for project %s: %s",
            project.project,
            error.detail,
        )
        return project
    try:
        refreshed = catalog.get_project(project.project)
    except CatalogError:
        return project
    return refreshed or project


def _portraits_need_replacement(catalog: Catalog, project_id: int) -> bool:
    checked = _crops_checked(catalog, project_id)
    for side in ("left-player", "right-player"):
        image = _still_bytes(catalog, project_id, side)
        if image is None:
            continue
        if _portrait_is_large(image) or _portrait_is_small(image):
            return True
        if not checked and _portrait_is_generated(image):
            return True
    return False


def _portrait_should_replace(data: bytes) -> bool:
    return _portrait_is_large(data) or _portrait_is_small(data)


def _portrait_is_generated(data: bytes) -> bool:
    """A drawn player is 512 by 768. A crop of a real player is a different shape."""

    return _jpeg_size(data) == (512, 768)


def _portrait_is_large(data: bytes) -> bool:
    """The previous generator saved each player at 768 by 1280."""

    size = _jpeg_size(data)
    if size is None:
        return False
    width, height = size
    return width >= 700 or height >= 1000


def _portrait_is_small(data: bytes) -> bool:
    """A crop under 90 pixels on the long side is too small to show a uniform."""

    size = _jpeg_size(data)
    if size is None:
        return False
    return max(size) < 90


def _jpeg_size(data: bytes) -> tuple[int, int] | None:
    if not data.startswith(b"\xff\xd8"):
        return None
    index = 2
    while index + 9 < len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        if marker in {0xC0, 0xC1, 0xC2}:
            height = int.from_bytes(data[index + 5 : index + 7], "big")
            width = int.from_bytes(data[index + 7 : index + 9], "big")
            if width > 0 and height > 0:
                return width, height
            return None
        if marker == 0xD9 or marker == 0xDA:
            return None
        if index + 4 > len(data):
            return None
        length = int.from_bytes(data[index + 2 : index + 4], "big")
        if length < 2:
            return None
        index += 2 + length
    return None


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


def _uniform_from_lineup_still(
    catalog: Catalog,
    project_id: int,
    side: str,
    uniform: Uniform,
) -> Uniform:
    """Prefer jersey hex measured on this side's lineup still when it is real cloth."""

    still = _still_bytes(catalog, project_id, side)
    if still is None:
        return uniform
    focused = focus_player(still, side)
    jersey_hex, _pants_hex = dominant_colors(focused)
    if _field_like_hex(jersey_hex):
        return uniform
    known = (uniform.jersey_hex or "").strip()
    if (
        known.startswith("#")
        and len(known) == 7
        and not _field_like_hex(known)
        and not _muddy_hex(known)
        and _hex_distance(known, jersey_hex) > 55 * 55 * 3
    ):
        # Keep a clear stored jersey when the still drifted to another color.
        return uniform
    words = color_words(jersey_hex)
    if words == "the color in the photo":
        words = uniform.jersey
    return replace(uniform, jersey_hex=jersey_hex, jersey=words or uniform.jersey)


def _field_like_hex(value: str) -> bool:
    """True when a measured swatch is turf, not a jersey."""

    parsed = _parse_hex(value)
    if parsed is None:
        return True
    red, green, blue = parsed
    # Olive and bright turf both read green-dominant.
    if green >= red and green >= blue and green > 55 and green - min(red, blue) >= 8:
        return True
    return max(red, green, blue) - min(red, green, blue) < 18 and max(red, green, blue) < 90


def _muddy_hex(value: str) -> bool:
    """True when a stored hex is a dull gray, not a real team color."""

    parsed = _parse_hex(value)
    if parsed is None:
        return True
    peak = max(parsed)
    span = peak - min(parsed)
    return span < 40 and 70 <= peak <= 200


def _parse_hex(value: str) -> tuple[int, int, int] | None:
    text = value.strip().lstrip("#")
    if len(text) != 6 or any(char not in "0123456789abcdefABCDEF" for char in text):
        return None
    return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))


def _hex_distance(left: str, right: str) -> int:
    left_rgb = _parse_hex(left)
    right_rgb = _parse_hex(right)
    if left_rgb is None or right_rgb is None:
        return 10**9
    return sum((a - b) ** 2 for a, b in zip(left_rgb, right_rgb, strict=True))


def _player_search_frames(catalog: Catalog, project_id: int) -> list[bytes]:
    """Near-broadcast play frames. Analysis JPEGs are too soft to crop one player."""

    try:
        project = catalog.get_project(project_id)
    except CatalogError:
        return []
    if project is None or not project.plays:
        return []
    try:
        video, _at = _lineup_video(catalog, project)
    except HTTPException:
        return []
    frames = [
        *player_action_frames(video, project.plays, limit=32),
        *player_frames(video, project.plays, limit=40),
    ]
    duration = project.duration_seconds or 0.0
    play_end = max((play.end_seconds for play in project.plays), default=0.0)
    play_start = min((play.start_seconds for play in project.plays), default=0.0)
    # Highlight reels leave most of the cut outside play spans — sample it too.
    if duration > play_end - play_start + 8:
        for index in range(1, 21):
            at = duration * index / 21
            if play_start <= at <= play_end:
                continue
            try:
                frames.append(still_frame(video, at))
            except (VideoAnalysisError, OSError, HTTPException):
                continue
    return frames[:96]


def _crops_checked(catalog: Catalog, project_id: int) -> bool:
    try:
        path = catalog.team_image(project_id, "left-player")
    except CatalogError:
        return False
    return path is not None and (path.parent / ".player-crops").is_file()


def _mark_crops_checked(catalog: Catalog, project_id: int) -> None:
    try:
        path = catalog.team_image(project_id, "left-player")
    except CatalogError:
        return
    if path is None:
        return
    with contextlib.suppress(OSError):
        (path.parent / ".player-crops").write_text("1", encoding="utf-8")


def _clear_crops_checked(catalog: Catalog, project_id: int) -> None:
    try:
        path = catalog.team_image(project_id, "left-player")
    except CatalogError:
        return
    if path is None:
        return
    with contextlib.suppress(OSError):
        (path.parent / ".player-crops").unlink(missing_ok=True)


def _team_frames(catalog: Catalog, project_id: int) -> list[bytes]:
    """Play photos used to find each team's players after the uniforms are known."""

    try:
        project = catalog.get_project(project_id)
    except CatalogError:
        return []
    if project is None or not project.plays:
        return []
    try:
        video, _at = _lineup_video(catalog, project)
    except HTTPException:
        return []
    return play_frames(video, project.plays, limit=6)


def _store_portraits(
    catalog: Catalog,
    project_id: int,
    teams: FilmTeams,
    keep_current: bool = False,
    sides: Sequence[str] = ("left", "right"),
    attempt: int = 1,
) -> None:
    """Crop one real player per team from the film. Nothing is generated."""

    pair = decode_uniforms(teams.uniforms)
    if pair is None:
        project = catalog.get_project(project_id)
        pair = (
            _detected_uniforms(catalog, project)
            if project is not None
            else (
                Uniform(teams.left_name, "white", "white", "white"),
                Uniform(teams.right_name, "dark blue", "white", "dark blue"),
            )
        )
    wearing = (
        portrait_uniform(pair[0], teams.left_name),
        portrait_uniform(pair[1], teams.right_name),
    )
    chosen = {side for side in sides if side in {"left", "right"}} or {"left", "right"}
    if not keep_current:
        _clear_crops_checked(catalog, project_id)
    waiting = {
        side: {
            "status": "scanning" if side in chosen else "idle",
            "label": "Scanning plays" if side in chosen else "Holding",
        }
        for side in ("left", "right")
    }
    _set_team_progress(
        project_id,
        step="find",
        percent=28,
        detail="Pulling live play frames to crop from.",
        attempt=attempt,
        event="Collecting play stills for player crops.",
        sides=waiting,
    )
    search_frames = _player_search_frames(catalog, project_id)
    _set_team_progress(
        project_id,
        step="find",
        percent=48,
        detail=f"Scanning {len(search_frames)} play frames for clear players.",
        attempt=attempt,
        event=f"Loaded {len(search_frames)} frames from the cut plays.",
        frames=len(search_frames),
        sides={
            side: {
                "status": "finding" if side in chosen else "idle",
                "label": "Finding player" if side in chosen else "Holding",
            }
            for side in ("left", "right")
        },
    )
    avoid: list[bytes] = []
    if not keep_current:
        for side in ("left", "right"):
            if side not in chosen:
                continue
            current_shot = _still_bytes(catalog, project_id, f"{side}-player")
            if current_shot:
                avoid.append(current_shot)
    cards = team_cards(search_frames, wearing[0], wearing[1], avoid)
    found = sum(
        1 for index, side in enumerate(("left", "right")) if side in chosen and cards[index]
    )
    _set_team_progress(
        project_id,
        step="find",
        percent=62,
        detail=(
            f"Matched a crop on {found} side{'s' if found != 1 else ''}."
            if found
            else "No clear crop yet. Trying a second pass."
        ),
        attempt=attempt,
        event=(
            f"First pass found {found} player crop{'s' if found != 1 else ''}."
            if found
            else "First pass missed. Trying lone-player search."
        ),
        frames=len(search_frames),
        sides={
            side: {
                "status": (
                    "found"
                    if side in chosen and cards[index]
                    else "searching" if side in chosen else "idle"
                ),
                "label": (
                    "Crop ready"
                    if side in chosen and cards[index]
                    else "Still looking" if side in chosen else "Holding"
                ),
            }
            for index, side in enumerate(("left", "right"))
        },
    )
    try:
        for index, side in enumerate(("left", "right")):
            if side not in chosen:
                continue
            label = teams.left_name if side == "left" else teams.right_name
            who = label or ("Home" if side == "left" else "Away")
            _set_team_progress(
                project_id,
                step="find",
                percent=66 + index * 8,
                detail=f"Working the {who} crop.",
                attempt=attempt,
                event=f"Checking {who} for a usable player still.",
                frames=len(search_frames),
                sides={
                    side: {"status": "finding", "label": f"Cropping {who}"},
                },
            )
            shot = cards[index] or b""
            if shot and not usable_player_jpeg(shot):
                shot = b""
            if not shot:
                other = wearing[1 - index]
                candidate = player_photo(search_frames, wearing[index], other) or b""
                if candidate and usable_player_jpeg(candidate):
                    shot = candidate
            # Wide lineup halves and referees are not team cards — leave blank.
            current = _still_bytes(catalog, project_id, f"{side}-player")
            if not shot:
                if keep_current and current is not None and not _portrait_should_replace(current):
                    _set_team_progress(
                        project_id,
                        step="find",
                        percent=70 + index * 10,
                        detail=f"Keeping the current {who} photo.",
                        attempt=attempt,
                        event=f"Kept the existing {who} player photo.",
                        sides={side: {"status": "kept", "label": "Keeping photo"}},
                    )
                    continue
                _set_team_progress(
                    project_id,
                    step="find",
                    percent=70 + index * 10,
                    detail=f"No clear {who} player in these frames.",
                    attempt=attempt,
                    event=f"Could not crop a clear {who} player yet.",
                    sides={side: {"status": "missed", "label": "No clear crop"}},
                )
                continue
            if (
                keep_current
                and current is not None
                and not _portrait_should_replace(current)
                and cards[index] is None
            ):
                _set_team_progress(
                    project_id,
                    step="find",
                    percent=70 + index * 10,
                    detail=f"Keeping the current {who} photo.",
                    attempt=attempt,
                    event=f"Kept the existing {who} player photo.",
                    sides={side: {"status": "kept", "label": "Keeping photo"}},
                )
                continue
            _set_team_progress(
                project_id,
                step="save",
                percent=78 + index * 10,
                detail=f"Saving the {who} player.",
                attempt=attempt,
                event=f"Saving the {who} player card.",
                sides={side: {"status": "saving", "label": "Saving card"}},
            )
            catalog.save_team_image(project_id, f"{side}-player", shot, ".jpg")
            catalog.save_team_image(project_id, f"{side}-reference", shot, ".jpg")
            _set_team_progress(
                project_id,
                step="save",
                percent=82 + index * 10,
                detail=f"{who} player is on the card.",
                attempt=attempt,
                event=f"Saved the {who} player card.",
                sides={side: {"status": "saved", "label": "Card ready"}},
            )
        _mark_crops_checked(catalog, project_id)
        _set_team_progress(
            project_id,
            step="save",
            percent=100,
            detail="Both teams are ready to scout.",
            attempt=attempt,
            running=False,
            event="Team cards are ready.",
            frames=len(search_frames),
        )
    except CatalogError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


def _store_references(catalog: Catalog, project_id: int, teams: FilmTeams) -> None:
    """Keep the photo the model is given, without drawing the players again."""

    pair = decode_uniforms(teams.uniforms)
    if pair is None:
        return
    wearing = (
        portrait_uniform(pair[0], teams.left_name),
        portrait_uniform(pair[1], teams.right_name),
    )
    frames = _team_frames(catalog, project_id)
    try:
        for index, (side, uniform) in enumerate(zip(("left", "right"), wearing, strict=True)):
            if _still_bytes(catalog, project_id, f"{side}-reference") is not None:
                continue
            other = wearing[1 - index]
            still = _still_bytes(catalog, project_id, side) or b""
            focused = focus_player(still, side)
            reference = reference_for_model([focused, still, *frames], uniform, other)
            if not _is_jpeg(reference):
                continue
            catalog.save_team_image(project_id, f"{side}-reference", reference, ".jpg")
    except CatalogError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


def _save_reference(
    catalog: Catalog,
    project_id: int,
    side: str,
    draw: Callable[..., tuple[bytes, str]],
    photos: list[bytes],
    uniform: Uniform,
    other: Uniform,
) -> None:
    sent = getattr(draw, "last_reference", None)
    reference = sent if _is_jpeg(sent) else reference_for_model(photos, uniform, other)
    if not _is_jpeg(reference):
        return
    catalog.save_team_image(project_id, f"{side}-reference", reference, ".jpg")


def _clear_reference(draw: Callable[..., tuple[bytes, str]]) -> None:
    try:
        draw.last_reference = None  # type: ignore[attr-defined]
    except AttributeError:
        return


def _is_jpeg(data: object) -> bool:
    return isinstance(data, bytes) and data.startswith(b"\xff\xd8\xff")


def _references_missing(catalog: Catalog, project_id: int) -> bool:
    return _still_bytes(catalog, project_id, "left-reference") is None or _still_bytes(
        catalog, project_id, "right-reference"
    ) is None


def _teams_view(
    catalog: Catalog,
    project_id: int,
    project: ProjectDetail | None = None,
    saved: FilmTeams | None = None,
    *,
    prepared: bool = False,
) -> dict[str, object] | None:
    try:
        if not prepared:
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
    if project is None and not prepared:
        try:
            project = catalog.get_project(project_id)
        except CatalogError:
            project = None
    found = decode_uniforms("" if saved is None else saved.uniforms)
    labeled = None if found is None else label_home_away(*found)
    guessed = (
        saved is not None and found is not None and _names_from_detection(saved, found)
    )
    if labeled is not None and (saved is None or _placeholder_names(saved) or guessed):
        left_name, right_name = labeled[0].name, labeled[1].name
    else:
        suggested = suggest_team_names("" if project is None else project.name)
        left_name = suggested[0] if saved is None or not saved.left_name else saved.left_name
        right_name = suggested[1] if saved is None or not saved.right_name else saved.right_name
    progress = _team_progress_view(project_id)
    payload: dict[str, object] = {
        "left_id": "left",
        "right_id": "right",
        "left_name": left_name,
        "right_name": right_name,
        "mine": None if saved is None else saved.mine,
        "left_uniform": "" if found is None else uniform_line(found[0]),
        "right_uniform": "" if found is None else uniform_line(found[1]),
        "left_colors": [] if found is None else uniform_colors(found[0]),
        "right_colors": [] if found is None else uniform_colors(found[1]),
        "left_still": url("left"),
        "right_still": url("right"),
        "left_player": url("left-player"),
        "right_player": url("right-player"),
        "left_reference": url("left-reference"),
        "right_reference": url("right-reference"),
    }
    if progress is not None:
        payload["building"] = bool(progress.get("running"))
        payload["progress"] = progress
    return payload


def _report_counts(catalog: Catalog | None, project_id: int) -> tuple[int, int]:
    """How many scouting-report plays are finished, and how many are in the table."""

    if catalog is None:
        return 0, 0
    try:
        rows = catalog.list_report_analysis(project_id)
    except CatalogError:
        return 0, 0
    done = sum(1 for row in rows if row.status in {"ready", "missed"})
    return done, len(rows)


def _list_report_progress(
    catalog: Catalog,
    running: set[int],
    lock: threading.Lock,
    item: ProjectSummary,
) -> tuple[str, int, int]:
    """Phase shown on the film list. A ready film that is running is charting."""

    if not _running(running, lock, item.project):
        return "", 0, 0
    if item.status != "ready":
        return item.status, 0, 0
    done, total = _report_counts(catalog, item.project)
    return "charting", done, total


def _project_payload(
    project: ProjectDetail,
    *,
    running: bool = False,
    steps: tuple[ActivityStep, ...] = (),
    catalog: Catalog | None = None,
    film_teams: FilmTeams | None = None,
    read_notes: list[ReadNote] | None = None,
    prepared: bool = False,
    still: _Still = still_frame,
    make_poster: bool = False,
) -> dict[str, object]:
    charting = running and project.status == "ready"
    report_done, report_total = (
        _report_counts(catalog, project.project) if charting else (0, 0)
    )
    return {
        "running": running,
        "phase": "charting" if charting else "",
        "report_done": report_done,
        "report_total": report_total,
        "project": project.project,
        "name": project.name,
        "status": project.status,
        "duration": project.duration_seconds,
        "scanned": project.scanned_seconds,
        "retries": project.retries,
        "retries_done": project.retries_done,
        "processing": processing_percent(
            project.duration_seconds,
            project.scanned_seconds,
            project.retries,
            project.retries_done,
        ),
        "error": project.error,
        "channel": project.channel,
        "youtube_url": project.youtube_url,
        "thumbnail": _thumbnail_for(project, catalog, still, make=make_poster),
        "downloaded": project.downloaded_bytes,
        "total_bytes": project.total_bytes,
        "steps": _step_times(steps),
        "teams": None
        if catalog is None
        else _teams_view(
            catalog,
            project.project,
            project,
            film_teams,
            prepared=prepared,
        ),
        "plays": [
            _play_summary(project, play, catalog) for play in _ordered_plays(project)
        ],
        "reads": []
        if catalog is None
        else _read_views(catalog, project.project, read_notes if prepared else None),
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


def _read_views(
    catalog: Catalog, project_id: int, notes: list[ReadNote] | None = None
) -> list[dict[str, object]]:
    if notes is None:
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
        if note.detail
    ]


def _priced_read(usage: object) -> tuple[int, int, float, str] | None:
    """Tokens, dollars, and model from a successful read. Empty usage is ignored."""

    if not isinstance(usage, tuple) or len(usage) != 3:
        return None
    incoming, outgoing, model = usage
    if not isinstance(incoming, int) or not isinstance(outgoing, int) or not isinstance(model, str):
        return None
    if incoming <= 0 and outgoing <= 0:
        return None
    return incoming, outgoing, read_cost(model, incoming, outgoing), model


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
    lineup: Callable[[Path, float], tuple[bytes, bytes]] = split_lineup,
    portrait: Callable[..., tuple[bytes, str]] = _film_player,
    still: _Still = still_frame,
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
        still=still,
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
