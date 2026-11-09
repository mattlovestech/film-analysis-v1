"""Run one ingest, label, or cut, then exit the request."""

import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from film_analysis.agents.cost import read_cost
from film_analysis.agents.errors import VideoAnalysisError
from film_analysis.agents.frames import video_duration
from film_analysis.agents.play_boundaries import agent_from_env
from film_analysis.agents.segments import FilmSpan
from film_analysis.agents.youtube import detach_progress, download_youtube, youtube_details
from film_analysis.storage.catalog import Catalog, CatalogError, catalog_from_env
from film_analysis.storage.r2 import source_url
from film_analysis.tasks.media import cut_url, extract_window_url
from film_analysis.web.app import _load_dotenv
from film_analysis.web.playback import prepare_playback

app = FastAPI(title="film-task", docs_url=None, redoc_url=None, openapi_url=None)


class Job(BaseModel):
    kind: str
    project_id: int = Field(ge=1)
    start: float = 0
    end: float = 0
    play_number: int = 0
    family: str = "scrimmage"
    youtube: str | None = None
    review: bool = False


@app.get("/ping")
def ping() -> dict[str, bool]:
    return {"ok": True}


@app.post("/run")
def run(job: Job, authorization: str | None = Header(default=None)) -> dict[str, object]:
    _check_token(authorization)
    try:
        if job.kind == "ingest":
            return {"duration": _ingest(job)}
        if job.kind == "label":
            spans, usage, model = _label(job)
            incoming, outgoing = usage
            return {
                "spans": spans,
                "input_tokens": incoming,
                "output_tokens": outgoing,
                "cost_usd": read_cost(model, incoming, outgoing),
                "model": model,
            }
        if job.kind == "cut":
            _cut(job)
            return {"ok": True}
    except (VideoAnalysisError, CatalogError, OSError, SystemExit) as error:
        raise HTTPException(status_code=400, detail=_short(error)) from error
    raise HTTPException(status_code=400, detail="Unknown job.")


def main() -> None:
    import uvicorn

    _load_dotenv()
    port = int(os.environ.get("FILM_ANALYSIS_PORT") or os.environ.get("PORT") or "8000")
    uvicorn.run(app, host="0.0.0.0", port=port)


def _ingest(job: Job) -> float:
    catalog = catalog_from_env()
    owned = Path(tempfile.mkdtemp(prefix="film-ingest-"))
    try:
        if job.youtube:
            try:
                info = youtube_details(job.youtube)
            except Exception:
                info = None
            if info is not None:
                catalog.describe_film(
                    job.project_id,
                    name=info.title or None,
                    channel=info.channel or None,
                    duration_seconds=info.duration_seconds,
                )
            path = download_youtube(job.youtube, owned, _download_progress(catalog, job.project_id))
        else:
            path = _download_source(job.project_id, owned)
        prepared = prepare_playback(path)
        catalog.save_source(job.project_id, prepared)
        return video_duration(prepared)
    finally:
        shutil.rmtree(owned, ignore_errors=True)


def _label(job: Job) -> tuple[list[dict[str, object]], tuple[int, int], str]:
    if job.end <= job.start:
        raise VideoAnalysisError("A stretch must end after it starts.")
    samples = extract_window_url(source_url(job.project_id), job.start, job.end - job.start)
    agent = agent_from_env()
    spans = agent.review_window(samples) if job.review else agent.label_window(samples)
    return [_span_payload(span) for span in spans], agent.last_usage, agent.model


def _cut(job: Job) -> None:
    if job.play_number < 1 or job.end <= job.start:
        raise VideoAnalysisError("A play must end after it starts.")
    folder = Path(tempfile.mkdtemp(prefix="film-cut-"))
    dest = folder / "clip.mp4"
    try:
        cut_url(source_url(job.project_id), job.start, job.end, dest)
        catalog_from_env().add_play(
            job.project_id,
            job.play_number,
            job.start,
            job.end,
            dest,
            job.family,
        )
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def _download_source(project_id: int, directory: Path) -> Path:
    dest = directory / "source.mp4"
    try:
        with urllib.request.urlopen(source_url(project_id), timeout=600) as response:
            dest.write_bytes(response.read())
    except OSError as error:
        raise VideoAnalysisError("Could not read that film.") from error
    if not dest.is_file() or dest.stat().st_size == 0:
        raise VideoAnalysisError("Could not read that film.")
    return dest


def _download_progress(catalog: Catalog, project_id: int):
    def send(done: int, total: int | None) -> None:
        try:
            catalog.set_download(project_id, done, total)
        except CatalogError:
            return

    return detach_progress(send)


def _span_payload(span: FilmSpan) -> dict[str, object]:
    return {"kind": span.kind, "start": span.start_seconds, "end": span.end_seconds}


def _check_token(authorization: str | None) -> None:
    expected = os.environ.get("FILM_INTERNAL_TOKEN", "").strip()
    if expected and authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Unauthorized.")


def _short(error: BaseException) -> str:
    if isinstance(error, SystemExit):
        return "Could not fetch that YouTube video."
    text = " ".join(str(error).split())
    if len(text) > 180:
        return text[:177] + "..."
    return text or "Could not read that film."
