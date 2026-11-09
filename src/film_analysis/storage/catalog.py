"""Project and play records, with one video file per play."""

import json
import os
import shutil
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol


class CatalogError(Exception):
    """The analysis catalog could not be read or written."""


@dataclass(frozen=True)
class ProjectSummary:
    project: int
    name: str
    play_count: int
    youtube_url: str | None = None
    status: str = "ready"
    downloaded_bytes: int = 0
    total_bytes: int | None = None
    created_at: str | None = None


@dataclass(frozen=True)
class PlayRecord:
    play_number: int
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class ProjectDetail:
    project: int
    name: str
    youtube_url: str | None
    plays: tuple[PlayRecord, ...]
    status: str = "ready"
    duration_seconds: float | None = None
    scanned_seconds: float = 0.0
    error: str | None = None
    channel: str | None = None
    downloaded_bytes: int = 0
    total_bytes: int | None = None


class Catalog(Protocol):
    """Store projects, play boundaries, and the clip for each play."""

    def list_projects(self) -> list[ProjectSummary]:
        """Return saved analyses, newest first."""

    def get_project(self, project_id: int) -> ProjectDetail | None:
        """Return one analysis, or nothing when the id is unknown."""

    def create_project(self, name: str, youtube_url: str | None) -> int:
        """Start an analysis and return its project number."""

    def set_progress(
        self,
        project_id: int,
        *,
        status: str,
        duration_seconds: float | None,
        scanned_seconds: float,
        error: str | None,
    ) -> None:
        """Record how far a long film has been downloaded and scanned."""

    def describe_film(
        self,
        project_id: int,
        *,
        name: str | None,
        channel: str | None,
        duration_seconds: float | None,
    ) -> None:
        """Store the title, channel, and length once they are known."""

    def set_download(
        self,
        project_id: int,
        downloaded_bytes: int,
        total_bytes: int | None,
    ) -> None:
        """Record how many bytes of the film file have arrived."""

    def add_span(self, project_id: int, kind: str, start: float, end: float) -> None:
        """Remember a stretch that is not a play."""

    def save_source(self, project_id: int, source: Path) -> None:
        """Keep the film the plays were cut from."""

    def add_play(
        self,
        project_id: int,
        play_number: int,
        start: float,
        end: float,
        clip: Path,
    ) -> None:
        """Attach one play and its clip to a project."""

    def delete_project(self, project_id: int) -> None:
        """Remove an analysis and its clips."""

    def clip_file(self, project_id: int, play_number: int) -> Path | None:
        """Return the play clip, when it is still on disk."""


class ClipStore(Protocol):
    """Where the source film and each play clip are kept."""

    def save_source(self, project_id: int, source: Path) -> str:
        """Store the full film and return its key."""

    def save_play(self, project_id: int, play_number: int, clip: Path) -> str:
        """Store one play clip and return its key."""

    def play_path(self, project_id: int, play_number: int) -> Path:
        """Return a local file for that play, downloading it when needed."""

    def remove(self, project_id: int) -> None:
        """Delete every object for a project."""


class ClipDirectory:
    """Play clips and source films under one directory."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def save_source(self, project_id: int, source: Path) -> str:
        dest = self._project(project_id) / "source.mp4"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        return dest.relative_to(self.root).as_posix()

    def save_play(self, project_id: int, play_number: int, clip: Path) -> str:
        dest = self.play_path(project_id, play_number)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(clip, dest)
        return dest.relative_to(self.root).as_posix()

    def play_path(self, project_id: int, play_number: int) -> Path:
        return self._project(project_id) / "plays" / f"{play_number}.mp4"

    def remove(self, project_id: int) -> None:
        shutil.rmtree(self._project(project_id), ignore_errors=True)

    def _project(self, project_id: int) -> Path:
        return self.root / "projects" / str(project_id)


class MemoryCatalog:
    """In-memory analyses for tests and for a server with no database configured."""

    def __init__(self, root: Path) -> None:
        self._clips = ClipDirectory(root)
        self._projects: list[ProjectDetail] = []
        self._spans: dict[int, list[tuple[str, float, float]]] = {}
        self._created: dict[int, str] = {}
        self._next = 1
        self._lock = threading.Lock()

    def list_projects(self) -> list[ProjectSummary]:
        with self._lock:
            projects = list(self._projects)
            created = dict(self._created)
        return [
            ProjectSummary(
                project.project,
                project.name,
                len(project.plays),
                project.youtube_url,
                project.status,
                project.downloaded_bytes,
                project.total_bytes,
                created.get(project.project),
            )
            for project in reversed(projects)
        ]

    def get_project(self, project_id: int) -> ProjectDetail | None:
        with self._lock:
            return self._find(project_id)

    def create_project(self, name: str, youtube_url: str | None) -> int:
        with self._lock:
            project_id = self._next
            self._next += 1
            self._created[project_id] = datetime.now(UTC).isoformat()
            self._projects.append(
                ProjectDetail(
                    project=project_id,
                    name=name,
                    youtube_url=youtube_url,
                    plays=(),
                    status="downloading",
                )
            )
            self._spans[project_id] = []
        return project_id

    def set_progress(
        self,
        project_id: int,
        *,
        status: str,
        duration_seconds: float | None,
        scanned_seconds: float,
        error: str | None,
    ) -> None:
        with self._lock:
            current = self._require(project_id)
            self._replace(
                project_id,
                ProjectDetail(
                    project=current.project,
                    name=current.name,
                    youtube_url=current.youtube_url,
                    plays=current.plays,
                    status=status,
                    duration_seconds=(
                        current.duration_seconds
                        if duration_seconds is None
                        else duration_seconds
                    ),
                    scanned_seconds=scanned_seconds,
                    error=error,
                    channel=current.channel,
                    downloaded_bytes=current.downloaded_bytes,
                    total_bytes=current.total_bytes,
                ),
            )

    def describe_film(
        self,
        project_id: int,
        *,
        name: str | None,
        channel: str | None,
        duration_seconds: float | None,
    ) -> None:
        with self._lock:
            current = self._require(project_id)
            self._replace(
                project_id,
                ProjectDetail(
                    project=current.project,
                    name=name or current.name,
                    youtube_url=current.youtube_url,
                    plays=current.plays,
                    status=current.status,
                    duration_seconds=(
                        duration_seconds
                        if duration_seconds is not None
                        else current.duration_seconds
                    ),
                    scanned_seconds=current.scanned_seconds,
                    error=current.error,
                    channel=channel or current.channel,
                    downloaded_bytes=current.downloaded_bytes,
                    total_bytes=current.total_bytes,
                ),
            )

    def set_download(
        self,
        project_id: int,
        downloaded_bytes: int,
        total_bytes: int | None,
    ) -> None:
        with self._lock:
            current = self._require(project_id)
            self._replace(
                project_id,
                ProjectDetail(
                    project=current.project,
                    name=current.name,
                    youtube_url=current.youtube_url,
                    plays=current.plays,
                    status=current.status,
                    duration_seconds=current.duration_seconds,
                    scanned_seconds=current.scanned_seconds,
                    error=current.error,
                    channel=current.channel,
                    downloaded_bytes=downloaded_bytes,
                    total_bytes=(
                        current.total_bytes if total_bytes is None else total_bytes
                    ),
                ),
            )

    def add_span(self, project_id: int, kind: str, start: float, end: float) -> None:
        with self._lock:
            self._require(project_id)
            self._spans.setdefault(project_id, []).append((kind, start, end))

    def save_source(self, project_id: int, source: Path) -> None:
        with self._lock:
            self._require(project_id)
        self._clips.save_source(project_id, source)

    def add_play(
        self,
        project_id: int,
        play_number: int,
        start: float,
        end: float,
        clip: Path,
    ) -> None:
        self._clips.save_play(project_id, play_number, clip)
        with self._lock:
            current = self._require(project_id)
            self._replace(
                project_id,
                ProjectDetail(
                    project=current.project,
                    name=current.name,
                    youtube_url=current.youtube_url,
                    plays=(*current.plays, PlayRecord(play_number, start, end)),
                    status=current.status,
                    duration_seconds=current.duration_seconds,
                    scanned_seconds=current.scanned_seconds,
                    error=current.error,
                    channel=current.channel,
                    downloaded_bytes=current.downloaded_bytes,
                    total_bytes=current.total_bytes,
                ),
            )

    def delete_project(self, project_id: int) -> None:
        with self._lock:
            self._projects = [
                project for project in self._projects if project.project != project_id
            ]
            self._spans.pop(project_id, None)
            self._created.pop(project_id, None)
        self._clips.remove(project_id)

    def clip_file(self, project_id: int, play_number: int) -> Path | None:
        path = self._clips.play_path(project_id, play_number)
        if path.is_file():
            return path
        return None

    def _replace(self, project_id: int, updated: ProjectDetail) -> None:
        self._projects = [
            updated if project.project == project_id else project for project in self._projects
        ]

    def _find(self, project_id: int) -> ProjectDetail | None:
        return next((project for project in self._projects if project.project == project_id), None)

    def _require(self, project_id: int) -> ProjectDetail:
        project = self._find(project_id)
        if project is None:
            raise CatalogError(f"Project {project_id} was not found.")
        return project


class D1Catalog:
    """Cloudflare D1 for the rows, and R2 or a local directory for the clips."""

    def __init__(
        self,
        account_id: str,
        api_token: str,
        database_id: str,
        clips: ClipStore,
    ) -> None:
        self._account_id = account_id
        self._api_token = api_token
        self._database_id = database_id
        self._clips = clips
        self._ensure_schema()

    def list_projects(self) -> list[ProjectSummary]:
        rows = self._query(
            """
            select projects.id as project, projects.name as name,
                   projects.youtube_url as youtube_url, projects.status as status,
                   projects.downloaded_bytes as downloaded_bytes,
                   projects.total_bytes as total_bytes,
                   projects.created_at as created_at,
                   count(plays.play_number) as play_count
            from projects
            left join plays on plays.project_id = projects.id
            group by projects.id
            order by projects.id desc
            """
        )
        return [
            ProjectSummary(
                int(row["project"]),
                str(row["name"]),
                int(row["play_count"]),
                _optional_text(row.get("youtube_url")),
                str(row.get("status") or "ready"),
                int(row.get("downloaded_bytes") or 0),
                _optional_int(row.get("total_bytes")),
                _optional_text(row.get("created_at")),
            )
            for row in rows
        ]

    def get_project(self, project_id: int) -> ProjectDetail | None:
        rows = self._query(
            """
            select id, name, youtube_url, status, duration_seconds, scanned_seconds, error,
                   channel, downloaded_bytes, total_bytes
            from projects where id = ?
            """,
            [project_id],
        )
        if not rows:
            return None
        project = rows[0]
        plays = self._query(
            """
            select play_number, start_seconds, end_seconds
            from plays
            where project_id = ?
            order by play_number
            """,
            [project_id],
        )
        return ProjectDetail(
            project=int(project["id"]),
            name=str(project["name"]),
            youtube_url=_optional_text(project.get("youtube_url")),
            plays=tuple(_play_record(play) for play in plays),
            status=str(project.get("status") or "ready"),
            duration_seconds=_optional_float(project.get("duration_seconds")),
            scanned_seconds=float(project.get("scanned_seconds") or 0),
            error=_optional_text(project.get("error")),
            channel=_optional_text(project.get("channel")),
            downloaded_bytes=int(project.get("downloaded_bytes") or 0),
            total_bytes=_optional_int(project.get("total_bytes")),
        )

    def create_project(self, name: str, youtube_url: str | None) -> int:
        created_at = datetime.now(UTC).isoformat()
        result = self._execute(
            """
            insert into projects (name, source_key, youtube_url, created_at, status)
            values (?, ?, ?, ?, ?)
            returning id
            """,
            [name, "", youtube_url, created_at, "downloading"],
        )
        rows = result.get("results") or []
        if rows:
            return int(rows[0]["id"])
        last_id = (result.get("meta") or {}).get("last_row_id")
        if not last_id:
            raise CatalogError("Could not save that analysis.")
        return int(last_id)

    def set_progress(
        self,
        project_id: int,
        *,
        status: str,
        duration_seconds: float | None,
        scanned_seconds: float,
        error: str | None,
    ) -> None:
        self._execute(
            """
            update projects
            set status = ?,
                duration_seconds = coalesce(?, duration_seconds),
                scanned_seconds = ?,
                error = ?
            where id = ?
            """,
            [status, duration_seconds, scanned_seconds, error, project_id],
        )

    def describe_film(
        self,
        project_id: int,
        *,
        name: str | None,
        channel: str | None,
        duration_seconds: float | None,
    ) -> None:
        self._execute(
            """
            update projects
            set name = coalesce(?, name),
                channel = coalesce(?, channel),
                duration_seconds = coalesce(?, duration_seconds)
            where id = ?
            """,
            [name, channel, duration_seconds, project_id],
        )

    def set_download(
        self,
        project_id: int,
        downloaded_bytes: int,
        total_bytes: int | None,
    ) -> None:
        self._execute(
            """
            update projects
            set downloaded_bytes = ?, total_bytes = coalesce(?, total_bytes)
            where id = ?
            """,
            [downloaded_bytes, total_bytes, project_id],
        )

    def add_span(self, project_id: int, kind: str, start: float, end: float) -> None:
        self._execute(
            """
            insert into spans (project_id, kind, start_seconds, end_seconds)
            values (?, ?, ?, ?)
            """,
            [project_id, kind, start, end],
        )

    def save_source(self, project_id: int, source: Path) -> None:
        key = self._clips.save_source(project_id, source)
        self._execute("update projects set source_key = ? where id = ?", [key, project_id])

    def add_play(
        self,
        project_id: int,
        play_number: int,
        start: float,
        end: float,
        clip: Path,
    ) -> None:
        key = self._clips.save_play(project_id, play_number, clip)
        self._execute(
            """
            insert into plays (project_id, play_number, start_seconds, end_seconds, clip_key)
            values (?, ?, ?, ?, ?)
            """,
            [project_id, play_number, start, end, key],
        )

    def delete_project(self, project_id: int) -> None:
        self._execute("delete from spans where project_id = ?", [project_id])
        self._execute("delete from plays where project_id = ?", [project_id])
        self._execute("delete from projects where id = ?", [project_id])
        self._clips.remove(project_id)

    def clip_file(self, project_id: int, play_number: int) -> Path | None:
        path = self._clips.play_path(project_id, play_number)
        if path.is_file():
            return path
        return None

    def _query(self, sql: str, params: list[object] | None = None) -> list[dict[str, object]]:
        result = self._execute(sql, params)
        rows = result.get("results") or []
        return [row for row in rows if isinstance(row, dict)]

    def _execute(self, sql: str, params: list[object] | None = None) -> dict[str, object]:
        body: dict[str, object] = {"sql": " ".join(sql.split())}
        if params:
            body["params"] = params
        request = urllib.request.Request(
            "https://api.cloudflare.com/client/v4/accounts/"
            f"{self._account_id}/d1/database/{self._database_id}/query",
            data=json.dumps(body).encode(),
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            raise CatalogError(_d1_failure(detail)) from error
        except urllib.error.URLError as error:
            raise CatalogError("Could not reach the analysis database.") from error
        if not payload.get("success"):
            raise CatalogError(_d1_failure(json.dumps(payload.get("errors"))))
        results = payload.get("result") or []
        if not results:
            return {}
        first = results[0]
        if not first.get("success", True):
            raise CatalogError("Could not save that analysis.")
        return first

    def _ensure_schema(self) -> None:
        statements = (
            "alter table projects add column status text not null default 'ready'",
            "alter table projects add column duration_seconds real",
            "alter table projects add column scanned_seconds real not null default 0",
            "alter table projects add column error text",
            "alter table projects add column channel text",
            "alter table projects add column downloaded_bytes integer not null default 0",
            "alter table projects add column total_bytes integer",
            """
            create table if not exists spans (
                project_id integer not null,
                start_seconds real not null,
                end_seconds real not null,
                kind text not null
            )
            """,
        )
        for sql in statements:
            try:
                self._execute(sql)
            except CatalogError as error:
                text = str(error).lower()
                if "duplicate column" in text or "already exists" in text:
                    continue
                raise


def catalog_from_env(root: Path | None = None) -> Catalog:
    """Use D1 and R2 when they are configured. Clip playback is cached under `root`."""

    directory = root or Path(os.environ.get("FILM_ANALYSIS_DATA", "data"))
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
    api_token = os.environ.get("CLOUDFLARE_API_TOKEN", "")
    database_id = os.environ.get("D1_DATABASE_ID", "")
    if account_id and api_token and database_id:
        return D1Catalog(account_id, api_token, database_id, _clips_from_env(directory))
    return MemoryCatalog(directory)


def _clips_from_env(directory: Path) -> ClipStore:
    access_key = os.environ.get("CLOUDFLARE_ACCESS_KEY_ID", "")
    secret_key = os.environ.get("CLOUDFLARE_SECRET_ACCESS_KEY", "")
    endpoint = os.environ.get("CLOUDFLARE_S3_ENDPOINT", "")
    bucket = os.environ.get("R2_BUCKET", "")
    if access_key and secret_key and endpoint and bucket:
        from film_analysis.storage.r2 import R2Clips

        return R2Clips(endpoint, access_key, secret_key, bucket, directory)
    return ClipDirectory(directory)


def _optional_text(value: object) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _play_record(play: dict[str, object]) -> PlayRecord:
    return PlayRecord(
        int(play["play_number"]),
        float(play["start_seconds"]),
        float(play["end_seconds"]),
    )


def _d1_failure(detail: str) -> str:
    text = " ".join(detail.split())
    if len(text) > 180:
        text = text[:177] + "..."
    if not text:
        return "Could not save that analysis."
    return f"Could not save that analysis. {text}"
