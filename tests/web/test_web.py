import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from film_analysis.chart.formation import FORMATIONS
from film_analysis.ingest.errors import MissingApiKeyError, VideoAnalysisError, tool_failure
from film_analysis.ingest.youtube import FilmInfo
from film_analysis.plays.boundaries import AnthropicPlayAgent
from film_analysis.plays.segments import FilmSpan, PlaySegment
from film_analysis.storage.catalog import FilmTeams, MemoryCatalog, ReadNote, decode_activity_steps
from film_analysis.web.app import create_app

_FORMATION_CHOICES = [{"key": key, "label": label} for key, label in FORMATIONS.items()]


class _Agent:
    name = "test"

    def __init__(self, segments: tuple[PlaySegment, ...] = ()) -> None:
        self.segments = segments
        self.paths: list[Path] = []
        self.urls: list[str] = []

    def segment(self, video_path: Path) -> tuple[PlaySegment, ...]:
        assert video_path.is_file()
        self.paths.append(video_path)
        return self.segments

    def label_window(self, samples: list[tuple[float, bytes]]) -> tuple[FilmSpan, ...]:
        if not samples:
            return ()
        start = samples[0][0]
        end = samples[-1][0] + 0.2
        return tuple(
            FilmSpan(kind="play", start_seconds=play.start_seconds, end_seconds=play.end_seconds)
            for play in self.segments
            if play.end_seconds > start and play.start_seconds < end
        )

    def segment_youtube(self, url: str) -> tuple[PlaySegment, ...]:
        self.urls.append(url)
        return self.segments


class _Raising:
    name = "raising"

    def __init__(self, error: Exception) -> None:
        self.error = error

    def segment(self, video_path: Path) -> tuple[PlaySegment, ...]:
        raise self.error

    def label_window(self, samples: list[tuple[float, bytes]]) -> tuple[FilmSpan, ...]:
        raise self.error

    def segment_youtube(self, url: str) -> tuple[PlaySegment, ...]:
        raise self.error


def _keep_source(path: Path) -> Path:
    return path


def _duration(path: Path) -> float:
    del path
    return 4.0


def _frames(path: Path, start: float, seconds: float) -> list[tuple[float, bytes]]:
    del path, seconds
    return [(start, b"jpeg"), (start + 3.0, b"jpeg")]


def _settle(client: TestClient, project_id: int) -> dict[str, object]:
    last: dict[str, object] = {}
    for _ in range(100):
        last = client.get(f"/api/projects/{project_id}").json()
        if last.get("status") in {"ready", "failed"}:
            return last
        time.sleep(0.02)
    raise AssertionError(last)


def _copy_clip(source: Path, start: float, end: float, dest: Path) -> None:
    del start, end
    dest.write_bytes(source.read_bytes())


def test_a_film_has_a_page_for_each_part() -> None:
    client = TestClient(create_app(agent=_Agent()))

    for path in ("/projects/4/review", "/projects/4/select-teams", "/projects/4/generate-report"):
        page = client.get(path)
        assert page.status_code == 200
        assert 'data-screen="review"' in page.text
        assert 'data-screen="generate-report"' in page.text

    missing = client.get("/projects/4/poster")
    assert missing.status_code == 404
    assert "data-screen" not in missing.text


def test_page_offers_a_video_upload() -> None:
    client = TestClient(create_app(agent=_Agent()))
    response = client.get("/")

    assert response.status_code == 200
    assert 'type="file"' in response.text
    assert 'name="youtube"' in response.text
    assert "Drop a video here" in response.text
    assert 'id="saved"' in response.text
    assert "Downloaded film" in response.text
    assert 'id="new-analysis" href="/"' in response.text
    assert '<h1><a href="/projects">Opponent scouting</a></h1>' in response.text
    assert ">Scout an opponent<" in response.text
    assert 'href="/projects"' in response.text
    assert ">Projects<" in response.text
    assert "Opponent games you've scouted." in response.text
    assert 'id="project-filters"' in response.text
    assert "Put back" in response.text
    assert 'id="refresh-analyses"' in response.text
    assert 'href="/activity"' in response.text
    assert 'href="/settings"' in response.text
    assert ">Settings<" in response.text
    assert 'id="settings"' in response.text
    assert "How a film becomes plays" in response.text
    assert "Charting a play comes later, after its clip is final." in response.text
    assert "keeps that map on the play page" in response.text
    assert "Read offensive formations" in response.text
    assert 'id="formation-reads" type="checkbox" checked' in response.text
    assert "Experimental, and last." not in response.text
    assert '<span class="experimental">Experimental</span>' not in response.text
    assert 'id="activity-loading"' in response.text
    assert "Checking the server" in response.text
    assert "Looking up films that are running or stopped." in response.text
    assert 'id="activity-legend"' in response.text
    assert 'className = "facts"' in response.text
    assert 'id="chart-time"' in response.text
    assert 'id="chart-plays"' in response.text
    assert 'id="chart-cost"' in response.text
    assert "Read cost" in response.text
    assert "Total time" in response.text
    assert "Reading plays per minute" in response.text
    assert 'id="activity-source"' in response.text
    assert 'data-source="local"' in response.text
    assert 'data-source="cloud"' in response.text
    assert "Show films by where they ran" in response.text
    assert "× realtime" in response.text
    assert "plays/min" in response.text
    assert 'id="film-steps"' in response.text
    assert 'id="film-loading"' in response.text
    assert "Opening this film" in response.text
    assert 'id="workers"' in response.text
    assert 'id="place-local"' in response.text
    assert 'id="place-cloud"' in response.text
    assert "Where jobs run" in response.text
    assert "Jobs run on this computer." in response.text
    assert "Requests go to Cloudflare." in response.text
    assert "This page stays here. Starting a film sends that job to Cloudflare." in response.text
    assert "Starting on Cloudflare" not in response.text
    assert 'id="analyze"' in response.text
    assert 'aria-busy="false"' in response.text
    assert "location.assign" not in response.text
    assert "Cloud · 1 worker" in response.text
    assert 'id="teams"' in response.text
    assert "Team being scouted" in response.text
    assert "pick who you are scouting so offense and defense line up" in response.text
    assert "The right arrow opens the next play." in response.text
    assert 'id="team-left-reference"' in response.text
    assert 'id="team-right-reference"' in response.text
    assert ">Reference</figcaption>" in response.text
    assert ">Scout</button>" in response.text
    assert "Try different photos" in response.text
    assert 'id="teams-retry"' in response.text
    assert 'id="teams-meter"' in response.text
    assert "Read film" in response.text
    assert "Find players" in response.text
    assert "Save cards" in response.text
    assert "New photo" in response.text
    assert "Reset team photos for all films" in response.text
    assert 'id="team-reset"' in response.text
    assert 'id="team-reset-meter"' in response.text
    assert 'id="team-reset-films"' in response.text
    assert 'id="team-reset-retry"' in response.text
    assert "Try failed films again" in response.text
    assert "Generate again" in response.text
    assert 'id="teams-note"' in response.text
    assert 'id="report"' in response.text
    assert "Scouting report" in response.text
    assert 'id="report-generate"' in response.text
    assert "Generate scouting report" in response.text
    assert 'id="report-options"' not in response.text
    assert "Play tendency table" not in response.text
    assert "Choose at least one part of the report." not in response.text
    sections = '["tendencies", "calls", "special", "plays", "formations"]'
    assert f"const REPORT_SECTIONS = {sections}" in response.text
    assert 'id="report-view"' in response.text
    assert 'id="report-analysis-table"' in response.text
    assert "Scouting analysis" in response.text
    assert "Start here" in response.text
    assert "Save trim" in response.text
    assert 'id="play-trim"' in response.text
    assert "The next play opens when the clip ends." in response.text
    assert "Download PDF" in response.text
    assert "Download CSV" in response.text
    assert 'id="report-downloads" class="exports" hidden' in response.text
    library = client.get("/projects")
    assert library.status_code == 200
    assert ">Projects<" in library.text
    assert client.get("/analyses").status_code == 200
    assert client.get("/activity").status_code == 200
    assert client.get("/settings").status_code == 200
    assert client.get("/api/settings").json() == {
        "workers": 4,
        "place": "local",
        "cloud": False,
        "formations": True,
    }
    assert client.get("/api/activity").json() == {
        "workers": 4,
        "place": "local",
        "cloud": False,
        "formations": True,
        "jobs": [],
    }
    assert "Couldn't reach the server" in response.text
    assert "Analysis complete" in response.text
    assert "Finding plays" in response.text
    assert "Checking the film" in response.text
    assert 'id="download-track"' in response.text
    assert 'id="download-pct"' in response.text
    assert "Retry download" in response.text
    assert "Needs a look" in response.text
    assert "Offense / defense" in response.text
    assert "canRetryJob" in response.text
    assert 'label: "Stopped"' in response.text
    assert 'id="film-error"' in response.text
    assert 'id="read-notes"' in response.text
    assert 'project.error || "Could not read that film."' in response.text
    assert "The server stopped while this was running." in response.text
    assert 'createElement("video")' in response.text
    assert "video.muted = true" in response.text
    assert 'id="play-video" controls playsinline preload="metadata" muted' in response.text
    assert "View play" in response.text
    assert 'id="play-details"' in response.text
    assert 'id="play-tracker"' in response.text
    assert "Mark reviewed" in response.text
    assert "/api/plays/" in response.text


def test_upload_saves_the_analysis_and_its_play_clip() -> None:
    agent = _Agent((PlaySegment(number=1, start_seconds=1, end_seconds=2),))
    client = TestClient(
        create_app(
            agent=agent,
            prepare=_keep_source,
            cut=_copy_clip,
            duration=_duration,
            frames=_frames,
        )
    )

    response = client.post(
        "/analyze",
        files={"video": ("clip.mp4", b"fake-video", "video/mp4")},
    )

    assert response.status_code == 200
    assert response.json()["project"] == 1
    assert response.json()["name"] == "clip.mp4"
    body = _settle(client, 1)
    assert body["status"] == "ready"
    assert body["plays"] == [
        {
            "uid": "1-1",
            "play_number": 1,
            "label": "Play 1: 0:01 - 0:02",
            "start": 1.0,
            "end": 2.0,
            "room_start": 0.0,
            "room_end": 4.0,
            "family": "scrimmage",
            "play_type": None,
            "unit": None,
            "formation": None,
            "formation_key": None,
            "formation_plot": None,
            "steps": {
                "play": "detected",
                "side": "open",
                "formation": "open",
                "reviewed": "open",
            },
            "formation_choices": _FORMATION_CHOICES,
            "reviewed": False,
            "discarded": False,
            "video": "/projects/1/plays/1",
        }
    ]
    film = client.get("/projects/1/plays/1")
    assert film.status_code == 200
    assert film.content == b"fake-video"
    assert film.headers["content-type"].startswith("video/mp4")
    saved = client.get("/api/projects").json()
    created = saved[0].pop("created")
    assert isinstance(created, str) and "T" in created
    assert saved == [
        {
            "project": 1,
            "name": "clip.mp4",
            "play_count": 1,
            "thumbnail": "/projects/1/poster",
            "status": "ready",
            "downloaded": 0,
            "total_bytes": None,
            "youtube": False,
            "running": False,
            "phase": "",
            "report_done": 0,
            "report_total": 0,
            "error": None,
            "scanned": 4.0,
            "review_count": 0,
            "duration": 4.0,
            "archived": False,
            "retries": 0,
            "retries_done": 0,
            "processing": 100,
        }
    ]

    again = client.post(
        "/analyze",
        files={"video": ("second.mp4", b"other-video", "video/mp4")},
    )
    assert again.json()["project"] == 2
    listed = client.get("/api/projects").json()
    assert [item["project"] for item in listed] == [2, 1]
    assert client.get("/projects/1/plays/1").content == b"fake-video"


def test_team_analysis_starts_when_the_plays_are_ready() -> None:
    agent = _Agent((PlaySegment(number=1, start_seconds=1, end_seconds=2),))
    client = TestClient(
        create_app(
            agent=agent,
            prepare=_keep_source,
            cut=_copy_clip,
            duration=_duration,
            frames=_frames,
            lineup=lambda video, at: (b"left", b"right"),
            portrait=lambda uniform, still=None, other=None: (
                f"<svg>{uniform.name}</svg>".encode(),
                ".svg",
            ),
        )
    )

    response = client.post(
        "/analyze",
        files={"video": ("clip.mp4", b"fake-video", "video/mp4")},
    )
    assert response.status_code == 200
    body = _settle(client, 1)

    assert body["status"] == "ready"
    teams = body["teams"]
    assert isinstance(teams, dict)
    assert teams["left_name"]
    assert teams["right_name"]
    assert teams["left_still"]
    assert teams["right_still"]
    assert client.get(str(teams["left_still"])).content == b"left"
    assert client.get(str(teams["right_still"])).content == b"right"
    # A card is kept only when the crop is one standing player. These stills
    # are not images, so neither side gets a card and the step says so.
    assert teams["left_player"] is None
    assert teams["right_player"] is None
    sides = teams["progress"]["sides"]
    assert sides["left"]["status"] == "missed"
    assert sides["right"]["status"] == "missed"


def test_archive_takes_a_film_off_the_shelf_until_it_is_put_back(tmp_path: Path) -> None:
    store = MemoryCatalog(tmp_path)
    kept = store.create_project("Kept", None)
    away = store.create_project("Away", None)
    for project_id in (kept, away):
        store.set_progress(
            project_id,
            status="ready",
            duration_seconds=10,
            scanned_seconds=10,
            error=None,
        )
    client = TestClient(create_app(agent=_Agent(), store=store))

    archived = client.post(f"/api/projects/{away}/archive", json={"archived": True})
    assert archived.status_code == 200
    assert archived.json() == {"project": away, "archived": True}
    by_id = {item["project"]: item["archived"] for item in client.get("/api/projects").json()}
    assert by_id == {kept: False, away: True}
    store.set_progress(
        away,
        status="ready",
        duration_seconds=12,
        scanned_seconds=12,
        error=None,
    )
    assert client.get("/api/projects").json()[0]["archived"] is True
    jobs = {job["project"] for job in client.get("/api/activity").json()["jobs"]}
    assert away not in jobs
    assert kept in jobs

    restored = client.post(f"/api/projects/{away}/archive", json={"archived": False})
    assert restored.status_code == 200
    assert restored.json()["archived"] is False
    assert client.post("/api/projects/99/archive", json={"archived": True}).status_code == 404


def test_an_unreadable_window_stays_on_the_film() -> None:
    class _Messages:
        def create(self, **kwargs: object) -> SimpleNamespace:
            del kwargs
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="The scoreboard was too small.")]
            )

    agent = AnthropicPlayAgent(
        api_key="test-key",
        client=SimpleNamespace(messages=_Messages()),
        frames=lambda path: [],
    )
    client = TestClient(
        create_app(
            agent=agent,
            prepare=_keep_source,
            cut=_copy_clip,
            duration=_duration,
            frames=_frames,
        )
    )

    response = client.post(
        "/analyze",
        files={"video": ("clip.mp4", b"fake-video", "video/mp4")},
    )

    body = _settle(client, response.json()["project"])
    assert body["status"] == "ready"
    assert body["reads"] == [
        {
            "start": 0.0,
            "end": 3.0,
            "detail": (
                "2 frames, 8 B. film spans were not valid JSON. "
                "Reply: The scoreboard was too small."
            ),
        }
    ]


def test_youtube_link_saves_a_project() -> None:
    agent = _Agent((PlaySegment(number=1, start_seconds=1, end_seconds=2),))

    def download(url: str, directory: Path) -> Path:
        agent.urls.append(url)
        path = directory / "film.mp4"
        path.write_bytes(b"clip-bytes")
        return path

    client = TestClient(
        create_app(
            agent=agent,
            downloader=download,
            prepare=_keep_source,
            cut=_copy_clip,
            duration=_duration,
            frames=_frames,
            details=lambda url: None,
        )
    )

    response = client.post("/analyze", data={"youtube": "https://youtu.be/abcdefghijk"})

    assert response.status_code == 200
    body = _settle(client, response.json()["project"])
    assert body["name"] == "https://www.youtube.com/watch?v=abcdefghijk"
    assert body["youtube_url"] == "https://www.youtube.com/watch?v=abcdefghijk"
    assert body["plays"][0]["label"] == "Play 1: 0:01 - 0:02"
    assert client.get(body["plays"][0]["video"]).content == b"clip-bytes"
    assert agent.urls == ["https://www.youtube.com/watch?v=abcdefghijk"]
    assert body["thumbnail"] == "https://i.ytimg.com/vi/abcdefghijk/maxresdefault.jpg"


def test_a_finished_cut_uses_an_action_still_for_the_film_and_the_list(tmp_path: Path) -> None:
    shot = b"\xff\xd8\xff\xd9action"
    catalog = MemoryCatalog(tmp_path)

    def grab(path: Path, at: float) -> bytes:
        del path, at
        return shot

    client = TestClient(
        create_app(
            agent=_Agent((PlaySegment(number=1, start_seconds=1, end_seconds=6),)),
            downloader=_download_clip,
            prepare=_keep_source,
            cut=_copy_clip,
            duration=lambda path: 12.0,
            frames=_frames,
            details=lambda url: None,
            store=catalog,
            still=grab,
        )
    )

    response = client.post("/analyze", data={"youtube": "https://youtu.be/abcdefghijk"})
    project_id = response.json()["project"]
    body = _settle(client, project_id)

    assert body["status"] == "ready"
    assert body["thumbnail"] == f"/projects/{project_id}/poster"
    assert client.get(body["thumbnail"]).content == shot
    listed = client.get("/api/projects").json()
    assert listed[0]["thumbnail"] == body["thumbnail"]


def test_the_action_still_replaces_youtube_art_while_the_second_look_runs(
    tmp_path: Path,
) -> None:
    from film_analysis.storage.catalog import PlayRecord, ProjectDetail
    from film_analysis.web.app import _thumbnail_for

    catalog = MemoryCatalog(tmp_path)
    project_id = catalog.create_project("Friday night", "https://youtu.be/abcdefghijk")
    project = ProjectDetail(
        project=project_id,
        name="Friday night",
        youtube_url="https://www.youtube.com/watch?v=abcdefghijk",
        plays=(PlayRecord(1, 10.0, 18.0, "scrimmage"),),
        status="scanning",
    )
    youtube = "https://i.ytimg.com/vi/abcdefghijk/maxresdefault.jpg"

    assert _thumbnail_for(project, catalog) == youtube

    catalog.save_poster(project_id, b"\xff\xd8\xffstill")

    assert _thumbnail_for(project, catalog) == f"/projects/{project_id}/poster"
    page = TestClient(create_app(agent=_Agent(), store=catalog, details=lambda url: None))
    picture = page.get(f"/projects/{project_id}/poster")
    assert picture.status_code == 200
    assert picture.content == b"\xff\xd8\xffstill"


def _download_clip(url: str, directory: Path) -> Path:
    del url
    path = directory / "film.mp4"
    path.write_bytes(b"clip-bytes")
    return path


def test_youtube_title_and_length_show_while_the_file_downloads() -> None:
    release = threading.Event()

    def download(url: str, directory: Path) -> Path:
        assert release.wait(timeout=2)
        path = directory / "film.mp4"
        path.write_bytes(b"clip-bytes")
        return path

    def details(url: str) -> FilmInfo:
        del url
        return FilmInfo("Carrollton vs. Buford", "Prep Highlights", 11119)

    client = TestClient(
        create_app(
            agent=_Agent((PlaySegment(number=1, start_seconds=1, end_seconds=2),)),
            downloader=download,
            cut=_copy_clip,
            duration=_duration,
            frames=_frames,
            details=details,
        )
    )

    response = client.post("/analyze", data={"youtube": "https://youtu.be/abcdefghijk"})
    project_id = response.json()["project"]
    shown: dict[str, object] = {}
    for _ in range(50):
        shown = client.get(f"/api/projects/{project_id}").json()
        if shown.get("channel"):
            break
        time.sleep(0.02)
    assert shown["status"] == "downloading"
    assert shown["name"] == "Carrollton vs. Buford"
    assert shown["channel"] == "Prep Highlights"
    assert shown["duration"] == 11119
    assert shown["thumbnail"] == "https://i.ytimg.com/vi/abcdefghijk/maxresdefault.jpg"
    release.set()
    assert _settle(client, project_id)["status"] == "ready"


def test_download_progress_shows_bytes_received() -> None:
    release = threading.Event()
    received = 120 * 1024 * 1024
    total = 660 * 1024 * 1024

    def download(url: str, directory: Path, progress) -> Path:
        del url
        progress(received, total)
        assert release.wait(timeout=2)
        path = directory / "film.mp4"
        path.write_bytes(b"clip-bytes")
        return path

    client = TestClient(
        create_app(
            agent=_Agent((PlaySegment(number=1, start_seconds=1, end_seconds=2),)),
            downloader=download,
            cut=_copy_clip,
            duration=_duration,
            frames=_frames,
            details=lambda url: None,
        )
    )

    response = client.post("/analyze", data={"youtube": "https://youtu.be/abcdefghijk"})
    project_id = response.json()["project"]
    shown: dict[str, object] = {}
    for _ in range(50):
        shown = client.get(f"/api/projects/{project_id}").json()
        if shown.get("downloaded"):
            break
        time.sleep(0.02)
    assert shown["status"] == "downloading"
    assert shown["downloaded"] == received
    assert shown["total_bytes"] == total
    listed = client.get("/api/projects").json()
    assert listed[0]["downloaded"] == received
    assert listed[0]["total_bytes"] == total
    assert listed[0]["play_count"] == 0
    assert listed[0]["thumbnail"] == shown["thumbnail"]
    release.set()
    assert _settle(client, project_id)["status"] == "ready"


def test_a_failed_download_can_be_retried() -> None:
    calls = {"n": 0}

    def download(url: str, directory: Path) -> Path:
        del url
        calls["n"] += 1
        if calls["n"] == 1:
            raise VideoAnalysisError("network dropped")
        path = directory / "film.mp4"
        path.write_bytes(b"clip-bytes")
        return path

    client = TestClient(
        create_app(
            agent=_Agent((PlaySegment(number=1, start_seconds=1, end_seconds=2),)),
            downloader=download,
            cut=_copy_clip,
            duration=_duration,
            frames=_frames,
            details=lambda url: None,
        )
    )
    response = client.post("/analyze", data={"youtube": "https://youtu.be/abcdefghijk"})
    project_id = response.json()["project"]
    failed = _settle(client, project_id)
    assert failed["status"] == "failed"
    assert failed["running"] is False
    error = str(failed["error"])
    assert error.startswith("Stopped while downloading.\nVideoAnalysisError: network dropped")
    assert "at film_analysis/" in error

    again = client.post(f"/api/projects/{project_id}/retry")

    assert again.status_code == 200
    assert again.json()["running"] is True
    assert _settle(client, project_id)["status"] == "ready"
    assert calls["n"] == 2


def test_retry_waits_until_the_current_download_finishes() -> None:
    release = threading.Event()

    def download(url: str, directory: Path) -> Path:
        del url
        assert release.wait(timeout=2)
        path = directory / "film.mp4"
        path.write_bytes(b"clip-bytes")
        return path

    client = TestClient(
        create_app(
            agent=_Agent((PlaySegment(number=1, start_seconds=1, end_seconds=2),)),
            downloader=download,
            cut=_copy_clip,
            duration=_duration,
            frames=_frames,
            details=lambda url: None,
        )
    )
    response = client.post("/analyze", data={"youtube": "https://youtu.be/abcdefghijk"})
    project_id = response.json()["project"]
    assert response.json()["running"] is True

    blocked = client.post(f"/api/projects/{project_id}/retry")

    assert blocked.status_code == 409
    release.set()
    assert _settle(client, project_id)["status"] == "ready"


def test_activity_shows_the_download_the_server_is_running() -> None:
    release = threading.Event()

    def download(url: str, directory: Path, progress) -> Path:
        del url
        progress(3, 10)
        assert release.wait(timeout=2)
        path = directory / "film.mp4"
        path.write_bytes(b"clip-bytes")
        return path

    client = TestClient(
        create_app(
            agent=_Agent((PlaySegment(number=1, start_seconds=1, end_seconds=2),)),
            downloader=download,
            cut=_copy_clip,
            duration=_duration,
            frames=_frames,
            details=lambda url: None,
        )
    )
    response = client.post("/analyze", data={"youtube": "https://youtu.be/abcdefghijk"})
    project_id = response.json()["project"]
    deadline = time.monotonic() + 2
    body: dict[str, object] = {}
    while time.monotonic() < deadline:
        body = client.get("/api/activity").json()
        jobs = body["jobs"]
        if isinstance(jobs, list) and jobs and jobs[0]["phase"] == "downloading":
            break
        time.sleep(0.02)

    jobs = body["jobs"]
    assert isinstance(jobs, list)
    assert jobs[0]["project"] == project_id
    assert jobs[0]["phase"] == "downloading"
    assert jobs[0]["running"] is True
    assert jobs[0]["downloaded"] == 3
    assert jobs[0]["total_bytes"] == 10
    assert body["workers"] == 4
    release.set()
    assert _settle(client, project_id)["status"] == "ready"
    finished = client.get("/api/activity").json()["jobs"]
    assert len(finished) == 1
    assert finished[0]["project"] == project_id
    assert finished[0]["phase"] == "ready"
    assert finished[0]["running"] is False
    assert finished[0]["play_count"] == 1


def test_jobs_can_run_locally_or_in_the_cloud(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FILM_INTERNAL_TOKEN", "secret")
    store = MemoryCatalog(tmp_path)
    headers = {"x-film-worker-url": "https://film.example"}
    client = TestClient(
        create_app(
            agent=_Agent((PlaySegment(number=1, start_seconds=1, end_seconds=2),)),
            store=store,
            downloader=_saved_download,
            cut=_copy_clip,
            duration=_duration,
            frames=_frames,
            details=lambda url: None,
        )
    )

    assert client.get("/api/activity", headers=headers).json()["place"] == "local"
    assert client.get("/api/activity", headers=headers).json()["cloud"] is False
    cloud_headers = {**headers, "x-film-delegate": "1"}
    assert client.get("/api/activity", headers=cloud_headers).json()["place"] == "cloud"
    assert client.get("/api/activity", headers=cloud_headers).json()["cloud"] is True
    refused = client.post("/api/settings", json={"place": "cloud"}, headers=headers)
    assert refused.status_code == 400
    local = client.post(
        "/analyze",
        data={"youtube": "https://youtu.be/abcdefghijk"},
        headers=headers,
    )
    assert local.status_code == 200
    assert "x-film-start" not in local.headers
    local_film = _settle(client, local.json()["project"])
    assert local_film["status"] == "ready"
    assert local_film["steps"]
    assert {step["stack"] for step in local_film["steps"]} == {"local"}

    client.post("/api/settings", json={"workers": 6}, headers=cloud_headers)
    handed = client.post(
        "/analyze",
        data={"youtube": "https://youtu.be/abcdefghijk"},
        headers=cloud_headers,
    )
    assert handed.status_code == 200
    assert handed.headers["x-film-start"] == str(handed.json()["project"])
    assert handed.headers["x-film-workers"] == "6"
    cloud_job = next(
        job
        for job in client.get("/api/activity", headers=headers).json()["jobs"]
        if job["project"] == handed.json()["project"]
    )
    starting = next(step for step in cloud_job["steps"] if step["name"] == "starting")
    assert starting["stack"] == "cloud"
    assert starting["workers"] == 6
    assert client.get(f"/api/projects/{handed.json()['project']}").json()["status"] == "downloading"

    direct = client.post(
        "/analyze",
        files={"video": ("clip.mp4", b"fake-video", "video/mp4")},
        headers=cloud_headers,
    )
    assert direct.status_code == 200
    assert direct.headers["x-film-start"] == str(direct.json()["project"])


def _saved_download(url: str, directory: Path) -> Path:
    del url
    path = directory / "film.mp4"
    path.write_bytes(b"clip-bytes")
    return path


def test_a_job_started_on_this_page_is_sent_to_cloudflare(monkeypatch) -> None:
    class _Upstream:
        status = 200
        headers = {"Content-Type": "application/json", "Content-Length": "2"}

        def __init__(self) -> None:
            self._body = b'{"project": 9, "status": "downloading"}'

        def read(self, _size: int = -1) -> bytes:
            data, self._body = self._body, b""
            return data

        def close(self) -> None:
            return None

    def _open(request, timeout=300):
        assert timeout == 300
        assert request.get_method() == "POST"
        assert request.full_url == (
            "https://film-analysis.broad-night-d47d.workers.dev/analyze"
        )
        assert b"abcdefghijk" in request.data
        return _Upstream()

    monkeypatch.setattr("film_analysis.web.app.urllib.request.urlopen", _open)
    client = TestClient(create_app(agent=_Agent()))

    response = client.post("/via-cloud/analyze", data={"youtube": "https://youtu.be/abcdefghijk"})

    assert response.status_code == 200
    assert response.json()["project"] == 9
    assert response.json()["status"] == "downloading"


def test_the_local_page_forwards_a_cloud_request(monkeypatch) -> None:
    class _Upstream:
        status = 200
        headers = {"Content-Type": "application/json", "Content-Length": "2"}

        def read(self, _size: int = -1) -> bytes:
            if self.headers["Content-Length"] == "0":
                return b""
            self.headers["Content-Length"] = "0"
            return b"{}"

        def close(self) -> None:
            return None

    def _open(request, timeout=120):
        del timeout
        assert request.full_url == "https://film-analysis.broad-night-d47d.workers.dev/api/settings"
        return _Upstream()

    monkeypatch.setattr("film_analysis.web.app.urllib.request.urlopen", _open)
    client = TestClient(create_app(agent=_Agent()))

    response = client.get("/via-cloud/api/settings")

    assert response.status_code == 200
    assert response.json() == {}


def test_worker_setting_is_saved_and_reported(tmp_path: Path) -> None:
    store = MemoryCatalog(tmp_path)
    client = TestClient(create_app(agent=_Agent(), store=store))

    assert client.get("/api/activity").json()["workers"] == 4
    saved = client.post("/api/settings", json={"workers": 8})
    assert saved.status_code == 200
    assert saved.json() == {"workers": 8, "place": "local", "cloud": False, "formations": True}
    formed = client.post("/api/settings", json={"formations": False})
    assert formed.status_code == 200
    assert formed.json()["formations"] is False
    assert store.formation_reads() is False
    enabled = client.post("/api/settings", json={"formations": True})
    assert enabled.status_code == 200
    assert enabled.json()["formations"] is True
    assert store.formation_reads() is True
    assert client.get("/api/activity").json()["workers"] == 8
    assert store.worker_count() == 8
    restarted = TestClient(create_app(agent=_Agent(), store=store))
    assert restarted.get("/api/activity").json()["workers"] == 8
    assert client.post("/api/settings", json={"workers": 0}).status_code == 422
    assert client.post("/api/settings", json={"workers": 20}).status_code == 422
    assert client.post("/api/settings", json={}).status_code == 422
    assert client.post("/api/settings", json={"place": "lab"}).status_code == 422
    blocked = client.post("/api/settings", json={"place": "cloud"})
    assert blocked.status_code == 400
    assert "Cloudflare" in blocked.json()["detail"]
    assert store.job_place() is None


def test_activity_graphs_the_cost_of_each_play(tmp_path: Path) -> None:
    store = MemoryCatalog(tmp_path)
    project_id = store.create_project("Friday night", None)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")
    store.add_play(project_id, 1, 10, 20, clip)
    store.add_play(project_id, 2, 30, 40, clip)
    store.add_read(
        project_id,
        ReadNote(
            0,
            50,
            4,
            100,
            "",
            input_tokens=2000,
            output_tokens=200,
            cost_usd=0.006,
            model="claude-sonnet-5-5",
        ),
    )
    store.set_progress(
        project_id,
        status="ready",
        duration_seconds=50,
        scanned_seconds=50,
        error=None,
    )
    client = TestClient(create_app(agent=_Agent(), store=store))

    job = client.get("/api/activity").json()["jobs"][0]
    costs = job["play_costs"]

    assert costs == [{"play": 1, "usd": 0.003}, {"play": 2, "usd": 0.003}]
    assert job["read_cost"] == 0.006
    assert job["read_model"] == "claude-sonnet-5-5"
    assert job["input_tokens"] == 2000
    assert job["output_tokens"] == 200


def test_activity_reads_the_saved_phase(tmp_path: Path) -> None:
    store = MemoryCatalog(tmp_path)
    project_id = store.create_project(
        "Pike County Football vs. Lamar County",
        "https://www.youtube.com/watch?v=abcdefghijk",
    )
    store.set_progress(
        project_id,
        status="ready",
        duration_seconds=120,
        scanned_seconds=120,
        error=None,
    )
    client = TestClient(create_app(agent=_Agent(), store=store))
    finished = client.get("/api/activity").json()["jobs"]
    assert len(finished) == 1
    assert finished[0]["phase"] == "ready"
    assert finished[0]["running"] is False

    store.record_activity(project_id, "scanning", running=True)
    jobs = client.get("/api/activity").json()["jobs"]

    assert len(jobs) == 1
    assert jobs[0]["project"] == project_id
    assert jobs[0]["phase"] == "scanning"
    assert jobs[0]["running"] is True
    assert jobs[0]["updated"]


def test_activity_lists_complete_stopped_and_running_newest_first(tmp_path: Path) -> None:
    store = MemoryCatalog(tmp_path)
    older = store.create_project("Older film", None)
    store.set_progress(
        older,
        status="ready",
        duration_seconds=10,
        scanned_seconds=10,
        error=None,
    )
    middle = store.create_project("Middle film", "https://youtu.be/abcdefghijk")
    store.set_progress(
        middle,
        status="failed",
        duration_seconds=10,
        scanned_seconds=4,
        error="Stopped while reading.",
    )
    newer = store.create_project("Newer film", None)
    store.set_progress(
        newer,
        status="scanning",
        duration_seconds=10,
        scanned_seconds=2,
        error=None,
    )
    client = TestClient(create_app(agent=_Agent(), store=store))
    store.record_activity(older, "ready", running=False, updated_at="2026-11-01T00:00:00+00:00")
    store.record_activity(middle, "stopped", running=False, updated_at="2026-11-02T00:00:00+00:00")
    store.record_activity(newer, "scanning", running=True, updated_at="2026-11-03T00:00:00+00:00")
    plain = store.create_project("Plain complete", None)
    store.set_progress(
        plain,
        status="ready",
        duration_seconds=8,
        scanned_seconds=8,
        error=None,
    )

    jobs = client.get("/api/activity").json()["jobs"]

    assert [job["name"] for job in jobs] == [
        "Plain complete",
        "Newer film",
        "Middle film",
        "Older film",
    ]
    assert [job["phase"] for job in jobs] == ["ready", "scanning", "stopped", "ready"]
    assert [job["running"] for job in jobs] == [False, True, False, False]
    assert jobs[0]["updated"] > jobs[1]["updated"] > jobs[2]["updated"] > jobs[3]["updated"]


def test_activity_times_each_step(tmp_path: Path) -> None:
    store = MemoryCatalog(tmp_path)
    project_id = store.create_project(
        "Pike County Football vs. Lamar County",
        "https://www.youtube.com/watch?v=abcdefghijk",
    )
    store.record_activity(
        project_id,
        "downloading",
        running=True,
        updated_at="2026-11-09T05:00:00+00:00",
    )
    store.record_activity(
        project_id,
        "downloading",
        running=True,
        updated_at="2026-11-09T05:00:40+00:00",
    )
    store.record_activity(
        project_id,
        "saving",
        running=True,
        updated_at="2026-11-09T05:01:30+00:00",
    )
    store.record_activity(
        project_id,
        "scanning",
        running=True,
        updated_at="2026-11-09T05:01:34+00:00",
    )
    store.record_activity(
        project_id,
        "checking",
        running=True,
        updated_at="2026-11-09T05:10:00+00:00",
    )
    store.record_activity(
        project_id,
        "ready",
        running=False,
        updated_at="2026-11-09T05:12:05+00:00",
    )
    client = TestClient(create_app(agent=_Agent(), store=store))

    jobs = client.get("/api/activity").json()["jobs"]
    job = next(item for item in jobs if item["project"] == project_id)
    steps = {step["name"]: step for step in job["steps"]}

    assert [step["name"] for step in job["steps"]] == [
        "downloading",
        "saving",
        "scanning",
        "checking",
    ]
    assert steps["downloading"]["seconds"] == 90
    assert steps["downloading"]["running"] is False
    assert steps["saving"]["seconds"] == 4
    assert steps["scanning"]["seconds"] == 506
    assert steps["checking"]["seconds"] == 125
    assert steps["checking"]["running"] is False
    film = client.get(f"/api/projects/{project_id}").json()
    assert [step["name"] for step in film["steps"]] == [
        "downloading",
        "saving",
        "scanning",
        "checking",
    ]
    assert steps["downloading"]["stack"] is None


def test_a_step_keeps_the_stack_that_started_it(tmp_path: Path) -> None:
    store = MemoryCatalog(tmp_path)
    project_id = store.create_project("Week 6", "https://youtu.be/abcdefghijk")
    store.record_activity(
        project_id,
        "downloading",
        running=True,
        stack="cloud",
        updated_at="2026-11-09T05:00:00+00:00",
    )
    store.record_activity(
        project_id,
        "downloading",
        running=True,
        stack="local",
        updated_at="2026-11-09T05:00:30+00:00",
    )
    store.record_activity(
        project_id,
        "scanning",
        running=True,
        stack="cloud",
        workers=8,
        updated_at="2026-11-09T05:01:00+00:00",
    )
    store.record_activity(
        project_id,
        "ready",
        running=False,
        updated_at="2026-11-09T05:04:00+00:00",
    )
    client = TestClient(create_app(agent=_Agent(), store=store))
    job = client.get("/api/activity").json()["jobs"][0]
    saved = {step["name"]: step for step in job["steps"]}

    assert saved["downloading"]["stack"] == "cloud"
    assert saved["downloading"]["workers"] is None
    assert saved["scanning"]["stack"] == "cloud"
    assert saved["scanning"]["workers"] == 8
    legacy = decode_activity_steps(
        '[{"name":"downloading","started":"2026-11-09T05:00:00+00:00","ended":null}]'
    )
    assert legacy[0].stack is None
    assert legacy[0].workers is None


def test_a_step_with_no_place_takes_the_next_one(tmp_path: Path) -> None:
    store = MemoryCatalog(tmp_path)
    project_id = store.create_project("Week 6", "https://youtu.be/abcdefghijk")
    store.record_activity(
        project_id,
        "starting",
        running=True,
        updated_at="2026-11-09T05:00:00+00:00",
    )
    store.record_activity(
        project_id,
        "starting",
        running=True,
        stack="local",
        updated_at="2026-11-09T05:00:20+00:00",
    )
    saved = {step.name: step for step in store.get_activity(project_id).steps}
    assert saved["starting"].stack == "local"


def test_activity_shows_the_error_that_stopped_a_job(tmp_path: Path) -> None:
    store = MemoryCatalog(tmp_path)
    reading = store.create_project(
        "Pike County Football vs. Lamar County",
        "https://www.youtube.com/watch?v=abcdefghijk",
    )
    store.set_progress(
        reading,
        status="scanning",
        duration_seconds=1000,
        scanned_seconds=20,
        error=None,
    )
    downloading = store.create_project(
        "Pike County Football vs. Lamar County",
        "https://www.youtube.com/watch?v=bbcdefghijk",
    )
    store.set_progress(
        downloading,
        status="downloading",
        duration_seconds=None,
        scanned_seconds=0,
        error=None,
    )
    store.set_download(downloading, 50 * 1024 * 1024, 581 * 1024 * 1024)
    blocked = store.create_project(
        "https://www.youtube.com/watch?v=0VdS3kg5RHU",
        "https://www.youtube.com/watch?v=0VdS3kg5RHU",
    )
    store.set_progress(
        blocked,
        status="failed",
        duration_seconds=14305,
        scanned_seconds=13300,
        error="Sign in to confirm you're not a bot.",
    )

    def download(url: str, directory: Path) -> Path:
        del url
        path = directory / "film.mp4"
        path.write_bytes(b"clip-bytes")
        return path

    client = TestClient(
        create_app(
            agent=_Agent((PlaySegment(number=1, start_seconds=1, end_seconds=2),)),
            downloader=download,
            cut=_copy_clip,
            duration=_duration,
            frames=_frames,
            details=lambda url: None,
            store=store,
        )
    )
    jobs = {job["project"]: job for job in client.get("/api/activity").json()["jobs"]}

    assert jobs[reading]["status"] == "scanning"
    assert jobs[reading]["running"] is False
    assert jobs[reading]["scanned"] == 20
    assert jobs[reading]["youtube"] is True
    assert jobs[reading]["error"] == "The server stopped while this was running."
    assert jobs[downloading]["downloaded"] == 50 * 1024 * 1024
    assert jobs[downloading]["error"] == "The server stopped while this was running."
    assert jobs[blocked]["error"] == "Sign in to confirm you're not a bot."
    assert jobs[blocked]["status"] == "failed"

    again = client.post(f"/api/projects/{reading}/retry")

    assert again.status_code == 200
    assert again.json()["running"] is True
    assert _settle(client, reading)["status"] == "ready"


def test_retry_continues_a_scan_from_where_it_stopped(tmp_path: Path) -> None:
    store = MemoryCatalog(tmp_path)
    project = store.create_project("Game film", "https://youtu.be/abcdefghijk")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"film-bytes")
    store.save_source(project, source)
    clip = tmp_path / "play.mp4"
    clip.write_bytes(b"play")
    store.add_play(project, 1, 2, 8, clip)
    store.set_progress(
        project,
        status="failed",
        duration_seconds=100,
        scanned_seconds=30,
        error="The model timed out.",
    )
    seen: list[float] = []

    def download(url: str, directory: Path) -> Path:
        del url, directory
        raise AssertionError("the scan should resume without downloading")

    def frames(path: Path, start: float, seconds: float) -> list[tuple[float, bytes]]:
        del path, seconds
        seen.append(start)
        return [(start, b"jpeg"), (start + 30, b"jpeg")]

    client = TestClient(
        create_app(
            agent=_Agent(
                (
                    PlaySegment(number=1, start_seconds=32, end_seconds=40),
                    PlaySegment(number=2, start_seconds=62, end_seconds=70),
                )
            ),
            downloader=download,
            cut=_copy_clip,
            duration=lambda path: 100.0,
            frames=frames,
            details=lambda url: None,
            store=store,
        )
    )

    again = client.post(f"/api/projects/{project}/retry")

    assert again.status_code == 200
    done = _settle(client, project)
    assert done["status"] == "ready"
    assert sorted(seen) == [30.0, 60.0]
    assert [(play["play_number"], play["start"], play["end"]) for play in done["plays"]] == [
        (1, 2.0, 8.0),
        (2, 32.0, 40.0),
        (3, 62.0, 70.0),
    ]


def test_unknown_analysis_is_missing() -> None:
    client = TestClient(create_app(agent=_Agent()))

    assert client.get("/api/projects/9").status_code == 404
    assert client.get("/projects/9/plays/1").status_code == 404


def test_youtube_link_must_be_youtube() -> None:
    agent = _Agent()

    response = TestClient(create_app(agent=agent)).post(
        "/analyze",
        data={"youtube": "https://vimeo.com/123456789"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Paste a YouTube link."
    assert agent.urls == []


def test_file_and_youtube_link_together_are_rejected() -> None:
    agent = _Agent()

    response = TestClient(create_app(agent=agent)).post(
        "/analyze",
        data={"youtube": "https://youtu.be/abcdefghijk"},
        files={"video": ("clip.mp4", b"fake-video", "video/mp4")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Use a file, a YouTube link, or a downloaded film."
    assert agent.urls == []
    assert agent.paths == []


def test_missing_source_is_rejected() -> None:
    response = TestClient(create_app(agent=_Agent())).post("/analyze", data={"youtube": "  "})

    assert response.status_code == 400
    assert (
        response.json()["detail"]
        == "Upload a video, paste a YouTube link, or choose a downloaded film."
    )


def test_a_downloaded_film_can_be_analyzed_again(tmp_path: Path) -> None:
    store = MemoryCatalog(tmp_path)
    calls = {"n": 0}

    def download(url: str, directory: Path) -> Path:
        del url, directory
        calls["n"] += 1
        raise AssertionError("the saved film should not be downloaded again")

    client = TestClient(
        create_app(
            agent=_Agent((PlaySegment(number=1, start_seconds=1, end_seconds=2),)),
            downloader=download,
            cut=_copy_clip,
            duration=_duration,
            frames=_frames,
            details=lambda url: None,
            store=store,
        )
    )
    uploaded = client.post(
        "/analyze",
        files={"video": ("clip.mp4", b"fake-video", "video/mp4")},
    )
    assert _settle(client, uploaded.json()["project"])["status"] == "ready"
    assert client.get("/api/downloads").json() == [{"project": 1, "name": "clip.mp4"}]

    again = client.post("/analyze", data={"saved": "1"})

    assert again.status_code == 200
    body = _settle(client, again.json()["project"])
    assert body["project"] == 2
    assert body["status"] == "ready"
    assert body["name"] == "clip.mp4"
    assert [play["play_number"] for play in body["plays"]] == [1]
    assert calls["n"] == 0
    source = store.source_file(2)
    assert source is not None
    assert source.read_bytes() == b"fake-video"
    missing = client.post("/analyze", data={"saved": "9"})
    assert missing.status_code == 404


def test_non_video_upload_is_rejected() -> None:
    response = TestClient(create_app(agent=_Agent())).post(
        "/analyze",
        files={"video": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Upload a video file."


def test_oversize_upload_is_rejected() -> None:
    response = TestClient(create_app(agent=_Agent(), max_bytes=4)).post(
        "/analyze",
        files={"video": ("clip.mp4", b"12345", "video/mp4")},
    )

    assert response.status_code == 413


def test_analyze_without_a_configured_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    response = TestClient(create_app()).post(
        "/analyze",
        files={"video": ("clip.mp4", b"fake-video", "video/mp4")},
    )

    assert response.status_code == 503
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]


def test_missing_api_key_tells_the_caller_what_to_set() -> None:
    client = TestClient(
        create_app(
            agent=_Raising(MissingApiKeyError("Set ANTHROPIC_API_KEY to analyze film.")),
            duration=_duration,
            frames=_frames,
        )
    )

    response = client.post(
        "/analyze",
        files={"video": ("clip.mp4", b"fake-video", "video/mp4")},
    )

    assert response.status_code == 200
    body = _settle(client, response.json()["project"])
    assert body["status"] == "failed"
    error = str(body["error"])
    assert "Stopped while reading 0:00–0:04 of 0:04." in error
    assert "MissingApiKeyError: Set ANTHROPIC_API_KEY to analyze film." in error
    assert "at film_analysis/" in error


def test_playback_file_is_a_seekable_mp4(tmp_path: Path) -> None:
    from film_analysis.web.playback import prepare_playback

    source = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=160x120:d=2",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )

    playable = prepare_playback(source)

    assert playable.is_file()
    assert playable.read_bytes()[4:8] == b"ftyp"
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(playable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert float(probe.stdout) == pytest.approx(2, abs=0.2)


def test_cut_clip_writes_one_play(tmp_path: Path) -> None:
    from film_analysis.web.playback import cut_clip

    source = tmp_path / "clip.mp4"
    dest = tmp_path / "play.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=160x120:d=2",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )

    cut_clip(source, 0, 1, dest)

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(dest),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert float(probe.stdout) == pytest.approx(1, abs=0.2)


def test_the_project_opens_while_the_film_is_still_being_read() -> None:
    started = threading.Event()
    release = threading.Event()

    class _Slow(_Agent):
        def label_window(self, samples: list[tuple[float, bytes]]) -> tuple[FilmSpan, ...]:
            started.set()
            assert release.wait(timeout=2)
            return super().label_window(samples)

    client = TestClient(
        create_app(
            agent=_Slow((PlaySegment(number=1, start_seconds=1, end_seconds=2),)),
            cut=_copy_clip,
            duration=_duration,
            frames=_frames,
        )
    )

    response = client.post(
        "/analyze",
        files={"video": ("clip.mp4", b"fake-video", "video/mp4")},
    )

    assert response.status_code == 200
    assert response.json()["status"] in {"downloading", "scanning"}
    assert response.json()["plays"] == []
    assert started.wait(timeout=2)
    release.set()
    body = _settle(client, response.json()["project"])
    assert body["status"] == "ready"
    assert body["plays"][0]["label"] == "Play 1: 0:01 - 0:02"


def test_a_long_film_is_read_in_overlapping_windows() -> None:
    seen: list[float] = []

    def duration(path: Path) -> float:
        del path
        return 100.0

    def frames(path: Path, start: float, seconds: float) -> list[tuple[float, bytes]]:
        del path
        seen.append(start)
        return [(start, b"jpeg"), (start + seconds - 0.1, b"jpeg")]

    agent = _Agent(
        (
            PlaySegment(number=1, start_seconds=1, end_seconds=2),
            PlaySegment(number=2, start_seconds=50, end_seconds=55),
        )
    )
    client = TestClient(
        create_app(agent=agent, cut=_copy_clip, duration=duration, frames=frames)
    )

    response = client.post(
        "/analyze",
        files={"video": ("game.mp4", b"fake-video", "video/mp4")},
    )

    body = _settle(client, response.json()["project"])
    assert body["status"] == "ready"
    assert [play["start"] for play in body["plays"]] == [1.0, 50.0]
    assert sorted(seen) == [0.0, 30.0, 60.0]


def test_unsure_footage_is_held_for_a_person_to_mark() -> None:
    class _Sorting(_Agent):
        def __init__(self) -> None:
            super().__init__()
            self.seen: list[float] = []

        def label_window(self, samples: list[tuple[float, bytes]]) -> tuple[FilmSpan, ...]:
            del samples
            return (
                FilmSpan(kind="scrimmage", start_seconds=20, end_seconds=24),
                FilmSpan(kind="junk", start_seconds=24, end_seconds=30),
                FilmSpan(kind="uncertain", start_seconds=1, end_seconds=6),
                FilmSpan(kind="uncertain", start_seconds=8, end_seconds=12),
                FilmSpan(kind="uncertain", start_seconds=32, end_seconds=38),
            )

        def review_window(self, samples: list[tuple[float, bytes]]) -> tuple[FilmSpan, ...]:
            start = samples[0][0]
            self.seen.append(start)
            if start < 2:
                return (FilmSpan(kind="special_teams", start_seconds=1, end_seconds=6),)
            if start < 10:
                return (FilmSpan(kind="uncertain", start_seconds=8, end_seconds=12),)
            return (FilmSpan(kind="uncertain", start_seconds=32, end_seconds=38),)

    agent = _Sorting()

    def duration(path: Path) -> float:
        del path
        return 40.0

    client = TestClient(
        create_app(agent=agent, cut=_copy_clip, duration=duration, frames=_frames)
    )
    response = client.post(
        "/analyze",
        files={"video": ("game.mp4", b"game-bytes", "video/mp4")},
    )
    project = response.json()["project"]
    body = _settle(client, project)

    assert sorted(agent.seen) == [1.0, 8.0, 32.0]
    assert [(play["start"], play["family"]) for play in body["plays"]] == [
        (1.0, "special_teams"),
        (20.0, "scrimmage"),
    ]
    assert [(item["start"], item["end"]) for item in body["reviews"]] == [(8.0, 12.0), (32.0, 38.0)]
    assert client.get(f"/projects/{project}/source").content == b"game-bytes"

    kept = client.post(
        f"/api/projects/{project}/reviews",
        json={"start": 8, "end": 12, "decision": "scrimmage"},
    )
    assert kept.status_code == 200
    discarded = client.post(
        f"/api/projects/{project}/reviews",
        json={"start": 32, "end": 38, "decision": "junk"},
    )
    assert discarded.status_code == 200
    done = discarded.json()
    assert [(play["start"], play["family"]) for play in done["plays"]] == [
        (1.0, "special_teams"),
        (8.0, "scrimmage"),
        (20.0, "scrimmage"),
    ]
    assert done["reviews"] == []
    assert client.get(f"/projects/{project}/plays/3").content == b"game-bytes"


def test_model_failure_shows_on_the_project() -> None:
    client = TestClient(
        create_app(
            agent=_Raising(VideoAnalysisError("The model could not read that video.")),
            duration=_duration,
            frames=_frames,
        )
    )

    response = client.post(
        "/analyze",
        files={"video": ("clip.webm", b"fake-video", "video/webm")},
    )

    assert response.status_code == 200
    body = _settle(client, response.json()["project"])
    assert body["status"] == "failed"
    error = str(body["error"])
    assert "Stopped while reading 0:00–0:04 of 0:04." in error
    assert "VideoAnalysisError: The model could not read that video." in error
    assert "at film_analysis/" in error


def test_an_unexpected_failure_keeps_the_window_and_the_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = TestClient(
        create_app(
            agent=_Raising(RuntimeError("decoder blew up")),
            duration=_duration,
            frames=_frames,
        )
    )

    response = client.post(
        "/analyze",
        files={"video": ("clip.webm", b"fake-video", "video/webm")},
    )

    body = _settle(client, response.json()["project"])
    assert body["status"] == "failed"
    error = str(body["error"])
    assert "Stopped while reading 0:00–0:04 of 0:04." in error
    assert "RuntimeError: decoder blew up" in error
    assert "at film_analysis/web/app.py:label:" in error
    assert "film 1 failed while reading" in caplog.text
    assert "decoder blew up" in caplog.text


def test_play_page_shows_one_play_and_its_neighbors(tmp_path: Path) -> None:
    store = MemoryCatalog(tmp_path)
    project = store.create_project("Pike County vs Lamar", None)
    store.describe_film(
        project,
        name="Pike County vs Lamar",
        channel="K13 Media",
        duration_seconds=120,
    )
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.write_bytes(b"first-play")
    second.write_bytes(b"second-play")
    store.add_play(project, 1, 12, 18, first, "scrimmage")
    store.add_play(project, 2, 40, 46, second, "special_teams")
    client = TestClient(create_app(agent=_Agent(), store=store))

    page = client.get("/play/1-2")
    assert page.status_code == 200
    assert 'id="play-details"' in page.text

    detail = client.get("/api/plays/1-2")
    assert detail.status_code == 200
    assert detail.json() == {
        "uid": "1-2",
        "play_number": 2,
        "label": "Play 2: 0:40 - 0:46",
        "start": 40.0,
        "end": 46.0,
        "room_start": 28.0,
        "room_end": 58.0,
        "family": "special_teams",
        "play_type": None,
        "unit": None,
        "formation": None,
        "formation_key": None,
        "formation_plot": None,
        "steps": {
            "play": "detected",
            "side": "skip",
            "formation": "skip",
            "reviewed": "open",
        },
        "formation_layout": None,
        "formation_choices": _FORMATION_CHOICES,
        "reviewed": False,
        "discarded": False,
        "video": "/projects/1/plays/2",
        "project": 1,
        "name": "Pike County vs Lamar",
        "channel": "K13 Media",
        "position": 2,
        "count": 2,
        "film": "/projects/1",
        "previous": "1-1",
        "next": None,
        "tracker": [
            {"uid": "1-1", "position": 1, "reviewed": False},
            {"uid": "1-2", "position": 2, "reviewed": False},
        ],
    }
    assert client.get("/projects/1/plays/2").content == b"second-play"
    assert client.get("/api/plays/1-9").status_code == 404
    assert client.get("/api/plays/nope").status_code == 404
    assert client.get("/api/plays/01-1").status_code == 404


def test_a_play_can_be_relabeled_or_dropped(tmp_path: Path) -> None:
    store = MemoryCatalog(tmp_path)
    project = store.create_project("Pike County vs Lamar", None)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")
    store.add_play(project, 1, 12, 18, clip, "scrimmage")
    store.add_play(project, 2, 40, 46, clip, "special_teams")
    client = TestClient(create_app(agent=_Agent(), store=store))

    page = client.get("/play/1-1")
    assert "Two-point" in page.text
    assert "Field goal" in page.text
    assert "Not a play" in page.text

    punt = client.post("/api/plays/1-1", json={"play_type": "punt"})
    assert punt.status_code == 200
    assert punt.json()["play_type"] == "punt"
    assert punt.json()["family"] == "special_teams"
    assert punt.json()["reviewed"] is True
    assert punt.json()["tracker"][0]["reviewed"] is True

    marked = client.post("/api/plays/1-2", json={"reviewed": True})
    assert marked.status_code == 200
    assert marked.json()["reviewed"] is True
    assert marked.json()["play_type"] is None
    cleared = client.post("/api/plays/1-2", json={"reviewed": False})
    assert cleared.status_code == 200
    assert cleared.json()["reviewed"] is False

    coarse = client.post("/api/plays/1-2", json={"play_type": "scrimmage"})
    assert coarse.status_code == 200
    assert coarse.json()["play_type"] is None
    assert coarse.json()["family"] == "scrimmage"

    assert client.post("/api/plays/1-2", json={"play_type": "lateral"}).status_code == 422
    assert client.post("/api/plays/9-1", json={"play_type": "run"}).status_code == 404

    dropped = client.post("/api/plays/1-1", json={"play_type": "junk"})
    assert dropped.status_code == 200
    assert dropped.json() == {"removed": True, "project": 1, "film": "/projects/1"}
    assert client.get("/api/plays/1-1").status_code == 404
    remaining = client.get("/api/projects/1").json()["plays"]
    assert [play["uid"] for play in remaining] == ["1-2"]
    assert remaining[0]["family"] == "scrimmage"


def test_a_play_can_be_left_out_of_the_report(tmp_path: Path) -> None:
    store = MemoryCatalog(tmp_path)
    project = store.create_project("Friday night", None)
    store.set_progress(project, status="ready", duration_seconds=40, scanned_seconds=40, error=None)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")
    store.add_play(project, 1, 12, 20, clip, "scrimmage")
    store.add_play(project, 2, 24, 30, clip, "scrimmage")
    store.set_film_teams(project, FilmTeams("Home", "Away", "left"))
    client = TestClient(create_app(agent=_Agent(), store=store))

    page = client.get("/play/1-1")
    assert "Discard" in page.text
    assert "Not a play" in page.text

    left = client.post("/api/plays/1-2", json={"discarded": True})
    assert left.status_code == 200
    body = left.json()
    assert body["discarded"] is True
    assert body["play_type"] is None
    assert body["uid"] == "1-2"
    listed = client.get("/api/projects/1").json()["plays"]
    assert [play["uid"] for play in listed] == ["1-1", "1-2"]
    assert listed[1]["discarded"] is True
    assert listed[0]["discarded"] is False

    report = client.get("/api/projects/1/report")
    assert report.status_code == 200
    text = "\n".join(block["text"] for block in report.json()["blocks"])
    assert "Play 2" not in text

    back = client.post("/api/plays/1-2", json={"discarded": False})
    assert back.status_code == 200
    assert back.json()["discarded"] is False
    assert client.get("/api/plays/1-2").status_code == 200


def test_the_projects_list_does_not_probe_posters(tmp_path: Path) -> None:
    store = MemoryCatalog(tmp_path)
    project = store.create_project("Friday night", "https://youtu.be/abcdefghijk")
    store.set_progress(project, status="ready", duration_seconds=40, scanned_seconds=40, error=None)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")
    store.add_play(project, 1, 2.0, 8.0, clip, "scrimmage")
    probes = {"n": 0}
    original = store.poster_file

    def counted(project_id: int):
        probes["n"] += 1
        return original(project_id)

    store.poster_file = counted  # type: ignore[method-assign]
    client = TestClient(create_app(agent=_Agent(), store=store))

    listed = client.get("/api/projects").json()
    assert listed[0]["thumbnail"] == "/projects/1/poster"
    assert probes["n"] == 0
    poster = client.get("/projects/1/poster", follow_redirects=False)
    assert poster.status_code == 302
    assert "abcdefghijk" in poster.headers["location"]


def test_a_play_can_be_tagged_offense_defense_or_special_teams(tmp_path: Path) -> None:
    store = MemoryCatalog(tmp_path)
    project = store.create_project("Friday night", None)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")
    store.add_play(project, 1, 2.0, 8.0, clip, "scrimmage")
    client = TestClient(create_app(agent=_Agent(), store=store))

    offense = client.post("/api/plays/1-1", json={"unit": "offense"})
    assert offense.status_code == 200
    assert offense.json()["unit"] == "offense"
    assert offense.json()["family"] == "scrimmage"
    assert offense.json()["reviewed"] is True

    special = client.post("/api/plays/1-1", json={"play_type": "special_teams"})
    assert special.status_code == 200
    assert special.json()["family"] == "special_teams"
    assert special.json()["unit"] is None

    defense = client.post("/api/plays/1-1", json={"unit": "defense"})
    assert defense.status_code == 200
    assert defense.json()["unit"] == "defense"
    assert defense.json()["family"] == "scrimmage"


def test_a_play_can_be_trimmed_for_later_reads(tmp_path: Path) -> None:
    store = MemoryCatalog(tmp_path)
    project = store.create_project("Friday night", None)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source-film")
    store.save_source(project, source)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"old-clip")
    store.add_play(project, 1, 12.0, 20.0, clip, "scrimmage")
    store.set_play_formation(project, 1, "shotgun")
    cuts: list[tuple[float, float]] = []

    def cut(path: Path, start: float, end: float, dest: Path) -> None:
        assert path.read_bytes() == b"source-film"
        cuts.append((start, end))
        dest.write_bytes(b"trimmed-clip")

    client = TestClient(create_app(agent=_Agent(), store=store, cut=cut))
    page = client.get("/projects/1")
    assert "Start here" in page.text
    assert "Save trim" in page.text
    assert 'id="play-trim"' in page.text

    too_short = client.post("/api/plays/1-1/trim", json={"start": 12.0, "end": 12.2})
    assert too_short.status_code == 422

    outside = client.post("/api/plays/1-1/trim", json={"start": 10.0, "end": 40.0})
    assert outside.status_code == 400
    assert "beside this play" in outside.json()["detail"]

    trimmed = client.post("/api/plays/1-1/trim", json={"start": 13.5, "end": 18.0})
    assert trimmed.status_code == 200
    body = trimmed.json()
    assert body["start"] == 13.5
    assert body["end"] == 18.0
    assert body["reviewed"] is True
    assert body["formation"] in {None, ""}
    assert cuts == [(13.5, 18.0)]
    saved = store.get_project(project)
    assert saved is not None
    assert saved.plays[0].start_seconds == 13.5
    assert saved.plays[0].end_seconds == 18.0
    assert saved.plays[0].formation is None
    assert store.clip_file(project, 1).read_bytes() == b"trimmed-clip"


def test_a_play_can_be_lengthened_past_its_cut(tmp_path: Path) -> None:
    store = MemoryCatalog(tmp_path)
    project = store.create_project("Friday night", None)
    store.set_progress(project, status="ready", duration_seconds=40, scanned_seconds=40, error=None)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source-film")
    store.save_source(project, source)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"old-clip")
    store.add_play(project, 1, 12.0, 20.0, clip, "scrimmage")
    store.add_play(project, 2, 28.0, 34.0, clip, "scrimmage")
    cuts: list[tuple[float, float]] = []

    def cut(path: Path, start: float, end: float, dest: Path) -> None:
        cuts.append((start, end))
        dest.write_bytes(b"longer-clip")

    client = TestClient(create_app(agent=_Agent(), store=store, cut=cut))
    listed = client.get("/api/projects/1").json()["plays"][0]
    assert listed["room_start"] == 0.0
    assert listed["room_end"] == 28.0

    longer = client.post("/api/plays/1-1/trim", json={"start": 8.0, "end": 26.0})
    assert longer.status_code == 200
    assert longer.json()["start"] == 8.0
    assert longer.json()["end"] == 26.0
    assert cuts == [(8.0, 26.0)]

    into_next = client.post("/api/plays/1-1/trim", json={"start": 8.0, "end": 30.0})
    assert into_next.status_code == 400
    assert "beside this play" in into_next.json()["detail"]


def test_a_tool_failure_keeps_the_programs_own_error() -> None:
    message = tool_failure(
        "Could not read frames from that video.",
        b"ffmpeg version\nInvalid data found when processing input\n",
        1,
    )

    assert message == (
        "Could not read frames from that video.\n"
        "ffmpeg version\n"
        "Invalid data found when processing input"
    )
