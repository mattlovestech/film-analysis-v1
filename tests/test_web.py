import subprocess
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from film_analysis.agents.errors import MissingApiKeyError, VideoAnalysisError
from film_analysis.agents.segments import FilmSpan, PlaySegment
from film_analysis.agents.youtube import FilmInfo
from film_analysis.web.app import create_app


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


def test_page_offers_a_video_upload() -> None:
    client = TestClient(create_app(agent=_Agent()))
    response = client.get("/")

    assert response.status_code == 200
    assert 'type="file"' in response.text
    assert 'name="youtube"' in response.text
    assert "Drop a video here" in response.text
    assert 'id="new-analysis" href="/"' in response.text
    assert "My analyses" in response.text
    assert 'href="/activity"' in response.text
    library = client.get("/analyses")
    assert library.status_code == 200
    assert "My analyses" in library.text
    assert client.get("/activity").status_code == 200
    assert client.get("/api/activity").json() == {"workers": 4, "jobs": []}
    assert "Couldn't reach the server" in response.text
    assert "Downloading the film" in response.text
    assert 'id="download-track"' in response.text
    assert 'id="download-pct"' in response.text
    assert "Retry download" in response.text
    assert 'createElement("video")' in response.text


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
            "play_number": 1,
            "label": "Play 1: 0:01 - 0:02",
            "start": 1.0,
            "end": 2.0,
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
            "thumbnail": None,
            "status": "ready",
            "downloaded": 0,
            "total_bytes": None,
            "youtube": False,
            "running": False,
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
    assert client.get("/api/activity").json()["jobs"] == []


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
    assert response.json()["detail"] == "Use a file or a YouTube link."
    assert agent.urls == []
    assert agent.paths == []


def test_missing_source_is_rejected() -> None:
    response = TestClient(create_app(agent=_Agent())).post("/analyze", data={"youtube": "  "})

    assert response.status_code == 400
    assert response.json()["detail"] == "Upload a video or paste a YouTube link."


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
    assert body["error"] == "Set ANTHROPIC_API_KEY to analyze film."


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
    assert "could not read" in str(body["error"])
