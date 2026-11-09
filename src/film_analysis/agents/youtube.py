"""Public YouTube links the play agent can watch."""

import http.client
import json
import math
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_HOSTS = {"youtube.com", "m.youtube.com", "music.youtube.com", "youtube-nocookie.com"}
_PATH_KINDS = {"embed", "shorts", "live", "v"}
_UA = "Mozilla/5.0"
_INSTANCES = "https://api.invidious.io/instances.json?sort_by=health"
_KNOWN_INSTANCE = "https://yt.chocolatemoo53.com"
_MAX_HEIGHT = 720
_PARTS = 16
_PART_FLOOR = 4 * 1024 * 1024
_BUFFER = 1024 * 1024
_Progress = Callable[[int, int | None], None]


def canonical_youtube_url(value: str) -> str:
    """Return `https://www.youtube.com/watch?v=…` for a public YouTube link."""

    text = value.strip()
    if not text:
        raise ValueError("Paste a YouTube link.")
    if "://" not in text:
        text = f"https://{text}"

    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Paste a YouTube link.")

    host = (parsed.hostname or "").lower().removeprefix("www.")
    video_id = _video_id(host, parsed.path, parsed.query)
    if video_id is None:
        raise ValueError("Paste a YouTube link.")
    return f"https://www.youtube.com/watch?v={video_id}"


def _video_id(host: str, path: str, query: str) -> str | None:
    if host == "youtu.be":
        candidate = path.strip("/").split("/", maxsplit=1)[0]
    elif host in _HOSTS:
        parts = [part for part in path.split("/") if part]
        if path == "/watch" or (parts and parts[0] == "watch"):
            candidate = parse_qs(query).get("v", [""])[0]
        elif len(parts) >= 2 and parts[0] in _PATH_KINDS:
            candidate = parts[1]
        else:
            return None
    else:
        return None

    if _VIDEO_ID.fullmatch(candidate) is None:
        return None
    return candidate


@dataclass(frozen=True)
class FilmInfo:
    """Title, channel, and length looked up before the file is downloaded."""

    title: str
    channel: str
    duration_seconds: float | None


def film_info(payload: object) -> FilmInfo | None:
    """Read a title and length out of an Invidious video payload."""

    if not isinstance(payload, dict):
        return None
    title = str(payload.get("title") or "").strip()
    channel = str(payload.get("author") or payload.get("uploader") or "").strip()
    length = payload.get("lengthSeconds", payload.get("duration"))
    duration: float | None
    try:
        duration = float(length) if length is not None and length != "" else None
    except (TypeError, ValueError):
        duration = None
    if duration is not None and duration <= 0:
        duration = None
    if not title and duration is None:
        return None
    return FilmInfo(title=title, channel=channel, duration_seconds=duration)


def youtube_details(url: str) -> FilmInfo | None:
    """Look up a public film's title, channel, and length without downloading it."""

    try:
        video_id = parse_qs(urlparse(canonical_youtube_url(url)).query)["v"][0]
    except (ValueError, KeyError, IndexError):
        return None
    for base in _invidious_bases():
        try:
            parsed = film_info(_read_json(f"{base}/api/v1/videos/{video_id}"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError, TimeoutError):
            continue
        if parsed is not None and parsed.title:
            return parsed
    return None


def thumbnail_url(url: str | None) -> str | None:
    """Return the public preview image for a YouTube link."""

    if not url:
        return None
    video_id = parse_qs(urlparse(url).query).get("v", [""])[0]
    if _VIDEO_ID.fullmatch(video_id) is None:
        return None
    return f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg"


def download_youtube(
    url: str,
    directory: Path,
    progress: _Progress | None = None,
) -> Path:
    """Save a public YouTube video into `directory` and return the file."""

    from film_analysis.agents.errors import VideoAnalysisError

    try:
        import yt_dlp
    except ImportError as error:
        raise VideoAnalysisError("yt-dlp is required to fetch a YouTube video.") from error

    options = {
        "format": "bv*[height<=720][ext=mp4]/bv*[height<=720]/bv*[height<=480][ext=mp4]/bv*",
        "outtmpl": str(directory / "film.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "socket_timeout": 30,
        "retries": 2,
        "fragment_retries": 3,
        "concurrent_fragment_downloads": _PARTS,
        "buffersize": _BUFFER,
        "progress_hooks": [_hook(progress)],
    }
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([url])
    except Exception as error:
        text = str(error)
        if "not a bot" not in text.lower():
            raise VideoAnalysisError(_download_reason(error)) from error
        for path in directory.glob("film.*"):
            path.unlink(missing_ok=True)
        _download_via_invidious(url, directory, progress)

    files = [path for path in directory.glob("film.*") if path.is_file()]
    if not files:
        raise VideoAnalysisError("Could not fetch that YouTube video.")
    merged = directory / "film.mp4"
    if merged.is_file():
        return merged
    return max(files, key=lambda path: path.stat().st_size)


def detach_progress(send: _Progress, interval: float = 0.4) -> _Progress:
    """Report progress without waiting on `send`.

    Saving the byte count can take longer than reading the next chunk. The
    download keeps going, and a newer count replaces one that has not been saved.
    """

    last_sent = 0.0
    lock = threading.Lock()
    latest: list[tuple[int, int | None] | None] = [None]
    running = False

    def report(done: int, total: int | None) -> None:
        nonlocal running
        now = time.monotonic()
        finished = total is not None and total > 0 and done >= total
        with lock:
            if latest[0] is None and not finished and now - last_sent < interval:
                return
            latest[0] = (done, total)
            if running:
                return
            running = True

        def pump() -> None:
            nonlocal running, last_sent
            while True:
                with lock:
                    item = latest[0]
                    latest[0] = None
                if item is None:
                    with lock:
                        if latest[0] is None:
                            running = False
                            return
                    continue
                send(*item)
                with lock:
                    last_sent = time.monotonic()

        threading.Thread(target=pump, daemon=True).start()

    return report


def _download_reason(error: Exception) -> str:
    lines = [line.strip() for line in str(error).splitlines() if line.strip()]
    useful = [line for line in lines if not line.lower().startswith("warning:")]
    chosen = (useful or lines)[-8:]
    name = type(error).__name__
    if not chosen:
        return f"{name}: Could not fetch that YouTube video."
    return f"{name}: " + "\n".join(chosen)


def _hook(progress: _Progress | None) -> Callable[[dict[str, object]], None]:
    def report(status: dict[str, object]) -> None:
        if progress is None or status.get("status") not in {"downloading", "finished"}:
            return
        done = _byte_count(status.get("downloaded_bytes"))
        if done is None:
            return
        total = _byte_count(status.get("total_bytes"))
        if total is None:
            total = _byte_count(status.get("total_bytes_estimate"))
        progress(done, total)

    return report


def _byte_count(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _download_via_invidious(
    url: str,
    directory: Path,
    progress: _Progress | None = None,
) -> None:
    from film_analysis.agents.errors import VideoAnalysisError

    video_id = parse_qs(urlparse(canonical_youtube_url(url)).query)["v"][0]
    for base in _invidious_bases():
        try:
            info = _read_json(f"{base}/api/v1/videos/{video_id}")
        except (OSError, urllib.error.URLError, json.JSONDecodeError, TimeoutError):
            continue
        stream = _select_stream(info)
        if stream is None:
            continue
        dest = directory / "film.mp4"
        try:
            _save_url(stream, dest, progress)
        except (OSError, urllib.error.URLError, TimeoutError):
            dest.unlink(missing_ok=True)
            continue
        if dest.stat().st_size > 100_000:
            return
        dest.unlink(missing_ok=True)
    raise VideoAnalysisError("Could not fetch that YouTube video.")


def _invidious_bases() -> Iterator[str]:
    """Yield the known server first, and look up others only if that one fails."""

    seen = {_KNOWN_INSTANCE}
    yield _KNOWN_INSTANCE
    try:
        payload = _read_json(_INSTANCES)
    except (OSError, urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return
    if not isinstance(payload, list):
        return
    for item in payload:
        if not isinstance(item, list) or len(item) < 2 or not isinstance(item[1], dict):
            continue
        meta = item[1]
        if meta.get("type") != "https":
            continue
        uri = str(meta.get("uri") or "").rstrip("/")
        if uri.startswith("https://") and uri not in seen:
            seen.add(uri)
            yield uri
        if len(seen) >= 5:
            return


def _select_stream(info: object) -> str | None:
    if not isinstance(info, dict):
        return None
    formats = list(info.get("adaptiveFormats") or [])
    ranked: list[tuple[int, int, str]] = []
    for item in formats:
        if not isinstance(item, dict):
            continue
        mime = str(item.get("type") or "")
        stream = item.get("url")
        height = _height(item.get("qualityLabel"))
        if not isinstance(stream, str) or "video/mp4" not in mime or height is None:
            continue
        if height > _MAX_HEIGHT:
            continue
        avc = 1 if "avc1" in mime or item.get("encoding") == "h264" else 0
        ranked.append((avc, height, stream))
    if not ranked:
        return None
    ranked.sort()
    return ranked[-1][2]


def _height(label: object) -> int | None:
    if not isinstance(label, str) or not label.endswith("p"):
        return None
    try:
        return int(label[:-1])
    except ValueError:
        return None


def _read_json(url: str) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response)


def _save_url(url: str, dest: Path, progress: _Progress | None = None) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": _UA, "Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        if getattr(response, "status", 200) != 206:
            _copy_body(response, dest, progress)
            return
        total = _range_total(response.headers.get("Content-Range"))
        response.read()
    if total is None or total < _PART_FLOOR:
        _save_linear(url, dest, progress)
        return
    _save_parallel(url, dest, total, progress)


def _save_linear(url: str, dest: Path, progress: _Progress | None) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(request, timeout=60) as response:
        _copy_body(response, dest, progress)


def _save_parallel(url: str, dest: Path, total: int, progress: _Progress | None) -> None:
    bounds = _part_bounds(total)
    dest.write_bytes(b"")
    done = [0] * len(bounds)
    reported = 0
    lock = threading.Lock()

    def grab(index: int, start: int, end: int) -> None:
        nonlocal reported
        request = urllib.request.Request(
            url,
            headers={"User-Agent": _UA, "Range": f"bytes={start}-{end}"},
        )
        with urllib.request.urlopen(request, timeout=60) as response, dest.open("r+b") as handle:
            if getattr(response, "status", 200) != 206:
                raise OSError("The video server did not accept a partial download.")
            handle.seek(start)
            while chunk := response.read(_BUFFER):
                handle.write(chunk)
                with lock:
                    done[index] += len(chunk)
                    got = sum(done)
                    fresh = got >= reported
                    if fresh:
                        reported = got
                if fresh and progress is not None:
                    progress(got, total)

    with ThreadPoolExecutor(max_workers=len(bounds)) as pool:
        jobs = [pool.submit(grab, index, start, end) for index, (start, end) in enumerate(bounds)]
        for job in jobs:
            job.result()


def _part_bounds(total: int, parts: int = _PARTS) -> list[tuple[int, int]]:
    count = max(1, min(parts, total))
    span = math.ceil(total / count)
    bounds: list[tuple[int, int]] = []
    start = 0
    while start < total:
        end = min(total - 1, start + span - 1)
        bounds.append((start, end))
        start = end + 1
    return bounds


def _range_total(header: str | None) -> int | None:
    if not header or "/" not in header:
        return None
    total = header.rsplit("/", 1)[-1]
    if not total.isdigit():
        return None
    size = int(total)
    return size if size > 0 else None


def _copy_body(
    response: http.client.HTTPResponse,
    dest: Path,
    progress: _Progress | None,
) -> None:
    total = _byte_count(response.headers.get("Content-Length"))
    done = 0
    with dest.open("wb") as handle:
        while chunk := response.read(_BUFFER):
            handle.write(chunk)
            done += len(chunk)
            if progress is not None:
                progress(done, total)
