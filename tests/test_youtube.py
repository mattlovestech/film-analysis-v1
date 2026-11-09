import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from film_analysis.agents.youtube import (
    _PART_FLOOR,
    _PARTS,
    FilmInfo,
    _hook,
    _part_bounds,
    _save_url,
    _select_stream,
    canonical_youtube_url,
    detach_progress,
    film_info,
)


def test_film_info_reads_the_title_channel_and_length() -> None:
    info = film_info(
        {
            "title": "Carrollton vs. Buford",
            "author": "Prep Highlights",
            "lengthSeconds": 11119,
        }
    )

    assert info == FilmInfo("Carrollton vs. Buford", "Prep Highlights", 11119)
    assert film_info({"title": "  "}) is None


def test_watch_short_and_embed_links_canonicalize() -> None:
    expected = "https://www.youtube.com/watch?v=abcdefghijk"

    assert canonical_youtube_url("https://www.youtube.com/watch?v=abcdefghijk&t=30s") == expected
    assert canonical_youtube_url("https://youtu.be/abcdefghijk") == expected
    assert canonical_youtube_url("https://www.youtube.com/shorts/abcdefghijk") == expected
    assert canonical_youtube_url("https://www.youtube.com/embed/abcdefghijk") == expected
    assert canonical_youtube_url("https://m.youtube.com/watch?v=abcdefghijk") == expected


def test_stream_choice_prefers_h264_at_or_below_360p() -> None:
    chosen = _select_stream(
        {
            "adaptiveFormats": [
                {
                    "type": 'video/mp4; codecs="avc1.4d401e"',
                    "qualityLabel": "1080p",
                    "url": "https://cdn/1080",
                },
                {"type": 'video/mp4; codecs="av01"', "qualityLabel": "480p", "url": "https://cdn/av1"},
                {
                    "type": 'video/mp4; codecs="avc1.4d401e"',
                    "qualityLabel": "240p",
                    "url": "https://cdn/240",
                },
                {
                    "type": 'video/mp4; codecs="av01"',
                    "qualityLabel": "360p",
                    "url": "https://cdn/av1-360",
                },
                {
                    "type": 'video/mp4; codecs="avc1.4d401e"',
                    "qualityLabel": "360p",
                    "url": "https://cdn/360",
                },
                {"type": "audio/mp4", "qualityLabel": "360p", "url": "https://cdn/audio"},
            ]
        }
    )

    assert chosen == "https://cdn/360"


def test_a_stream_reports_each_chunk(tmp_path: Path) -> None:
    body = b"x" * 2500

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    seen: list[tuple[int, int | None]] = []
    port = server.server_address[1]
    dest = tmp_path / "film.mp4"

    def report(done: int, total: int | None) -> None:
        seen.append((done, total))

    try:
        _save_url(f"http://127.0.0.1:{port}/film.mp4", dest, report)
    finally:
        server.shutdown()

    assert dest.read_bytes() == body
    assert seen == [(2500, 2500)]


def test_a_large_stream_is_fetched_in_parts(tmp_path: Path) -> None:
    body = bytes(index % 251 for index in range(_PART_FLOOR + 1024))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            length = len(body)
            header = self.headers.get("Range")
            if header and header.startswith("bytes="):
                start_text, end_text = header.removeprefix("bytes=").split("-", 1)
                start = int(start_text) if start_text else 0
                end = int(end_text) if end_text else length - 1
                end = min(end, length - 1)
                chunk = body[start : end + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{length}")
                self.send_header("Content-Length", str(len(chunk)))
                self.end_headers()
                self.wfile.write(chunk)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(length))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    dest = tmp_path / "film.mp4"
    port = server.server_address[1]
    try:
        _save_url(f"http://127.0.0.1:{port}/film.mp4", dest, lambda done, total: None)
    finally:
        server.shutdown()

    assert dest.read_bytes() == body
    assert len(_part_bounds(len(body))) == _PARTS


def test_progress_does_not_wait_on_a_slow_save() -> None:
    started = threading.Event()
    release = threading.Event()

    def send(done: int, total: int | None) -> None:
        del done, total
        started.set()
        assert release.wait(timeout=2)

    report = detach_progress(send, interval=0)
    began = time.monotonic()
    report(10, 100)

    assert started.wait(timeout=1)
    assert time.monotonic() - began < 0.5
    release.set()
    seen: list[tuple[int, int | None]] = []
    _hook(lambda done, total: seen.append((done, total)))(
        {"status": "downloading", "downloaded_bytes": 10, "total_bytes_estimate": 40}
    )

    assert seen == [(10, 40)]


@pytest.mark.parametrize(
    "value",
    [
        "",
        "https://vimeo.com/123456789",
        "https://www.youtube.com/watch?v=short",
        "https://www.youtube.com/@channel",
        "javascript:alert(1)",
    ],
)
def test_rejects_links_that_are_not_a_video(value: str) -> None:
    with pytest.raises(ValueError, match="YouTube"):
        canonical_youtube_url(value)
