"""Read one stretch of a film over HTTPS. The full file stays in R2."""

import shutil
import subprocess
import tempfile
from pathlib import Path

from film_analysis.ingest.errors import VideoAnalysisError
from film_analysis.ingest.frames import FRAME_FILTER, FRAME_INTERVAL, FRAME_QUALITY


def extract_window_url(url: str, start: float, seconds: float) -> list[tuple[float, bytes]]:
    """Return one JPEG every two seconds for a stretch, with times from the file start."""

    _require_ffmpeg()
    with tempfile.TemporaryDirectory() as tmp:
        pattern = str(Path(tmp) / "frame_%04d.jpg")
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{max(0.0, start):.3f}",
                "-i",
                url,
                "-t",
                f"{max(0.1, seconds):.3f}",
                "-an",
                "-vf",
                FRAME_FILTER,
                "-q:v",
                FRAME_QUALITY,
                pattern,
            ],
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            raise VideoAnalysisError("Could not read frames from that video.")
        return [
            (start + index * FRAME_INTERVAL, frame.read_bytes())
            for index, frame in enumerate(sorted(Path(tmp).glob("frame_*.jpg")))
        ]


def cut_url(url: str, start: float, end: float, dest: Path) -> None:
    """Write one play, from `start` to `end`, reading the film over HTTPS."""

    if shutil.which("ffmpeg") is None:
        raise VideoAnalysisError("ffmpeg is required to split the film into plays.")
    duration = end - start
    if duration <= 0:
        raise VideoAnalysisError("A play must end after it starts.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            url,
            "-t",
            f"{duration:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(dest),
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0 or not dest.is_file():
        raise VideoAnalysisError("Could not split that play out of the film.")


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    raise VideoAnalysisError("ffmpeg is required to read film.")
