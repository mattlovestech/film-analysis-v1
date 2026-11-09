"""Pull timestamped JPEG frames out of a video file."""

import shutil
import subprocess
import tempfile
from pathlib import Path

from film_analysis.agents.errors import VideoAnalysisError, tool_failure

FRAME_INTERVAL = 2.0
# One frame every two seconds, at 768px wide. That is enough to see a lineup,
# a snap, and a dead ball, and it is much less for the model to read than 720p.
FRAME_FILTER = "fps=1/2,scale=768:-2"
FRAME_QUALITY = "8"


def video_duration(video_path: Path) -> float:
    """Return the length of a video in seconds."""

    return _duration(video_path)


def extract_window(video_path: Path, start: float, seconds: float) -> list[tuple[float, bytes]]:
    """Return one JPEG every two seconds for a stretch of film, with times from the file start."""

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
                str(video_path),
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
            raise VideoAnalysisError(
                tool_failure(
                    "Could not read frames from that video.",
                    result.stderr,
                    result.returncode,
                )
            )
        return [
            (start + index * FRAME_INTERVAL, frame.read_bytes())
            for index, frame in enumerate(sorted(Path(tmp).glob("frame_*.jpg")))
        ]


def extract_frames(video_path: Path) -> list[tuple[float, bytes]]:
    """Return one JPEG every two seconds, labeled with its offset from the start."""

    _require_ffmpeg()
    with tempfile.TemporaryDirectory() as tmp:
        pattern = str(Path(tmp) / "frame_%04d.jpg")
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video_path),
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
            raise VideoAnalysisError(
                tool_failure(
                    "Could not read frames from that video.",
                    result.stderr,
                    result.returncode,
                )
            )

        frames = [
            (index * FRAME_INTERVAL, frame.read_bytes())
            for index, frame in enumerate(sorted(Path(tmp).glob("frame_*.jpg")))
        ]
    if not frames:
        raise VideoAnalysisError(
            "Could not read frames from that video. ffmpeg produced no frames."
        )
    return frames


def _duration(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(video_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise VideoAnalysisError(
            tool_failure("Could not read that video.", result.stderr, result.returncode)
        )
    try:
        return float(result.stdout.strip())
    except ValueError as error:
        raise VideoAnalysisError("Could not read the length of that video.") from error


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    raise VideoAnalysisError("ffmpeg is required to read film.")
