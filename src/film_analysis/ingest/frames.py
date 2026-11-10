"""Pull timestamped JPEG frames out of a video file."""

import shutil
import subprocess
import tempfile
from pathlib import Path

from film_analysis.ingest.errors import VideoAnalysisError, tool_failure

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


def sample_frames(
    video_path: Path, seconds: float, count: int = 6, width: int = 768
) -> list[bytes]:
    """Return up to `count` JPEGs spread evenly over one play clip.

    The first frame skips a fraction of a second, which is often a broadcast
    dissolve. Short clips still give at least one frame.
    """

    _require_ffmpeg()
    frames: list[bytes] = []
    for moment in sample_times(seconds, count):
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{moment:.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-an",
                "-vf",
                f"scale={width}:-2",
                "-q:v",
                FRAME_QUALITY,
                "-f",
                "image2pipe",
                "-vcodec",
                "mjpeg",
                "-",
            ],
            check=False,
            capture_output=True,
        )
        if result.returncode == 0 and result.stdout.startswith(b"\xff\xd8\xff"):
            frames.append(result.stdout)
    return frames


def sample_times(seconds: float, count: int = 6) -> list[float]:
    """The moments `sample_frames` reads, so one of them can be read again larger."""

    length = max(0.1, seconds)
    lead = min(0.3, length / 4)
    span = max(0.0, length - lead - 0.2)
    count = max(1, count)
    return [lead + span * index / max(1, count - 1) for index in range(count)]


def full_frame(video_path: Path, moment: float) -> bytes | None:
    """One JPEG at the video's own size, for cropping a single player."""

    _require_ffmpeg()
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{max(0.0, moment):.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-an",
            "-q:v",
            "2",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "-",
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0 or not result.stdout.startswith(b"\xff\xd8\xff"):
        return None
    return result.stdout


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
