"""Turn a film file into a seekable MP4 the browser can play."""

import json
import shutil
import subprocess
from pathlib import Path

from film_analysis.agents.errors import VideoAnalysisError


def prepare_playback(source: Path) -> Path:
    """Copy or transcode `source` to an H.264 MP4 with the index at the front."""

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise VideoAnalysisError("ffmpeg is required to play film in the browser.")

    dest = source.with_name(f"{source.name}.play.mp4")
    video_codec, audio_codec = _codecs(source)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source)]
    if video_codec == "h264":
        command += ["-c:v", "copy"]
    else:
        command += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p"]
    if audio_codec == "aac":
        command += ["-c:a", "copy"]
    elif audio_codec:
        command += ["-c:a", "aac", "-b:a", "128k"]
    else:
        command.append("-an")
    command += ["-movflags", "+faststart", str(dest)]

    result = subprocess.run(command, check=False, capture_output=True)
    if result.returncode != 0 or not dest.is_file():
        raise VideoAnalysisError("Could not prepare that video for playback.")
    return dest


def cut_clip(source: Path, start: float, end: float, dest: Path) -> None:
    """Write one play, from `start` to `end`, as its own MP4."""

    if shutil.which("ffmpeg") is None:
        raise VideoAnalysisError("ffmpeg is required to split the film into plays.")
    duration = end - start
    if duration <= 0:
        raise VideoAnalysisError("A play must end after it starts.")
    _video, audio = _codecs(source)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(source),
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
    ]
    if audio:
        command += ["-c:a", "aac", "-b:a", "128k"]
    else:
        command.append("-an")
    command += ["-movflags", "+faststart", str(dest)]
    result = subprocess.run(command, check=False, capture_output=True)
    if result.returncode != 0 or not dest.is_file():
        raise VideoAnalysisError("Could not split that play out of the film.")


def _codecs(source: Path) -> tuple[str, str]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-print_format",
            "json",
            str(source),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    video = ""
    audio = ""
    if result.returncode != 0:
        return video, audio
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return video, audio
    for stream in payload.get("streams", []):
        kind = stream.get("codec_type")
        name = stream.get("codec_name") or ""
        if kind == "video" and not video:
            video = name
        elif kind == "audio" and not audio:
            audio = name
    return video, audio
