"""Lineup stills and original roster portraits for the two teams in a film."""

import base64
import binascii
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path

from film_analysis.agents.errors import VideoAnalysisError
from film_analysis.storage.catalog import PlayRecord

_FALLBACK = ("#1c3f73", "#f2efe6")
_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")
_KLEIN = "@cf/black-forest-labs/flux-2-klein-4b"
_SCHNELL = "@cf/black-forest-labs/flux-1-schnell"


def lineup_moment(plays: Sequence[PlayRecord]) -> float:
    """A second inside the first scrimmage, where both teams are usually lined up."""

    chosen = next((play for play in plays if play.family == "scrimmage"), None)
    if chosen is None:
        chosen = plays[0]
    length = max(0.0, chosen.end_seconds - chosen.start_seconds)
    return chosen.start_seconds + min(0.6, length / 2)


def split_lineup(video: Path, at: float) -> tuple[bytes, bytes]:
    """Return JPEG stills of the left and right halves of one frame."""

    if shutil.which("ffmpeg") is None:
        raise VideoAnalysisError("Could not read a lineup from that film.")
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        frame = root / "frame.jpg"
        _run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{max(0.0, at):.3f}",
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-vf",
                "scale=960:-2",
                "-q:v",
                "4",
                str(frame),
            ]
        )
        left = root / "left.jpg"
        right = root / "right.jpg"
        _crop(frame, "crop=iw/2:ih:0:0", left)
        _crop(frame, "crop=iw/2:ih:iw/2:0", right)
        return left.read_bytes(), right.read_bytes()


def dominant_colors(still: bytes) -> tuple[str, str]:
    """Jersey and pants colors from a lineup still, skipping grass and sky."""

    try:
        pixels = _sample_pixels(still)
    except (OSError, VideoAnalysisError):
        return _FALLBACK
    return colors_from_pixels(pixels)


def colors_from_pixels(pixels: Sequence[tuple[int, int, int]]) -> tuple[str, str]:
    """Pick the two strongest uniform colors in already-sampled pixels."""

    counts: dict[tuple[int, int, int], int] = {}
    for red, green, blue in pixels:
        if _skip_color(red, green, blue):
            continue
        key = (red // 32 * 32, green // 32 * 32, blue // 32 * 32)
        counts[key] = counts.get(key, 0) + 1
    ranked = sorted(counts, key=lambda color: counts[color], reverse=True)
    if not ranked:
        return _FALLBACK
    jersey = ranked[0]
    pants = next((color for color in ranked[1:] if _apart(jersey, color)), None)
    return _hex(jersey), _FALLBACK[1] if pants is None else _hex(pants)


def make_portrait(still: bytes, team_name: str) -> tuple[bytes, str]:
    """A standing number-00 player in this team's uniform, or a drawing if that fails."""

    jersey, pants = dominant_colors(still)
    image = _flux(still, team_name, jersey, pants)
    if image is not None:
        return image, ".jpg"
    return roster_svg(jersey, pants), ".svg"


def player_prompt(team_name: str, jersey: str, pants: str) -> str:
    """Ask for one full-body player whose uniform matches the lineup photo."""

    return (
        "The reference photo shows this team's football uniform on the field. "
        "Match its jersey, pants, and helmet colors. Do not copy the photo. "
        "Create one original 3D American football player standing full length, "
        "facing the camera, both feet on the ground, arms relaxed, "
        "sports video-game create-a-player render, helmet on, shoulder pads, "
        f"jersey color {jersey}, pants color {pants}, helmet matching the jersey, "
        "large jersey number 00 and nothing else written on the uniform, cleats, "
        "plain dark studio background, entire body from helmet to cleats in frame, "
        "no field, no crowd, no logos, no trademarks, no real person, "
        f"uniform for the team named {team_name}."
    )


def roster_svg(jersey: str, pants: str) -> bytes:
    """An original roster drawing: helmet, jersey number 00, and pants."""

    shirt = _safe_color(jersey, _FALLBACK[0])
    legs = _safe_color(pants, _FALLBACK[1])
    lines = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 240 320">',
        '<rect width="240" height="320" fill="#14161a"/>',
        f'<path d="M86 36c0-22 68-22 68 0v22H86z" fill="{shirt}"/>',
        '<ellipse cx="120" cy="72" rx="22" ry="18" fill="#c4a574"/>',
        f'<path d="M70 108h100l18 36v70H52v-70z" fill="{shirt}"/>',
        '<text x="120" y="176" text-anchor="middle" font-size="46"',
        ' font-family="sans-serif" font-weight="700" fill="#f7f4ee">00</text>',
        f'<path d="M68 214h104l10 86H58z" fill="{legs}"/>',
        "</svg>",
    )
    return "\n".join(lines).encode()


def _crop(frame: Path, crop: str, dest: Path) -> None:
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(frame),
            "-vf",
            crop,
            "-q:v",
            "4",
            str(dest),
        ]
    )


def _run(args: list[str]) -> None:
    result = subprocess.run(args, capture_output=True, check=False)
    if result.returncode != 0:
        raise VideoAnalysisError("Could not read a lineup from that film.")


def _sample_pixels(still: bytes) -> list[tuple[int, int, int]]:
    if shutil.which("ffmpeg") is None:
        raise VideoAnalysisError("Could not read colors from that still.")
    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder) / "still.jpg"
        source.write_bytes(still)
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-vf",
                "scale=16:16",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ],
            capture_output=True,
            check=False,
        )
    raw = result.stdout
    if result.returncode != 0 or len(raw) < 3:
        raise VideoAnalysisError("Could not read colors from that still.")
    usable = len(raw) - (len(raw) % 3)
    return [(raw[index], raw[index + 1], raw[index + 2]) for index in range(0, usable, 3)]


def _skip_color(red: int, green: int, blue: int) -> bool:
    if max(red, green, blue) < 40 or min(red, green, blue) > 210:
        return True
    if max(red, green, blue) - min(red, green, blue) < 18:
        return True
    return green > red + 12 and green > blue + 8 and green > 70


def _apart(left: tuple[int, int, int], right: tuple[int, int, int]) -> bool:
    return sum(abs(a - b) for a, b in zip(left, right, strict=True)) >= 48


def _hex(color: tuple[int, int, int]) -> str:
    return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"


def _safe_color(value: str, fallback: str) -> str:
    return value if _COLOR.fullmatch(value) else fallback


def _flux(still: bytes, team_name: str, jersey: str, pants: str) -> bytes | None:
    account = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
    token = os.environ.get("CLOUDFLARE_API_TOKEN", "")
    if not account or not token:
        return None
    prompt = player_prompt(team_name, jersey, pants)
    reference = _reference_jpeg(still)
    if reference is not None:
        image = _run_reference(account, token, prompt, reference)
        if image is not None:
            return image
    return _run_prompt(account, token, prompt)


def _run_reference(account: str, token: str, prompt: str, reference: bytes) -> bytes | None:
    body, content_type = _form(
        [("prompt", prompt), ("width", "768"), ("height", "1280")],
        reference,
    )
    return _post_image(account, token, _KLEIN, body, content_type)


def _run_prompt(account: str, token: str, prompt: str) -> bytes | None:
    body = json.dumps({"prompt": prompt, "width": 768, "height": 1280}).encode()
    return _post_image(account, token, _SCHNELL, body, "application/json")


def _post_image(
    account: str,
    token: str,
    model: str,
    body: bytes,
    content_type: str,
) -> bytes | None:
    url = f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{model}"
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": content_type},
    )
    try:
        with urllib.request.urlopen(request, timeout=70) as response:
            payload = json.loads(response.read().decode())
    except (OSError, urllib.error.URLError, json.JSONDecodeError, UnicodeError, TimeoutError):
        return None
    return _image_bytes(payload)


def _form(fields: list[tuple[str, str]], reference: bytes) -> tuple[bytes, str]:
    boundary = "film-player-boundary"
    chunks: list[bytes] = []
    for name, value in fields:
        chunks.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        )
    chunks.append(
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="input_image_0"; '
            'filename="lineup.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'
        ).encode()
    )
    chunks.append(reference)
    chunks.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _reference_jpeg(still: bytes) -> bytes | None:
    """Shrink the lineup still so the image model can use it as a uniform reference."""

    if shutil.which("ffmpeg") is None or not still:
        return None
    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder) / "still.jpg"
        dest = Path(folder) / "reference.jpg"
        source.write_bytes(still)
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-vf",
                "scale=480:480:force_original_aspect_ratio=decrease",
                "-q:v",
                "4",
                str(dest),
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0 or not dest.is_file():
            return None
        data = dest.read_bytes()
    if not data.startswith(b"\xff\xd8\xff") or len(data) > 1_500_000:
        return None
    return data


def _image_bytes(payload: object) -> bytes | None:
    if not isinstance(payload, dict):
        return None
    result = payload.get("result", payload)
    image = result.get("image") if isinstance(result, dict) else None
    if not isinstance(image, str) or not image:
        return None
    if image.startswith("data:"):
        image = image.split(",", 1)[-1]
    try:
        decoded = base64.b64decode(image)
    except (binascii.Error, ValueError):
        return None
    if decoded.startswith(b"\xff\xd8\xff"):
        return decoded
    if decoded.startswith(b"\x89PNG\r\n\x1a\n"):
        return _jpeg_from_png(decoded)
    return None


def _jpeg_from_png(png: bytes) -> bytes | None:
    if shutil.which("ffmpeg") is None:
        return None
    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder) / "player.png"
        dest = Path(folder) / "player.jpg"
        source.write_bytes(png)
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-q:v",
                "3",
                str(dest),
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0 or not dest.is_file():
            return None
        data = dest.read_bytes()
    return data if data.startswith(b"\xff\xd8\xff") else None
