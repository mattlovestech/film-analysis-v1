"""Agent that watches film frames and marks each play."""

import base64
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, runtime_checkable

from film_analysis.agents.errors import MissingApiKeyError, VideoAnalysisError
from film_analysis.agents.frames import extract_frames
from film_analysis.agents.segments import (
    FilmSpan,
    PlaySegment,
    parse_film_spans,
    parse_play_segments,
)
from film_analysis.agents.youtube import canonical_youtube_url, download_youtube

PROMPT = """
You chart American football film.
Each image is labeled with its time in seconds from the start of the video.

A play starts at the snap, kickoff, or punt and ends when the ball is dead.

Return JSON with this shape and nothing else:
{"plays": [{"play": 1, "start_seconds": 0.0, "end_seconds": 0.0}]}

Use the frame times. List plays in order. Do not overlap them. Include every snap and kick.
""".strip()

SPAN_PROMPT = """
You chart American football film.
Each image is labeled with its time in seconds from the start of the video.
This window runs from {start:.1f}s to {end:.1f}s. Label every moment in that range.

A play starts at the snap, kickoff, or punt and ends when the ball is dead.
Anything else is not a play. Use kind huddle, dead, replay, crowd, or graphic.

Return JSON with this shape and nothing else:
{{"spans": [{{"kind": "play", "start_seconds": 0.0, "end_seconds": 0.0}}]}}

Use the frame times. Cover the window from its first time to its last. Do not overlap spans.
""".strip()

DEFAULT_MODEL = "claude-sonnet-5-5"
_WINDOW = 40

FrameSampler = Callable[[Path], list[tuple[float, bytes]]]
Downloader = Callable[[str, Path], Path]


@runtime_checkable
class PlayBoundaryAgent(Protocol):
    """Watch one film and return its plays."""

    name: str

    def label_window(self, samples: list[tuple[float, bytes]]) -> tuple[FilmSpan, ...]:
        """Label one stretch of film as plays and stretches that are not plays."""

    def segment(self, video_path: Path) -> tuple[PlaySegment, ...]:
        """Return plays in an uploaded file, in kickoff-to-whistle order."""

    def segment_youtube(self, url: str) -> tuple[PlaySegment, ...]:
        """Return plays in a public YouTube video, in kickoff-to-whistle order."""


class AnthropicPlayAgent:
    """Show Claude timestamped frames and read play boundaries back."""

    name = "anthropic-play-boundaries"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        client: object | None = None,
        frames: FrameSampler | None = None,
        downloader: Downloader | None = None,
    ) -> None:
        if not api_key.strip():
            raise MissingApiKeyError("Set ANTHROPIC_API_KEY to analyze film.")
        self.model = model
        self._api_key = api_key
        self._client = client
        self._frames = frames or extract_frames
        self._downloader = downloader or download_youtube

    def segment(self, video_path: Path) -> tuple[PlaySegment, ...]:
        if not video_path.is_file():
            raise VideoAnalysisError(f"Video not found: {video_path.name}")
        samples = self._frames(video_path)
        if not samples:
            raise VideoAnalysisError("Could not read frames from that video.")

        try:
            groups = []
            start = 0
            while start < len(samples):
                window = samples[start : start + _WINDOW]
                groups.append(parse_play_segments(self._ask(window)))
                if len(window) < _WINDOW:
                    break
                start += _WINDOW - 1
        except ValueError as error:
            raise VideoAnalysisError(
                f"The model returned play timestamps that could not be read. {error}"
            ) from error
        return _merge(groups)

    def label_window(self, samples: list[tuple[float, bytes]]) -> tuple[FilmSpan, ...]:
        """Label one short stretch as plays and everything that is not a play."""

        if not samples:
            return ()
        start = samples[0][0]
        end = samples[-1][0]
        prompt = SPAN_PROMPT.format(start=start, end=end)
        try:
            return parse_film_spans(self._ask(samples, prompt))
        except ValueError as error:
            raise VideoAnalysisError(
                f"The model returned film spans that could not be read. {error}"
            ) from error

    def segment_youtube(self, url: str) -> tuple[PlaySegment, ...]:
        try:
            canonical = canonical_youtube_url(url)
        except ValueError as error:
            raise VideoAnalysisError(str(error)) from error

        with tempfile.TemporaryDirectory() as tmp:
            video = self._downloader(canonical, Path(tmp))
            return self.segment(video)

    def _ask(self, samples: list[tuple[float, bytes]], prompt: str = PROMPT) -> str:
        client = self._client or _anthropic_client(self._api_key)
        content: list[dict[str, object]] = []
        for seconds, jpeg in samples:
            content.append({"type": "text", "text": f"t={seconds:.1f}s"})
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64.standard_b64encode(jpeg).decode("ascii"),
                    },
                }
            )
        content.append({"type": "text", "text": prompt})
        try:
            message = client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": content}],
            )
        except VideoAnalysisError:
            raise
        except Exception as error:
            raise VideoAnalysisError(_model_failure(error)) from error

        texts = [
            block.text
            for block in getattr(message, "content", [])
            if getattr(block, "type", None) == "text" and getattr(block, "text", "")
        ]
        text = "\n".join(texts).strip()
        if not text:
            raise VideoAnalysisError("The model returned no play timestamps.")
        return text


def agent_from_env() -> AnthropicPlayAgent:
    """Build the play agent from `ANTHROPIC_API_KEY`."""

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise MissingApiKeyError("Set ANTHROPIC_API_KEY to analyze film.")
    model = os.environ.get("FILM_ANALYSIS_MODEL", DEFAULT_MODEL)
    return AnthropicPlayAgent(api_key=api_key, model=model)


def _merge(groups: list[tuple[PlaySegment, ...]]) -> tuple[PlaySegment, ...]:
    ordered = sorted(
        (play for group in groups for play in group),
        key=lambda play: (play.start_seconds, play.end_seconds),
    )
    kept: list[PlaySegment] = []
    for play in ordered:
        if kept and play.start_seconds < kept[-1].end_seconds:
            if play.end_seconds > kept[-1].end_seconds:
                previous = kept[-1]
                kept[-1] = PlaySegment(
                    number=previous.number,
                    start_seconds=previous.start_seconds,
                    end_seconds=play.end_seconds,
                )
            continue
        kept.append(play)
    return tuple(
        PlaySegment(
            number=index,
            start_seconds=play.start_seconds,
            end_seconds=play.end_seconds,
        )
        for index, play in enumerate(kept, start=1)
    )


def _model_failure(error: Exception) -> str:
    detail = " ".join(str(error).split())
    if not detail:
        return "The model could not read that video."
    if len(detail) > 180:
        detail = detail[:177] + "..."
    return f"The model could not read that video. {detail}"


def _anthropic_client(api_key: str) -> object:
    import anthropic

    return anthropic.Anthropic(api_key=api_key)
