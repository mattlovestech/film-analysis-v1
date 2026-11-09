import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from film_analysis.agents.errors import MissingApiKeyError, VideoAnalysisError
from film_analysis.agents.frames import extract_frames
from film_analysis.agents.play_boundaries import AnthropicPlayAgent, agent_from_env


class _Messages:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.text)])


class _Client:
    def __init__(self, messages: _Messages) -> None:
        self.messages = messages


def _frames(path: Path) -> list[tuple[float, bytes]]:
    assert path.is_file()
    return [(1.0, b"jpeg-a"), (2.0, b"jpeg-b")]


def test_agent_sends_timestamped_frames(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    messages = _Messages('{"plays": [{"start_seconds": 1, "end_seconds": 2}]}')
    agent = AnthropicPlayAgent(api_key="test-key", client=_Client(messages), frames=_frames)

    segments = agent.segment(video)

    assert [segment.label for segment in segments] == ["Play 1: 0:01 - 0:02"]
    call = messages.calls[0]
    assert call["model"] == "claude-sonnet-5-5"
    content = call["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "t=1.0s"}
    assert content[1]["type"] == "image"
    assert content[1]["source"]["media_type"] == "image/jpeg"
    assert "snap" in content[-1]["text"]


def test_youtube_link_is_downloaded_then_charted(tmp_path: Path) -> None:
    seen: list[str] = []

    def download(url: str, directory: Path) -> Path:
        seen.append(url)
        video = directory / "film.mp4"
        video.write_bytes(b"video")
        return video

    messages = _Messages('{"plays": [{"start_seconds": 1, "end_seconds": 2}]}')
    agent = AnthropicPlayAgent(
        api_key="test-key",
        client=_Client(messages),
        frames=_frames,
        downloader=download,
    )

    segments = agent.segment_youtube("https://www.youtube.com/watch?v=5mHoasOclJs")

    assert [segment.label for segment in segments] == ["Play 1: 0:01 - 0:02"]
    assert seen == ["https://www.youtube.com/watch?v=5mHoasOclJs"]


def test_api_error_includes_the_reason(tmp_path: Path) -> None:
    class _Broken:
        def create(self, **kwargs: object) -> SimpleNamespace:
            raise RuntimeError("credit balance is too low")

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"video")
    agent = AnthropicPlayAgent(
        api_key="test-key",
        client=SimpleNamespace(messages=_Broken()),
        frames=_frames,
    )

    with pytest.raises(VideoAnalysisError, match="credit balance"):
        agent.segment(video)


def test_non_youtube_link_is_rejected_before_download() -> None:
    def download(url: str, directory: Path) -> Path:
        raise AssertionError("should not download")

    agent = AnthropicPlayAgent(
        api_key="test-key",
        client=_Client(_Messages("{}")),
        downloader=download,
    )

    with pytest.raises(VideoAnalysisError, match="YouTube"):
        agent.segment_youtube("https://vimeo.com/123456789")


def test_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(MissingApiKeyError, match="ANTHROPIC_API_KEY"):
        agent_from_env()


def test_extract_frames_from_a_short_clip(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
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
            str(video),
        ],
        check=True,
    )

    frames = extract_frames(video)

    assert [timestamp for timestamp, _jpeg in frames[:2]] == [0.0, 1.0]
    assert frames[0][1].startswith(b"\xff\xd8")
