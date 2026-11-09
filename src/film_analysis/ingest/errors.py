"""Errors raised while charting a film."""


class VideoAnalysisError(Exception):
    """The film could not be segmented."""


class MissingApiKeyError(VideoAnalysisError):
    """No model credential is configured."""


def tool_failure(message: str, stderr: str | bytes | None, code: int) -> str:
    """Attach the tool's own error text so a failure can be diagnosed."""

    raw = stderr or ""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    tail = "\n".join(lines[-6:])
    if len(tail) > 700:
        tail = tail[-700:]
    if tail:
        return f"{message}\n{tail}"
    return f"{message} The tool exited {code}."
