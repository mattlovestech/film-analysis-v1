"""Errors raised while charting a film."""


class VideoAnalysisError(Exception):
    """The film could not be segmented."""


class MissingApiKeyError(VideoAnalysisError):
    """No model credential is configured."""
