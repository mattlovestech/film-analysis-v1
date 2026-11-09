from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from film_analysis.storage.catalog import CatalogError
from film_analysis.storage.r2 import R2Clips


class _Bucket:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def upload_file(
        self,
        filename: str,
        bucket: str,
        key: str,
        ExtraArgs: dict | None = None,
    ) -> None:
        del bucket, ExtraArgs
        self.objects[key] = Path(filename).read_bytes()

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        del bucket
        body = self.objects.get(key)
        if body is None:
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "GetObject")
        Path(filename).write_bytes(body)

    def list_objects_v2(self, **kwargs: str) -> dict:
        prefix = kwargs["Prefix"]
        contents = [{"Key": key} for key in self.objects if key.startswith(prefix)]
        return {"Contents": contents, "IsTruncated": False}

    def delete_objects(self, Bucket: str, Delete: dict) -> None:
        del Bucket
        for item in Delete["Objects"]:
            self.objects.pop(item["Key"], None)


def _clips(tmp_path: Path, bucket: _Bucket) -> R2Clips:
    return R2Clips(
        "https://example.r2.cloudflarestorage.com",
        "key",
        "secret",
        "film-analysis",
        tmp_path,
        bucket,
    )


def test_play_clip_round_trips_through_the_bucket(tmp_path: Path) -> None:
    bucket = _Bucket()
    clips = _clips(tmp_path, bucket)
    source = tmp_path / "play.mp4"
    source.write_bytes(b"play-bytes")

    key = clips.save_play(4, 2, source)
    cached = tmp_path / key
    cached.unlink()

    assert key == "projects/4/plays/2.mp4"
    assert clips.play_path(4, 2).read_bytes() == b"play-bytes"
    clips.remove(4)
    assert bucket.objects == {}
    assert clips.play_path(4, 2).is_file() is False


def test_upload_failure_names_the_bucket_error(tmp_path: Path) -> None:
    class _Broken(_Bucket):
        def upload_file(
            self,
            filename: str,
            bucket: str,
            key: str,
            ExtraArgs: dict | None = None,
        ) -> None:
            del filename, bucket, key, ExtraArgs
            raise ClientError({"Error": {"Code": "AccessDenied", "Message": "no"}}, "PutObject")

    clips = _clips(tmp_path, _Broken())
    source = tmp_path / "play.mp4"
    source.write_bytes(b"play-bytes")

    with pytest.raises(CatalogError, match="AccessDenied"):
        clips.save_play(1, 1, source)
