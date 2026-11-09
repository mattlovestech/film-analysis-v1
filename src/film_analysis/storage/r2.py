"""Play clips in a Cloudflare R2 bucket."""

import contextlib
import json
import shutil
import tempfile
import time
from pathlib import Path

from botocore.exceptions import ClientError

from film_analysis.storage.catalog import (
    PLAY_IMAGE_KINDS,
    PLAY_LAYOUT_KIND,
    TEAM_IMAGE_KINDS,
    CatalogError,
    team_media,
)


class R2Clips:
    """Upload each play to R2 and keep a local copy for the video player."""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        cache: Path,
        client: object | None = None,
    ) -> None:
        self.bucket = bucket
        self.cache = cache
        self._client = client or _client(endpoint, access_key, secret_key)
        self._missing_posters: dict[int, float] = {}

    def save_source(self, project_id: int, source: Path) -> str:
        return self._put(source, f"projects/{project_id}/source.mp4")

    def save_play(self, project_id: int, play_number: int, clip: Path) -> str:
        return self._put(clip, f"projects/{project_id}/plays/{play_number}.mp4")

    def play_path(self, project_id: int, play_number: int) -> Path:
        return self._local(f"projects/{project_id}/plays/{play_number}.mp4")

    def save_play_image(
        self, project_id: int, play_number: int, kind: str, data: bytes
    ) -> str:
        cleaned = (kind or "").strip().lower()
        if cleaned not in PLAY_IMAGE_KINDS:
            raise CatalogError("Could not store that play image.")
        key = f"projects/{project_id}/plays/{play_number}-{cleaned}.jpg"
        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            tmp.write(data)
            tmp.flush()
            return self._put(Path(tmp.name), key, "image/jpeg")

    def play_image(self, project_id: int, play_number: int, kind: str) -> Path | None:
        cleaned = (kind or "").strip().lower()
        if cleaned not in PLAY_IMAGE_KINDS:
            return None
        path = self._local(f"projects/{project_id}/plays/{play_number}-{cleaned}.jpg")
        if path.is_file() and path.stat().st_size:
            return path
        return None

    def save_play_layout(
        self, project_id: int, play_number: int, payload: dict[str, object]
    ) -> str:
        key = f"projects/{project_id}/plays/{play_number}-{PLAY_LAYOUT_KIND}.json"
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", encoding="utf-8") as tmp:
            tmp.write(json.dumps(payload))
            tmp.flush()
            return self._put(Path(tmp.name), key, "application/json")

    def play_layout(self, project_id: int, play_number: int) -> dict[str, object] | None:
        path = self._local(f"projects/{project_id}/plays/{play_number}-{PLAY_LAYOUT_KIND}.json")
        if not path.is_file() or not path.stat().st_size:
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return raw if isinstance(raw, dict) else None

    def source_path(self, project_id: int) -> Path:
        return self._local(f"projects/{project_id}/source.mp4")

    def copy_source(self, source_id: int, project_id: int) -> str | None:
        source_key = f"projects/{source_id}/source.mp4"
        dest_key = f"projects/{project_id}/source.mp4"
        try:
            self._client.copy_object(
                Bucket=self.bucket,
                Key=dest_key,
                CopySource={"Bucket": self.bucket, "Key": source_key},
            )
        except ClientError:
            return None
        cached = self.cache / source_key
        if cached.is_file():
            dest = self.cache / dest_key
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached, dest)
        return dest_key

    def source_url(self, project_id: int, expires: int = 10800) -> str:
        """A temporary HTTPS link a labeler or cutter can stream from."""

        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": f"projects/{project_id}/source.mp4"},
            ExpiresIn=expires,
        )

    def _local(self, key: str) -> Path:
        dest = self.cache / key
        if dest.is_file():
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._client.download_file(self.bucket, key, str(dest))
        except ClientError:
            dest.unlink(missing_ok=True)
        return dest

    def remove(self, project_id: int) -> None:
        prefix = f"projects/{project_id}/"
        token: str | None = None
        while True:
            kwargs: dict[str, str] = {"Bucket": self.bucket, "Prefix": prefix}
            if token is not None:
                kwargs["ContinuationToken"] = token
            page = self._client.list_objects_v2(**kwargs)
            keys = [{"Key": item["Key"]} for item in page.get("Contents", [])]
            if keys:
                self._client.delete_objects(Bucket=self.bucket, Delete={"Objects": keys})
            if not page.get("IsTruncated"):
                break
            token = page.get("NextContinuationToken")
        shutil.rmtree(self.cache / "projects" / str(project_id), ignore_errors=True)

    def save_team_image(self, project_id: int, kind: str, data: bytes, suffix: str) -> str:
        cleaned, content_type = team_media(kind, suffix)
        key = f"projects/{project_id}/teams/{kind}{cleaned}"
        other = ".svg" if cleaned == ".jpg" else ".jpg"
        other_key = f"projects/{project_id}/teams/{kind}{other}"
        with contextlib.suppress(ClientError):
            self._client.delete_object(Bucket=self.bucket, Key=other_key)
        (self.cache / other_key).unlink(missing_ok=True)
        with tempfile.NamedTemporaryFile(suffix=cleaned) as tmp:
            tmp.write(data)
            tmp.flush()
            return self._put(Path(tmp.name), key, content_type)

    def clear_team_image(self, project_id: int, kind: str) -> None:
        from film_analysis.storage.catalog import TEAM_IMAGE_KINDS, CatalogError

        if kind not in TEAM_IMAGE_KINDS:
            raise CatalogError("Could not store that team image.")
        for suffix in (".jpg", ".svg"):
            key = f"projects/{project_id}/teams/{kind}{suffix}"
            with contextlib.suppress(ClientError):
                self._client.delete_object(Bucket=self.bucket, Key=key)
            (self.cache / key).unlink(missing_ok=True)
        if kind.endswith("-player"):
            marker = self.cache / f"projects/{project_id}/teams/.player-crops"
            left = any(
                (self.cache / f"projects/{project_id}/teams/left-player{suffix}").is_file()
                for suffix in (".jpg", ".svg")
            )
            right = any(
                (self.cache / f"projects/{project_id}/teams/right-player{suffix}").is_file()
                for suffix in (".jpg", ".svg")
            )
            if not left and not right:
                marker.unlink(missing_ok=True)

    def team_image(self, project_id: int, kind: str) -> Path | None:
        if kind not in TEAM_IMAGE_KINDS:
            return None
        for suffix in (".jpg", ".svg"):
            path = self._local(f"projects/{project_id}/teams/{kind}{suffix}")
            if path.is_file() and path.stat().st_size:
                return path
        return None

    def save_poster(self, project_id: int, data: bytes) -> str:
        self._missing_posters.pop(project_id, None)
        key = f"projects/{project_id}/poster.jpg"
        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            tmp.write(data)
            tmp.flush()
            return self._put(Path(tmp.name), key, "image/jpeg")

    def poster_path(self, project_id: int) -> Path | None:
        cached = self.cache / f"projects/{project_id}/poster.jpg"
        if cached.is_file() and cached.stat().st_size:
            self._missing_posters.pop(project_id, None)
            return cached
        missed = self._missing_posters.get(project_id)
        if missed is not None and time.monotonic() - missed < 30:
            return None
        path = self._local(f"projects/{project_id}/poster.jpg")
        if path.is_file() and path.stat().st_size:
            self._missing_posters.pop(project_id, None)
            return path
        self._missing_posters[project_id] = time.monotonic()
        return None

    def _put(self, source: Path, key: str, content_type: str = "video/mp4") -> str:
        try:
            self._client.upload_file(
                str(source),
                self.bucket,
                key,
                ExtraArgs={"ContentType": content_type},
            )
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code", "")
            message = "Could not store that clip."
            if code:
                message = f"{message} {code}"
            raise CatalogError(message) from error
        cached = self.cache / key
        cached.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, cached)
        return key


def source_url(project_id: int) -> str:
    """Presign the film for this project so a task can read a stretch of it."""

    import os

    endpoint = os.environ.get("CLOUDFLARE_S3_ENDPOINT", "")
    access_key = os.environ.get("CLOUDFLARE_ACCESS_KEY_ID", "")
    secret_key = os.environ.get("CLOUDFLARE_SECRET_ACCESS_KEY", "")
    bucket = os.environ.get("R2_BUCKET", "")
    if not (endpoint and access_key and secret_key and bucket):
        raise CatalogError("Could not read that film.")
    cache = Path(os.environ.get("FILM_ANALYSIS_DATA", "data"))
    return R2Clips(endpoint, access_key, secret_key, bucket, cache).source_url(project_id)


def _client(endpoint: str, access_key: str, secret_key: str) -> object:
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )
