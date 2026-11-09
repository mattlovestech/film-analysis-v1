"""Play clips in a Cloudflare R2 bucket."""

import shutil
from pathlib import Path

from botocore.exceptions import ClientError

from film_analysis.storage.catalog import CatalogError


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

    def save_source(self, project_id: int, source: Path) -> str:
        return self._put(source, f"projects/{project_id}/source.mp4")

    def save_play(self, project_id: int, play_number: int, clip: Path) -> str:
        return self._put(clip, f"projects/{project_id}/plays/{play_number}.mp4")

    def play_path(self, project_id: int, play_number: int) -> Path:
        key = f"projects/{project_id}/plays/{play_number}.mp4"
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

    def _put(self, source: Path, key: str) -> str:
        try:
            self._client.upload_file(
                str(source),
                self.bucket,
                key,
                ExtraArgs={"ContentType": "video/mp4"},
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


def _client(endpoint: str, access_key: str, secret_key: str) -> object:
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )
