"""Hand a film to the Cloudflare cutter instead of processing it in the website."""

import json
import os
import urllib.error
import urllib.request

from film_analysis.storage.catalog import CatalogError


def cloud_target(header: str | None) -> tuple[str, str] | None:
    """Return the worker URL and token when this process should hand work off."""

    token = os.environ.get("FILM_INTERNAL_TOKEN", "").strip()
    configured = os.environ.get("FILM_WORKER_URL", "").strip()
    url = configured or (header or "").strip()
    if not token or not url:
        return None
    return url.rstrip("/"), token


def start_remote(
    worker: str,
    token: str,
    project_id: int,
    youtube: str | None,
    workers: int,
) -> None:
    """Ask the cutter for this film to start. The website does not wait for the cut."""

    body = json.dumps(
        {
            "project_id": project_id,
            "youtube": youtube,
            "worker_url": worker,
            "workers": workers,
        }
    ).encode()
    request = urllib.request.Request(
        f"{worker}/internal/films/{project_id}/start",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            if response.status >= 300:
                raise CatalogError("Could not start that film.")
    except urllib.error.HTTPError as error:
        raise CatalogError("Could not start that film.") from error
    except urllib.error.URLError as error:
        raise CatalogError("Could not reach the cutter.") from error
