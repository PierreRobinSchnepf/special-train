"""Téléchargement HTTP idempotent avec retry, pour tous les scripts fetch_*."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "gas-consumption-forecast/1.0 (data pipeline)"})


def download(
    url: str,
    dest: Path,
    *,
    force: bool = False,
    max_retries: int = 4,
    timeout: int = 60,
    params: dict | None = None,
) -> Path:
    """Télécharge `url` vers `dest`. Idempotent : skip si `dest` existe déjà et
    n'est pas vide, sauf si force=True. Retry avec backoff exponentiel sur
    erreurs réseau/5xx."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0 and not force:
        logger.info("skip (cache hit): %s", dest)
        return dest

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = _SESSION.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(resp.content)
            tmp.replace(dest)
            logger.info("downloaded (%d bytes): %s", len(resp.content), dest)
            return dest
        except (requests.RequestException, OSError) as exc:
            last_exc = exc
            wait = 2 ** attempt
            logger.warning(
                "download failed (attempt %d/%d) for %s: %s — retry in %ds",
                attempt, max_retries, url, exc, wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"failed to download {url} after {max_retries} attempts") from last_exc


def get_json(url: str, *, params: dict | None = None, timeout: int = 60, max_retries: int = 4) -> dict:
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = _SESSION.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, OSError) as exc:
            last_exc = exc
            wait = 2 ** attempt
            logger.warning(
                "GET json failed (attempt %d/%d) for %s: %s — retry in %ds",
                attempt, max_retries, url, exc, wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"failed to GET {url} after {max_retries} attempts") from last_exc
