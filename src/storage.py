"""Runtime-artifact access backed by S3 object storage (MinIO / SSP Cloud).

Purpose: make the dashboard deployable on Streamlit Community Cloud, whose
filesystem is ephemeral and cannot ship the ~50 MB of runtime artifacts
(`processed` datasets + model `.pkl` files) — let alone the 1.3 GB of
`data/raw`.

"Sync at boot" strategy (the least intrusive): when the app starts, the
`data/processed/` and `data/models/` prefixes are downloaded from S3 to the
local ephemeral disk, THEN all the existing code reads `data/` exactly as
before, unmodified. The 1.3 GB of `data/raw/` is never synced (useless for
display — only the admin-only, local refresh needs it).

Mode detection:
  - "cloud": `st.secrets` contains an [s3] section -> read from S3.
  - "local": no S3 secrets -> historical behavior (disk reads), no network
    calls, nothing to configure.

Expected configuration in `.streamlit/secrets.toml` (local) or in the
Streamlit Cloud secrets (see docs/deployment.md):

    admin_password = "..."          # unlocks the write buttons

    [s3]
    endpoint_url = "https://minio.lab.sspcloud.fr"
    aws_access_key_id = "..."
    aws_secret_access_key = "..."
    aws_session_token = ""          # optional (temporary Onyxia credentials)
    bucket = "my-bucket"
    prefix = "gas-dashboard"        # root prefix inside the bucket (optional)
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Prefixes (relative to the repo root) synced from S3 at boot.
# data/raw is NOT synced (1.3 GB, useless for display).
RUNTIME_PREFIXES = ("data/processed", "data/models")


def _secrets() -> dict:
    """Return st.secrets as a dict, or {} outside a Streamlit context."""
    try:
        import streamlit as st

        # st.secrets behaves like a mapping; access raises when absent.
        return dict(st.secrets)
    except Exception:
        return {}


def is_cloud() -> bool:
    """True when an S3 configuration is present (deployment)."""
    return "s3" in _secrets()


def admin_password() -> str | None:
    return _secrets().get("admin_password")


def admin_unlocked() -> bool:
    """Local mode: always unlocked. Cloud mode: only when the admin password
    has been entered and validated in the session (see dashboard.services.auth)."""
    if not is_cloud():
        return True
    try:
        import streamlit as st

        return bool(st.session_state.get("_admin_ok", False))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------
def _s3_client_and_conf() -> tuple[object, dict]:
    import boto3

    conf = _secrets()["s3"]
    session_token = conf.get("aws_session_token") or None
    client = boto3.client(
        "s3",
        endpoint_url=conf["endpoint_url"],
        aws_access_key_id=conf["aws_access_key_id"],
        aws_secret_access_key=conf["aws_secret_access_key"],
        aws_session_token=session_token,
    )
    return client, conf


def _remote_key(conf: dict, rel_path: str) -> str:
    prefix = (conf.get("prefix") or "").strip("/")
    return f"{prefix}/{rel_path}" if prefix else rel_path


def download_runtime_artifacts(log=lambda m: None) -> dict:
    """Download data/processed + data/models from S3 to the local disk.

    Idempotent: skips any object already present locally at the same size.
    Safe locally (no-op without S3 config). Returns a small summary.
    """
    if not is_cloud():
        return {"mode": "local", "downloaded": 0, "skipped": 0}

    client, conf = _s3_client_and_conf()
    bucket = conf["bucket"]
    downloaded = skipped = 0

    for prefix in RUNTIME_PREFIXES:
        remote_prefix = _remote_key(conf, prefix)
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=remote_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                # remote key -> local path (strip the root prefix)
                rel = key[len(_remote_key(conf, "")):] if conf.get("prefix") else key
                local_path = REPO_ROOT / rel
                if local_path.exists() and local_path.stat().st_size == obj["Size"]:
                    skipped += 1
                    continue
                local_path.parent.mkdir(parents=True, exist_ok=True)
                log(f"[get] {rel} ({obj['Size'] / 1e6:.1f} MB)")
                client.download_file(bucket, key, str(local_path))
                downloaded += 1

    return {"mode": "cloud", "downloaded": downloaded, "skipped": skipped}


def upload_runtime_artifacts(log=print) -> dict:
    """Push data/processed + data/models from the local disk to S3.

    Used by scripts/upload_artifacts.py after a local (re)training. Requires
    an S3 config (secrets). Never uploads data/raw.
    """
    if not is_cloud():
        raise RuntimeError(
            "No S3 configuration in st.secrets — cannot upload. "
            "Fill in .streamlit/secrets.toml (section [s3])."
        )
    client, conf = _s3_client_and_conf()
    bucket = conf["bucket"]
    uploaded = 0

    for prefix in RUNTIME_PREFIXES:
        base = REPO_ROOT / prefix
        if not base.exists():
            log(f"(absent, skipped) {prefix}")
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(REPO_ROOT).as_posix()
            key = _remote_key(conf, rel)
            log(f"[put] {rel} ({path.stat().st_size / 1e6:.1f} MB)")
            client.upload_file(str(path), bucket, key)
            uploaded += 1

    return {"bucket": bucket, "uploaded": uploaded}
