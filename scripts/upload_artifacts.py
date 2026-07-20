"""Push the runtime artifacts (data/processed + data/models) to S3/MinIO.

Rerun after every local (re)training to refresh what the deployed dashboard
sees. NEVER uploads data/raw (1.3 GB, useless online).

Prerequisite: a valid `.streamlit/secrets.toml` at the repo root ([s3]
section) — see `.streamlit/secrets.toml.example` and docs/deployment.md.

    python scripts/upload_artifacts.py

This script reads the secrets DIRECTLY from the TOML file (it does not run in
a Streamlit context) and calls the same upload logic as the app.
"""
from __future__ import annotations

import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SECRETS_PATH = REPO_ROOT / ".streamlit" / "secrets.toml"


def main() -> int:
    if not SECRETS_PATH.exists():
        print(f"[error] {SECRETS_PATH} not found. Copy .streamlit/secrets.toml.example "
              f"and fill in the [s3] section.")
        return 1

    with open(SECRETS_PATH, "rb") as f:
        secrets = tomllib.load(f)
    if "s3" not in secrets:
        print("[error] [s3] section missing from secrets.toml.")
        return 1

    # Inject the secrets into a fake st.secrets so src.storage can be reused
    # without a Streamlit runtime.
    import src.storage as storage

    storage._secrets = lambda: secrets  # type: ignore[assignment]

    result = storage.upload_runtime_artifacts(log=print)
    print(f"[ok] {result['uploaded']} file(s) uploaded to s3://{result['bucket']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
