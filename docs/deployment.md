# Deploying the dashboard on Streamlit Community Cloud

The dashboard needs ~50 MB of runtime artifacts (processed datasets + model
pickles) that are gitignored and cannot live in the ephemeral Streamlit
Cloud filesystem. The deployment strategy is **sync at boot**: artifacts are
stored in an S3 bucket (MinIO on [SSP Cloud / Onyxia](https://datalab.sspcloud.fr)),
and the app downloads them once per container at startup (`src/storage.py`).

Locally, nothing changes: without S3 secrets the app reads `data/` from disk
and every feature stays unlocked.

## 1. One-time setup

### S3 bucket (SSP Cloud)

1. Log in to SSP Cloud (Onyxia) → *My account* → *Storage connection* to
   get your MinIO credentials.
2. ⚠️ The credentials shown there are a **temporary session token (~7
   days)**. For a durable deployment, create a **permanent access key**
   instead: open the MinIO console → *Access Keys* → *Create access key*
   (no expiry), and use that pair with an empty `aws_session_token`.

### Local secrets

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Fill in (the file is gitignored — never commit it):

```toml
admin_password = "change-me"          # unlocks the write buttons in the app

[s3]
endpoint_url = "https://minio.lab.sspcloud.fr"
aws_access_key_id = "..."
aws_secret_access_key = "..."
aws_session_token = ""                # empty with a permanent key
bucket = "your-bucket"
prefix = "gas-dashboard"              # optional root prefix
```

## 2. Upload the artifacts

After any local (re)training:

```bash
python scripts/upload_artifacts.py
```

This pushes `data/processed/` + `data/models/` (never `data/raw/`, 1.3 GB)
to `s3://<bucket>/<prefix>/`.

## 3. Create the Streamlit Cloud app

1. Push the repo to GitHub (code only — `data/` and secrets are ignored).
2. On <https://share.streamlit.io> → *New app* → select the repo, main file
   **`dashboard/app.py`**.
3. *App settings → Secrets*: paste the full content of your local
   `.streamlit/secrets.toml`.
4. Deploy. At boot the app downloads the artifacts from S3 (once per
   container; later visitors reuse the cache).

## 4. Access control

With S3 secrets present, the app runs in "cloud" mode:

- The **Check & refresh** button (heavy download + retraining + writes) and
  the **live forecast** button (external API calls + DB writes) are hidden.
- Entering `admin_password` in the sidebar (*Admin access*) unlocks them
  for the session.

## 5. Refreshing the published data

The refresh workflow is intentionally local-first:

```bash
python scripts/fetch_data.py --only gas_regional
python scripts/build_regional_dataset.py
python scripts/train_regional_models.py
python scripts/upload_artifacts.py     # push the new artifacts
```

then *Reboot* the Streamlit Cloud app (or wait for its container to
recycle) so it re-syncs from S3.

## Troubleshooting

- **`FileNotFoundError: dataset_final.parquet` on the deployed app** — the
  S3 secrets are missing or wrong in *App settings → Secrets*: the app fell
  back to "local" mode and found an empty `data/` directory.
- **Long first load** — the ~50 MB S3 sync runs once per container (cached
  with `@st.cache_resource`); every later visitor skips it. Sleeping apps
  (free tier) re-sync when they wake up.
- **Credentials expired** — Onyxia session tokens last ~7 days; switch to a
  permanent MinIO access key (§1).
