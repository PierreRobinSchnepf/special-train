"""Accès aux artefacts runtime depuis un stockage objet S3 (MinIO / SSP Cloud).

Objectif : rendre le dashboard déployable sur Streamlit Community Cloud, dont le
système de fichiers est éphémère et ne peut pas héberger les ~51 Mo d'artefacts
(datasets `processed` + modèles `.pkl`) — a fortiori pas les 1,3 Go de `data/raw`.

Stratégie « sync au boot » (la moins intrusive) : au démarrage de l'app, on
télécharge depuis S3 les préfixes `data/processed/` et `data/models/` vers le
disque local éphémère, PUIS tout le reste du code lit `data/` comme avant, sans
modification. Le 1,3 Go de `data/raw/` n'est jamais synchronisé (inutile à
l'affichage — seule l'actualisation, réservée à l'admin en local, en a besoin).

Détection du mode :
  - « cloud »  : `st.secrets` contient une section [s3]  -> on lit depuis S3.
  - « local »  : pas de secrets S3 -> comportement historique (lecture disque),
    aucun appel réseau, rien à configurer.

Configuration attendue dans `.streamlit/secrets.toml` (local) ou dans les
secrets Streamlit Cloud (cf. DEPLOY.md) :

    admin_password = "..."          # débloque les boutons d'écriture

    [s3]
    endpoint_url = "https://minio.lab.sspcloud.fr"
    aws_access_key_id = "..."
    aws_secret_access_key = "..."
    aws_session_token = ""          # optionnel (identifiants temporaires Onyxia)
    bucket = "mon-bucket"
    prefix = "gaz-dashboard"        # préfixe racine dans le bucket (optionnel)
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Préfixes (relatifs à la racine du repo) synchronisés depuis S3 au boot.
# On NE synchronise PAS data/raw (1,3 Go, inutile à l'affichage).
RUNTIME_PREFIXES = ("data/processed", "data/models")


def _secrets() -> dict:
    """Renvoie st.secrets sous forme de dict, ou {} hors contexte Streamlit."""
    try:
        import streamlit as st

        # st.secrets se comporte comme un mapping ; l'accès lève si absent.
        return dict(st.secrets)
    except Exception:
        return {}


def is_cloud() -> bool:
    """True si une configuration S3 est présente (déploiement)."""
    return "s3" in _secrets()


def admin_password() -> str | None:
    return _secrets().get("admin_password")


def admin_unlocked() -> bool:
    """En local : toujours débloqué. En cloud : seulement si le mot de passe
    admin a été saisi et validé dans la session (cf. dashboard.services.auth)."""
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
    """Télécharge data/processed + data/models depuis S3 vers le disque local.

    Idempotent : saute un objet déjà présent localement à la même taille. Sûr en
    local (no-op si pas de config S3). Retourne un petit résumé.
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
                # clé distante -> chemin local (on retire le préfixe racine)
                rel = key[len(_remote_key(conf, "")):] if conf.get("prefix") else key
                local_path = REPO_ROOT / rel
                if local_path.exists() and local_path.stat().st_size == obj["Size"]:
                    skipped += 1
                    continue
                local_path.parent.mkdir(parents=True, exist_ok=True)
                log(f"[get] {rel} ({obj['Size'] / 1e6:.1f} Mo)")
                client.download_file(bucket, key, str(local_path))
                downloaded += 1

    return {"mode": "cloud", "downloaded": downloaded, "skipped": skipped}


def upload_runtime_artifacts(log=print) -> dict:
    """Pousse data/processed + data/models du disque local vers S3.

    Utilisé par scripts/upload_artifacts.py après un (ré)entraînement local.
    Requiert une config S3 (secrets). N'envoie jamais data/raw.
    """
    if not is_cloud():
        raise RuntimeError(
            "Pas de configuration S3 dans st.secrets — impossible d'uploader. "
            "Renseigne .streamlit/secrets.toml (section [s3])."
        )
    client, conf = _s3_client_and_conf()
    bucket = conf["bucket"]
    uploaded = 0

    for prefix in RUNTIME_PREFIXES:
        base = REPO_ROOT / prefix
        if not base.exists():
            log(f"(absent, ignoré) {prefix}")
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(REPO_ROOT).as_posix()
            key = _remote_key(conf, rel)
            log(f"[put] {rel} ({path.stat().st_size / 1e6:.1f} Mo)")
            client.upload_file(str(path), bucket, key)
            uploaded += 1

    return {"bucket": bucket, "uploaded": uploaded}
