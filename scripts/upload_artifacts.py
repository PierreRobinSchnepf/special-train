"""Pousse les artefacts runtime (data/processed + data/models) vers S3/MinIO.

À relancer après chaque (ré)entraînement local pour rafraîchir ce que voit le
dashboard déployé. N'envoie JAMAIS data/raw (1,3 Go, inutile en ligne).

Prérequis : un fichier `.streamlit/secrets.toml` valide à la racine du repo
(section [s3]) — voir `.streamlit/secrets.toml.example` et DEPLOY.md.

    python scripts/upload_artifacts.py

Ce script lit les secrets DIRECTEMENT depuis le fichier TOML (il ne tourne pas
dans un contexte Streamlit) et appelle la même logique d'upload que l'app.
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
        print(f"❌ {SECRETS_PATH} introuvable. Copie .streamlit/secrets.toml.example "
              f"et renseigne la section [s3].")
        return 1

    with open(SECRETS_PATH, "rb") as f:
        secrets = tomllib.load(f)
    if "s3" not in secrets:
        print("❌ Section [s3] absente de secrets.toml.")
        return 1

    # On injecte les secrets dans un faux st.secrets pour réutiliser src.storage
    # sans dépendre d'un runtime Streamlit.
    import src.storage as storage

    storage._secrets = lambda: secrets  # type: ignore[assignment]

    result = storage.upload_runtime_artifacts(log=print)
    print(f"✅ {result['uploaded']} fichier(s) envoyé(s) vers s3://{result['bucket']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
