"""Chargement de la configuration unique du pipeline (config.yaml)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = REPO_ROOT / "config.yaml") -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative: str) -> Path:
    """Résout un chemin de config.yaml (toujours relatif à la racine du repo)."""
    p = Path(relative)
    return p if p.is_absolute() else REPO_ROOT / p
