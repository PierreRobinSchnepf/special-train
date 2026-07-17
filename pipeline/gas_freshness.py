"""Reconstruction d'un total national gaz plus frais que l'agrégat officiel.

Constat (recherche documentée dans le README du pipeline) : le dataset ODRÉ
utilisé pour l'entraînement (`consommation-quotidienne-brute`) publie ses
valeurs "Définitif" avec ~45-50 jours de retard — inutilisable pour une
assimilation quotidienne. Deux datasets régionaux, sommés sur les 13 régions
et les 2 opérateurs (NaTran + Teréga), reconstituent le même total à ~0.2%
près, avec un retard réduit à ~15-20 jours :
  - `conso-journa-industriel-grtgazterega`      (clients industriels)
  - `courbe-de-charge-eldgrd-regional-grtgaz-terega` (distributions publiques, GRD/ELD)

Point non documenté ailleurs et vérifié empiriquement (comparaison heure par
heure contre notre dataset UTC de référence, cf. conversation) : les colonnes
horaires de ces deux datasets sont en heure LOCALE Europe/Paris (pas UTC, et
pas de décalage "jour gazier" malgré l'ordre d'export 06h→05h du dataset
industriel — cet ordre est un artefact d'affichage, chaque colonne reste
étiquetée par son heure locale réelle). Conversion locale→UTC obligatoire
avant toute fusion avec le reste du pipeline (qui est tout en UTC).
"""
from __future__ import annotations

import datetime as dt
import re
from io import StringIO

import pandas as pd
import requests

INDUSTRIAL_DATASET = "conso-journa-industriel-grtgazterega"
DISTRIBUTION_DATASET = "courbe-de-charge-eldgrd-regional-grtgaz-terega"
CALENDAR_TZ = "Europe/Paris"

_HOUR_COL_RE = re.compile(r"^(\d{2})_00")

_SESSION = requests.Session()


def _fetch_export_csv(dataset_id: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    url = f"https://odre.opendatasoft.com/api/v2/catalog/datasets/{dataset_id}/exports/csv"
    params = {
        "where": f"date in [date'{start.isoformat()}'..date'{end.isoformat()}']",
        "timezone": "UTC",
    }
    resp = _SESSION.get(url, params=params, timeout=60)
    resp.raise_for_status()
    return pd.read_csv(StringIO(resp.text), sep=";")


def _hour_columns(df: pd.DataFrame) -> dict[str, int]:
    """Mappe chaque colonne horaire à son heure locale (0-23), quel que soit
    le nom exact (les deux datasets ont des conventions de nommage différentes
    et parfois incohérentes en interne — cf. docstring du module)."""
    mapping = {}
    for col in df.columns:
        m = _HOUR_COL_RE.match(col)
        if m:
            mapping[col] = int(m.group(1))
    return mapping


def _wide_to_utc_series(df: pd.DataFrame) -> pd.Series:
    """Somme toutes les régions/opérateurs, convertit heure locale -> UTC,
    retourne une série indexée par timestamp UTC (MWh -> traité comme MW,
    cohérent avec le reste du pipeline : valeur horaire = puissance moyenne)."""
    hour_cols = _hour_columns(df)
    if not hour_cols:
        return pd.Series(dtype=float)

    long_rows = []
    dates = pd.to_datetime(df["date"]).dt.date
    for col, hour in hour_cols.items():
        local_ts = pd.to_datetime(dates.astype(str)) + pd.to_timedelta(hour, unit="h")
        long_rows.append(pd.DataFrame({"local_ts": local_ts, "value": df[col]}))

    long_df = pd.concat(long_rows, ignore_index=True).dropna(subset=["value"])
    grouped_local = long_df.groupby("local_ts")["value"].sum()

    # Localisation Europe/Paris -> UTC. Les rares heures ambiguës/inexistantes
    # (transitions DST) sont abandonnées plutôt que devinées.
    localized = grouped_local.index.tz_localize(CALENDAR_TZ, ambiguous="NaT", nonexistent="NaT")
    utc_series = pd.Series(grouped_local.to_numpy(), index=localized).dropna(how="all")
    utc_series = utc_series[utc_series.index.notna()]
    utc_series.index = utc_series.index.tz_convert("UTC")
    return utc_series.sort_index()


def fetch_fresh_gas_total(start: dt.date, end: dt.date) -> pd.Series:
    """Total national reconstitué (industriel + distribution), en MW, indexé
    UTC, pour la fenêtre [start, end]. Peut être plus court que demandé côté
    droit si les données ne sont pas encore publiées jusque `end` — c'est
    justement ce qu'on utilise pour détecter le "jour G" (dernière donnée
    disponible)."""
    industrial = _fetch_export_csv(INDUSTRIAL_DATASET, start, end)
    distribution = _fetch_export_csv(DISTRIBUTION_DATASET, start, end)

    industrial_utc = _wide_to_utc_series(industrial)
    distribution_utc = _wide_to_utc_series(distribution)

    total = industrial_utc.add(distribution_utc, fill_value=None)
    # add(fill_value=None) garde NaN si l'un des deux manque à une heure donnée
    # (on ne veut pas d'un total partiel silencieux) ; les deux sources
    # couvrent en pratique les mêmes jours donc c'est rare.
    return total.rename("y_gas_mw_fresh").dropna().sort_index()


def last_available_day(series: pd.Series) -> dt.date | None:
    """Dernier jour CIVIL (Europe/Paris) entièrement couvert par `series`
    (24 heures présentes) — le "jour G" jusqu'où l'état peut être mis à jour."""
    if series.empty:
        return None
    local_dates = series.index.tz_convert(CALENDAR_TZ).date
    counts = pd.Series(1, index=local_dates).groupby(level=0).sum()
    complete_days = counts[counts >= 24].index
    return max(complete_days) if len(complete_days) else None
