"""Ingestion et reconstruction de la consommation gaz RÉGIONALE (cible du
modèle région par région).

Le dataset national `consommation-quotidienne-brute` n'est pas ventilé par
région. On reconstitue la consommation par région en sommant, par région, deux
datasets ODRÉ régionaux :
  - `conso-journa-industriel-grtgazterega`          (clients industriels)
  - `courbe-de-charge-eldgrd-regional-grtgaz-terega` (distributions publiques, GRD/ELD)

C'est la même paire que `pipeline/gas_freshness.py`, à une différence près : ce
module **garde la maille région** (`code_region`) au lieu de tout sommer pour
reconstituer un total national. La mécanique de parsing est identique et
partage les mêmes propriétés vérifiées empiriquement :

- format LARGE : une colonne par heure (`06_00`, `07_00`, ... ; les deux
  datasets ont des conventions de nommage internes incohérentes — `00_00_00`
  vs `07_00` — d'où le regex tolérant `^(\\d{2})_00`) ;
- heure **LOCALE Europe/Paris** (pas UTC, pas de décalage "jour gazier") :
  conversion locale -> UTC obligatoire avant toute fusion avec le reste du
  pipeline (tout en UTC) ;
- plusieurs lignes par (date, région) : un par opérateur (NaTran + Teréga) et
  par secteur d'activité — on **somme toutes ces lignes** pour obtenir le total
  régional.

On indexe toujours sur le `code_region` numérique (stable), jamais sur le
libellé texte (qui varie entre les deux datasets : "Grand-Est"/"Grand Est",
"Ile-de-France"/"Île-de-France"...).
"""
from __future__ import annotations

import datetime as dt
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import resolve_path
from src.http_utils import download

logger = logging.getLogger(__name__)

CALENDAR_TZ = "Europe/Paris"
_HOUR_COL_RE = re.compile(r"^(\d{2})_00")


# ---------------------------------------------------------------------------
# Étape 1 : ingestion brute (cache idempotent, par dataset et par année)
# ---------------------------------------------------------------------------

def _raw_dir(config: dict) -> Path:
    return resolve_path(config["output"]["raw_dir"]) / config["gas_regional"]["raw_subdir"]


def _export_url(config: dict, dataset_id: str) -> str:
    return config["gas_regional"]["export_url_template"].format(dataset_id=dataset_id)


def latest_available_source_day(config: dict) -> dt.date | None:
    """Interroge ODRÉ pour le dernier jour publié dans la SOURCE régionale (les
    deux datasets se mettent à jour ~tous les 15-20 j). Retourne le min des deux
    max (un total complet a besoin des deux). Sert au bouton "actualiser" pour
    savoir s'il y a du neuf à télécharger sans tout re-fetcher."""
    import requests

    reg_cfg = config["gas_regional"]
    date_field = reg_cfg["date_field"]
    maxima = []
    for key in ("industrial_dataset_id", "distribution_dataset_id"):
        url = f"https://odre.opendatasoft.com/api/v2/catalog/datasets/{reg_cfg[key]}/records"
        params = {"select": date_field, "order_by": f"{date_field} desc", "limit": 1}
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            recs = resp.json().get("records", [])
            if recs:
                d = recs[0]["record"]["fields"][date_field]
                maxima.append(dt.date.fromisoformat(d[:10]))
        except (requests.RequestException, KeyError, ValueError) as exc:
            logger.warning("freshness check failed for %s: %s", key, exc)
    return min(maxima) if maxima else None


def fetch_regional_gas(
    config: dict,
    force: bool = False,
    refresh_current_year: bool = True,
) -> None:
    """Télécharge les deux datasets régionaux par chunk annuel dans
    data/raw/<raw_subdir>/. Idempotent (skip si déjà en cache), sauf l'année en
    cours qui est retéléchargée (données encore provisoires), comme fetch_gas.
    """
    reg_cfg = config["gas_regional"]
    raw_dir = _raw_dir(config)
    date_field = reg_cfg["date_field"]
    request_timeout = reg_cfg.get("request_timeout", 60)

    start_year = dt.datetime.fromisoformat(
        config["date_range"]["start"].replace("Z", "+00:00")
    ).year
    end = config["date_range"]["end"]
    end_year = (
        dt.datetime.fromisoformat(end.replace("Z", "+00:00")).year
        if end
        else dt.datetime.now(dt.timezone.utc).year
    )
    current_year = dt.datetime.now(dt.timezone.utc).year

    datasets = {
        "industrial": reg_cfg["industrial_dataset_id"],
        "distribution": reg_cfg["distribution_dataset_id"],
    }
    for label, dataset_id in datasets.items():
        for year in range(start_year, end_year + 1):
            dest = raw_dir / f"{label}_{year}.csv"
            year_force = force or (refresh_current_year and year == current_year)
            where = f"{date_field} in [date'{year}-01-01'..date'{year}-12-31']"
            params = {"where": where, "timezone": "UTC"}
            logger.info("fetching regional gas %s year=%d (force=%s)", label, year, year_force)
            download(
                _export_url(config, dataset_id), dest,
                params=params, force=year_force, timeout=request_timeout,
            )


# ---------------------------------------------------------------------------
# Étape 2 : parsing wide (heure locale) -> long UTC, par région
# ---------------------------------------------------------------------------

def _hour_columns(df: pd.DataFrame) -> dict[str, int]:
    return {col: int(m.group(1)) for col in df.columns if (m := _HOUR_COL_RE.match(col))}


def _wide_to_regional_utc(df: pd.DataFrame, region_code_field: str) -> pd.DataFrame:
    """Convertit un dataframe LARGE (heure locale) en dataframe indexé par
    timestamp UTC, une colonne par `code_region`, valeur = somme sur opérateurs
    et secteurs. Les heures ambiguës/inexistantes (transitions DST) sont
    abandonnées plutôt que devinées.
    """
    hour_cols = _hour_columns(df)
    if not hour_cols or region_code_field not in df.columns:
        return pd.DataFrame()

    dates = pd.to_datetime(df["date"]).dt.normalize()
    codes = pd.to_numeric(df[region_code_field], errors="coerce")

    long_parts = []
    for col, hour in hour_cols.items():
        local_ts = dates + pd.to_timedelta(hour, unit="h")
        long_parts.append(
            pd.DataFrame({"local_ts": local_ts, "code_region": codes, "value": df[col]})
        )
    long_df = pd.concat(long_parts, ignore_index=True).dropna(subset=["value", "code_region"])
    long_df["code_region"] = long_df["code_region"].astype(int)

    # Somme sur opérateurs + secteurs, par (région, heure locale).
    grouped = long_df.groupby(["local_ts", "code_region"])["value"].sum().unstack("code_region")

    # Heure locale Europe/Paris -> UTC.
    localized_idx = grouped.index.tz_localize(
        CALENDAR_TZ, ambiguous="NaT", nonexistent="NaT"
    )
    grouped = grouped[localized_idx.notna()]
    grouped.index = localized_idx[localized_idx.notna()].tz_convert("UTC")
    return grouped.sort_index()


def _load_one_dataset(config: dict, label: str) -> pd.DataFrame:
    """Charge et concatène tous les CSV annuels cachés d'un dataset régional,
    retourne un dataframe UTC (index) x code_region (colonnes)."""
    reg_cfg = config["gas_regional"]
    raw_dir = _raw_dir(config)
    files = sorted(raw_dir.glob(f"{label}_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"no regional gas files '{label}_*.csv' in {raw_dir} — run fetch_regional_gas first"
        )

    per_year = []
    for f in files:
        df = pd.read_csv(f, sep=";", encoding="utf-8", encoding_errors="replace")
        wide = _wide_to_regional_utc(df, reg_cfg["region_code_field"])
        if not wide.empty:
            per_year.append(wide)

    if not per_year:
        return pd.DataFrame()
    combined = pd.concat(per_year)
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return combined


def _mask_isolated_dips(df: pd.DataFrame, dip_fraction: float) -> pd.DataFrame:
    """Remplace par NaN toute valeur < dip_fraction x min(voisin précédent,
    voisin suivant) : un effondrement d'une seule heure entouré de valeurs
    normales est physiquement impossible pour la conso gaz (inertie thermique),
    donc un artefact de la source (cf. glitch DST de mars). Ne touche pas les
    vrais creux (entourés d'heures également basses). Une valeur au bord (sans
    l'un des deux voisins) n'est jamais masquée (comparaison NaN -> False)."""
    prev = df.shift(1)
    nxt = df.shift(-1)
    neighbor_min = np.minimum(prev, nxt)   # NaN si un voisin manque
    dip = df < (dip_fraction * neighbor_min)
    n = int(dip.to_numpy().sum())
    if n:
        logger.info("regional gas: masked %d isolated-dip value(s) as NaN (DST-boundary artefact)", n)
    return df.mask(dip)


def load_regional_gas(config: dict) -> pd.DataFrame:
    """Cible régionale : consommation totale (industriel + distribution) par
    région, au pas horaire UTC. Retourne un DataFrame indexé par timestamp UTC,
    une colonne par `code_region` (int), en MW.

    industriel + distribution sont additionnés par (région, heure). Une valeur
    reste NaN si l'une des deux sources manque à cette heure pour cette région
    (pas de total partiel silencieux) — en pratique la distribution ne démarre
    qu'à `distribution_start`, donc les heures antérieures sont NaN partout.
    """
    industrial = _load_one_dataset(config, "industrial")
    distribution = _load_one_dataset(config, "distribution")

    # Union des index et des colonnes (régions), puis somme stricte : NaN si
    # l'un des deux manque (add sans fill_value propage le NaN).
    total = industrial.add(distribution)

    regions_cfg = {int(k): v for k, v in config["gas_regional"]["regions"].items()}
    known = [c for c in total.columns if c in regions_cfg]
    unexpected = [c for c in total.columns if c not in regions_cfg]
    if unexpected:
        logger.warning("regional gas: unexpected region codes ignored: %s", unexpected)
    total = total[sorted(known)]

    # Nettoyage : une conso gaz régionale <= 0 est impossible (trou de source),
    # puis filtre des chutes isolées d'une heure (artefact DST récurrent).
    total = total.mask(total <= 0)
    total = _mask_isolated_dips(total, float(config["gas_regional"].get("dip_fraction", 0.5)))

    n_regions = total.shape[1]
    logger.info(
        "regional gas: %d regions, %s -> %s (%d hourly points)",
        n_regions, total.index.min(), total.index.max(), len(total),
    )
    return total
