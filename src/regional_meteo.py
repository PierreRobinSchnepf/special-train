"""Température PAR RÉGION (Étape B du modèle région par région).

Réutilise exactement la même mécanique que la température nationale
(`build_dataset.load_per_department_temp` + `renormalized_weighted_mean`) :
sélection automatique du poste le mieux couvert par département, comblement des
trous courts, puis moyenne pondérée population. La seule différence est le
regroupement : au lieu d'agréger les 16 départements en une seule série
nationale, on les regroupe par `region_code` (INSEE) pour produire une
température par région, alignée sur les 12 régions gazières.

Chaque région gazière est couverte par >= 1 département du panier météo (cf.
config.yaml § meteo.stations, `region_code`). Les départements dont le
`region_code` ne correspond à aucune des 12 régions gazières sont ignorés (il
n'y en a pas aujourd'hui, mais la fonction reste robuste si le panier évolue).
"""
from __future__ import annotations

import logging

import pandas as pd

from build_dataset import load_per_department_temp, renormalized_weighted_mean

logger = logging.getLogger(__name__)


def load_meteo_regional(config: dict) -> pd.DataFrame:
    """Température horaire par région, indexée UTC. Retourne un DataFrame
    (index UTC) x (code_region int), en °C. Chaque colonne est la moyenne
    pondérée population des stations de la région, avec renormalisation des
    poids sur les stations disponibles à chaque heure.
    """
    meteo_cfg = config["meteo"]
    per_dept_series, weights = load_per_department_temp(config)

    # dep -> region_code, depuis la config (source unique de vérité).
    dep_to_region: dict[str, int] = {}
    for station_cfg in meteo_cfg["stations"]:
        code = station_cfg.get("region_code")
        if code is not None:
            dep_to_region[station_cfg["department"]] = int(code)

    # Régions gazières attendues (la cible régionale ne couvre qu'elles).
    gas_regions = {int(k) for k in config["gas_regional"]["regions"]}

    wide = pd.DataFrame(per_dept_series)
    max_gap = meteo_cfg["max_ffill_gap_hours"]
    wide_filled = wide.ffill(limit=max_gap)

    region_series: dict[int, pd.Series] = {}
    for region_code in sorted(gas_regions):
        deps = [d for d, r in dep_to_region.items() if r == region_code and d in wide_filled.columns]
        if not deps:
            logger.warning("region %s has no meteo station, temperature will be all-NaN", region_code)
            continue
        sub = wide_filled[deps]
        sub_weights = {d: weights[d] for d in deps}
        region_series[region_code] = renormalized_weighted_mean(sub, sub_weights)
        logger.info(
            "region %s: temperature from %d station(s) %s", region_code, len(deps), deps
        )

    regional_temp = pd.DataFrame(region_series)
    missing = gas_regions - set(regional_temp.columns)
    if missing:
        logger.warning("regions without any temperature series: %s", sorted(missing))
    return regional_temp
