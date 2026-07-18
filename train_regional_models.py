"""Entraîne et persiste les modèles de prévision gaz PAR RÉGION (Étape C/D).

Un jeu de modèles par région (12 régions gazières) et par usage, réutilisant
sans les modifier `HourlySUREModel` et `HourlyKalmanSURModel`. L'OLS est
volontairement exclu (jugé sans valeur ajoutée vs SURE — cf. config.yaml §
regional_models.models).

Deux jeux (comme au national, cf. train_models.py) :
  - "backtest"   : hold-out réservé [test_start, test_end). Sert au dashboard de
    rejeu et aux métriques. Le Kalman avance son état sur le test.
  - "production" : entraîné sur toute la fenêtre propre (train_start -> dernière
    donnée), sans hold-out. Sert à la prévision réelle.

Artefacts : data/models/regional/region<code>_<model>_<set>.pkl
Métriques (jeu backtest) : data/processed/regional/metrics_regional.json

Usage :
    python train_regional_models.py                       # 12 régions, 2 jeux
    python train_regional_models.py --set production        # un seul jeu
    python train_regional_models.py --only 11 75             # sous-ensemble
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import pandas as pd

from models.dataset import build_hourly_equations, load_dataset, split_train_test, target_series
from models.kalman import HourlyKalmanSURModel
from models.metrics import combine_hourly, evaluate, evaluate_overall
from models.persistence import ARTIFACTS_DIR, REGIONAL_SUBDIR, regional_artifact_name, save_artifact
from models.sure import HourlySUREModel
from src.config import load_config, resolve_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("train_regional_models")

# Fenêtre "production" : test vide -> le train couvre tout ce qui suit
# train_start (jusqu'à la dernière donnée du dataset).
_FAR_FUTURE = "2100-01-01"


def _region_codes(config: dict) -> list[int]:
    return sorted(int(k) for k in config["gas_regional"]["regions"])


def _dataset_path(config: dict, code: int):
    subdir = config["regional_models"]["dataset_subdir"]
    return resolve_path(config["output"]["processed_dir"]) / subdir / f"dataset_region_{code}.parquet"


def _fit_sure_kalman(train):
    """Entraîne SURE (statique) + Kalman (dynamique) sur un panel train."""
    sure = HourlySUREModel().fit(train)
    kalman = HourlyKalmanSURModel().fit(train)
    return sure, kalman


def _overall_with_bias(y_true: dict, y_pred: dict) -> dict:
    """evaluate_overall + biais net signé (%) : positif = sur-prévision
    (prévu > réel), négatif = sous-prévision. Sert à colorer la carte régionale."""
    m = evaluate_overall(y_true, y_pred)
    yt = combine_hourly(y_true)
    yp = combine_hourly(y_pred)
    yt, yp = yt.align(yp, join="inner")
    total = yt.to_numpy().sum()
    m["bias_pct"] = float(100 * (yp.to_numpy().sum() - total) / total) if total else 0.0
    return m


def train_region_backtest(config: dict, code: int, label: str) -> dict:
    """Jeu backtest : hold-out réservé, métriques + persistance."""
    rm = config["regional_models"]
    df = load_dataset(_dataset_path(config, code))
    per_hour = build_hourly_equations(df)
    train, test = split_train_test(
        per_hour, test_start=rm["test_start"], test_end=rm["test_end"], train_start=rm["train_start"]
    )
    y_train, y_test = target_series(train), target_series(test)

    t0 = time.time()
    sure, kalman = _fit_sure_kalman(train)
    sure_pred_test = sure.predict(test)
    sur_frozen_test, kalman_pred_test = kalman.predict(test)  # avance l'état sur le test
    logger.info("region %s (%s) backtest: SURE+Kalman en %.1fs", code, label, time.time() - t0)

    save_artifact(sure, regional_artifact_name(code, "sure", "backtest"))
    save_artifact(kalman, regional_artifact_name(code, "kalman", "backtest"))

    metrics = {
        "region_label": label,
        "n_train_days": int(len(train[0])),
        "n_test_days": int(len(test[0])),
        "sure": {"test": _overall_with_bias(y_test, sure_pred_test),
                 "train": evaluate_overall(y_train, sure.predict(train))},
        "kalman": {"test": _overall_with_bias(y_test, kalman_pred_test),
                   "test_sur_frozen_baseline": evaluate_overall(y_test, sur_frozen_test)},
    }
    return {
        "metrics": metrics,
        "y_test": combine_hourly(y_test),
        "sure_test": combine_hourly(sure_pred_test),
        "kalman_test": combine_hourly(kalman_pred_test),
    }


def train_region_production(config: dict, code: int, label: str) -> None:
    """Jeu production : entraîné jusqu'à la dernière donnée, sans hold-out."""
    rm = config["regional_models"]
    df = load_dataset(_dataset_path(config, code))
    per_hour = build_hourly_equations(df)
    # test vide -> train = tout ce qui suit train_start
    train, _ = split_train_test(per_hour, _FAR_FUTURE, _FAR_FUTURE, train_start=rm["train_start"])

    t0 = time.time()
    sure, kalman = _fit_sure_kalman(train)
    logger.info("region %s (%s) production: SURE+Kalman en %.1fs", code, label, time.time() - t0)

    save_artifact(sure, regional_artifact_name(code, "sure", "production"))
    save_artifact(kalman, regional_artifact_name(code, "kalman", "production"))


def _national_sanity_check(per_region: dict[int, dict]) -> dict:
    """Somme les prévisions test des 12 régions et compare à la somme des
    vérités régionales — les modèles régionaux doivent collectivement
    reconstituer le total national."""
    def _sum(key: str) -> pd.Series:
        return pd.concat([r[key] for r in per_region.values()], axis=1).sum(axis=1, min_count=len(per_region))

    y = _sum("y_test")
    return {
        "sure_sum_vs_truth_sum": evaluate(y, _sum("sure_test")),
        "kalman_sum_vs_truth_sum": evaluate(y, _sum("kalman_test")),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", type=int, default=None, help="codes région à entraîner")
    parser.add_argument("--set", choices=["backtest", "production", "both"], default="both")
    args = parser.parse_args(argv)

    config = load_config()
    codes = args.only if args.only else _region_codes(config)
    names = {int(k): v for k, v in config["gas_regional"]["regions"].items()}
    (ARTIFACTS_DIR / REGIONAL_SUBDIR).mkdir(parents=True, exist_ok=True)

    per_region: dict[int, dict] = {}
    for code in codes:
        logger.info("=== Région %s (%s) ===", code, names[code])
        if args.set in ("backtest", "both"):
            per_region[code] = train_region_backtest(config, code, names[code])
        if args.set in ("production", "both"):
            train_region_production(config, code, names[code])

    if per_region:
        all_metrics = {str(c): r["metrics"] for c, r in per_region.items()}
        if len(per_region) == len(_region_codes(config)):
            all_metrics["_national_sanity_check"] = _national_sanity_check(per_region)

        out_dir = resolve_path(config["output"]["processed_dir"]) / config["regional_models"]["dataset_subdir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics_regional.json").write_text(
            json.dumps(all_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        print("\n=== Resume test (RMSE MW / MAPE %) par region ===")
        print(f"{'reg':>4} {'label':<26} {'SURE rmse':>10} {'SURE mape':>10} {'KAL rmse':>10} {'KAL mape':>10}")
        for c in sorted(per_region):
            m = per_region[c]["metrics"]
            s, k = m["sure"]["test"], m["kalman"]["test"]
            print(f"{c:>4} {m['region_label'][:26]:<26} {s['rmse']:>10.1f} {s['mape']:>10.2f} {k['rmse']:>10.1f} {k['mape']:>10.2f}")
        if "_national_sanity_check" in all_metrics:
            sc = all_metrics["_national_sanity_check"]
            print(f"\nSanity national (sum of regions): SURE MAPE={sc['sure_sum_vs_truth_sum']['mape']:.2f}%  "
                  f"Kalman MAPE={sc['kalman_sum_vs_truth_sum']['mape']:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
