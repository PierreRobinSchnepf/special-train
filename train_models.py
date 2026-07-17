"""Entraîne et persiste les modèles de benchmark, pour que le dashboard et le
pipeline réel n'aient plus à ré-entraîner (~40s) à chaque lancement.

Deux jeux d'artefacts (voir `models/persistence.py` pour le détail des
fenêtres d'entraînement) :
  - "backtest"   : OLS/SURE sur [2020, 2025), Kalman sur [2018, 2025) + état
    avancé sur le test 2025 — utilisé par les onglets Forecast/Benchmark/Suivi.
  - "production" : mêmes bornes basses, mais entraînés jusqu'à fin 2025
    inclus (pas de test réservé) — utilisé par l'onglet Pipeline réel.

Usage :
    python train_models.py            # (re)génère les deux jeux
    python train_models.py --only backtest
    python train_models.py --only production
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

from models.dataset import build_hourly_equations, load_dataset, split_train_test
from models.kalman import HourlyKalmanSURModel
from models.ols import HourlyOLSModel
from models.persistence import (
    BACKTEST_TEST_END,
    BACKTEST_TEST_START,
    KALMAN_TRAIN_START,
    OLS_TRAIN_START,
    SURE_TRAIN_START,
    save_artifact,
)
from models.sure import HourlySUREModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("train_models")


def _timed_fit(label: str, fn):
    t0 = time.time()
    result = fn()
    logger.info("%s entraîné en %.1fs", label, time.time() - t0)
    return result


def train_backtest(per_hour) -> None:
    logger.info("=== Jeu 'backtest' (test 2025 réservé) ===")
    ols_train, _ = split_train_test(per_hour, BACKTEST_TEST_START, BACKTEST_TEST_END, train_start=OLS_TRAIN_START)
    sure_train, _ = split_train_test(per_hour, BACKTEST_TEST_START, BACKTEST_TEST_END, train_start=SURE_TRAIN_START)
    kalman_train, kalman_test = split_train_test(per_hour, BACKTEST_TEST_START, BACKTEST_TEST_END, train_start=KALMAN_TRAIN_START)

    ols = _timed_fit("OLS (backtest, train >= 2020)", lambda: HourlyOLSModel().fit(ols_train))
    sure = _timed_fit("SURE (backtest, train >= 2020)", lambda: HourlySUREModel().fit(sure_train))
    kalman = _timed_fit("Kalman (backtest, train >= 2018)", lambda: HourlyKalmanSURModel().fit(kalman_train))
    kalman.predict(kalman_test)  # avance l'état sur tout le test 2025

    save_artifact(ols, "backtest_ols")
    save_artifact(sure, "backtest_sure")
    save_artifact(kalman, "backtest_kalman")
    logger.info("Artefacts 'backtest' sauvegardés dans data/models/.")


def train_production(per_hour) -> None:
    logger.info("=== Jeu 'production' (entraîné jusqu'à fin 2025 inclus) ===")
    # test_start == test_end == BACKTEST_TEST_END : test vide, train = tout
    # ce qui précède le 01/01/2026, donc 2025 inclus dans le train.
    ols_train, _ = split_train_test(per_hour, BACKTEST_TEST_END, BACKTEST_TEST_END, train_start=OLS_TRAIN_START)
    sure_train, _ = split_train_test(per_hour, BACKTEST_TEST_END, BACKTEST_TEST_END, train_start=SURE_TRAIN_START)
    kalman_train, _ = split_train_test(per_hour, BACKTEST_TEST_END, BACKTEST_TEST_END, train_start=KALMAN_TRAIN_START)

    ols = _timed_fit("OLS (production, train >= 2020, jusqu'à fin 2025)", lambda: HourlyOLSModel().fit(ols_train))
    sure = _timed_fit("SURE (production, train >= 2020, jusqu'à fin 2025)", lambda: HourlySUREModel().fit(sure_train))
    kalman = _timed_fit("Kalman (production, train >= 2018, jusqu'à fin 2025)", lambda: HourlyKalmanSURModel().fit(kalman_train))

    save_artifact(ols, "production_ols")
    save_artifact(sure, "production_sure")
    save_artifact(kalman, "production_kalman")
    logger.info("Artefacts 'production' sauvegardés dans data/models/.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["backtest", "production"], default=None)
    args = parser.parse_args(argv)

    df = load_dataset()
    per_hour = build_hourly_equations(df)

    if args.only in (None, "backtest"):
        train_backtest(per_hour)
    if args.only in (None, "production"):
        train_production(per_hour)

    return 0


if __name__ == "__main__":
    sys.exit(main())
