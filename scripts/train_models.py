"""Train and persist the benchmark models, so the dashboard and the live
pipeline no longer need to retrain (~40s) on every launch.

Two artifact sets (see `models/persistence.py` for the training-window
details):
  - "backtest"   : OLS/SURE on [2020, 2025), Kalman on [2018, 2025) + state
    advanced over the 2025 test — used by the Forecast/Benchmark/Monitoring tabs.
  - "production" : same lower bounds, but trained through the end of 2025
    (no held-out test) — used by the Live pipeline tab.

Usage:
    python scripts/train_models.py            # (re)generate both sets
    python scripts/train_models.py --only backtest
    python scripts/train_models.py --only production
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    logger.info("%s trained in %.1fs", label, time.time() - t0)
    return result


def train_backtest(per_hour) -> None:
    logger.info("=== 'backtest' set (2025 test held out) ===")
    ols_train, _ = split_train_test(per_hour, BACKTEST_TEST_START, BACKTEST_TEST_END, train_start=OLS_TRAIN_START)
    sure_train, _ = split_train_test(per_hour, BACKTEST_TEST_START, BACKTEST_TEST_END, train_start=SURE_TRAIN_START)
    kalman_train, kalman_test = split_train_test(per_hour, BACKTEST_TEST_START, BACKTEST_TEST_END, train_start=KALMAN_TRAIN_START)

    ols = _timed_fit("OLS (backtest, train >= 2020)", lambda: HourlyOLSModel().fit(ols_train))
    sure = _timed_fit("SURE (backtest, train >= 2020)", lambda: HourlySUREModel().fit(sure_train))
    kalman = _timed_fit("Kalman (backtest, train >= 2018)", lambda: HourlyKalmanSURModel().fit(kalman_train))
    kalman.predict(kalman_test)  # advance the state over the whole 2025 test

    save_artifact(ols, "backtest_ols")
    save_artifact(sure, "backtest_sure")
    save_artifact(kalman, "backtest_kalman")
    logger.info("'backtest' artifacts saved to data/models/.")


def train_production(per_hour) -> None:
    logger.info("=== 'production' set (trained through end of 2025) ===")
    # test_start == test_end == BACKTEST_TEST_END: empty test, train = all
    # data before 2026-01-01, so 2025 is included in the train set.
    ols_train, _ = split_train_test(per_hour, BACKTEST_TEST_END, BACKTEST_TEST_END, train_start=OLS_TRAIN_START)
    sure_train, _ = split_train_test(per_hour, BACKTEST_TEST_END, BACKTEST_TEST_END, train_start=SURE_TRAIN_START)
    kalman_train, _ = split_train_test(per_hour, BACKTEST_TEST_END, BACKTEST_TEST_END, train_start=KALMAN_TRAIN_START)

    ols = _timed_fit("OLS (production, train >= 2020, through end of 2025)", lambda: HourlyOLSModel().fit(ols_train))
    sure = _timed_fit("SURE (production, train >= 2020, through end of 2025)", lambda: HourlySUREModel().fit(sure_train))
    kalman = _timed_fit("Kalman (production, train >= 2018, through end of 2025)", lambda: HourlyKalmanSURModel().fit(kalman_train))

    save_artifact(ols, "production_ols")
    save_artifact(sure, "production_sure")
    save_artifact(kalman, "production_kalman")
    logger.info("'production' artifacts saved to data/models/.")


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
