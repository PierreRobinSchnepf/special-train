"""Train and persist the PER-REGION gas forecasting models (Step C/D).

One model set per region (12 gas regions) and per usage, reusing
`HourlySUREModel` and `HourlyKalmanSURModel` unmodified. OLS is deliberately
excluded (judged to add no value over SURE — see config.yaml §
regional_models.models).

Two sets (like the national ones, see train_models.py):
  - "backtest"   : held-out test [test_start, test_end). Feeds the replay
    dashboard and the metrics. The Kalman advances its state over the test.
  - "production" : trained over the whole clean window (train_start -> latest
    data), no hold-out. Feeds the live forecast.

Artifacts: data/models/regional/region<code>_<model>_<set>.pkl
Metrics (backtest set): data/processed/regional/metrics_regional.json

Usage:
    python scripts/train_regional_models.py                       # 12 regions, 2 sets
    python scripts/train_regional_models.py --set production        # a single set
    python scripts/train_regional_models.py --only 11 75             # subset
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from models.dataset import build_hourly_equations, load_dataset, split_train_test, target_series
from models.kalman import HourlyKalmanSURModel
from models.metrics import combine_hourly, evaluate, evaluate_overall
from models.persistence import ARTIFACTS_DIR, REGIONAL_SUBDIR, regional_artifact_name, save_artifact
from models.sure import HourlySUREModel
from src.config import load_config, resolve_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("train_regional_models")

# "Production" window: empty test -> the train set covers everything after
# train_start (through the dataset's latest data).
_FAR_FUTURE = "2100-01-01"


def _region_codes(config: dict) -> list[int]:
    return sorted(int(k) for k in config["gas_regional"]["regions"])


def _dataset_path(config: dict, code: int):
    subdir = config["regional_models"]["dataset_subdir"]
    return resolve_path(config["output"]["processed_dir"]) / subdir / f"dataset_region_{code}.parquet"


def _fit_sure_kalman(train):
    """Train SURE (static) + Kalman (dynamic) on a training panel."""
    sure = HourlySUREModel().fit(train)
    kalman = HourlyKalmanSURModel().fit(train)
    return sure, kalman


def _overall_with_bias(y_true: dict, y_pred: dict) -> dict:
    """evaluate_overall + net signed bias (%): positive = over-forecast
    (predicted > actual), negative = under-forecast. Colors the regional map."""
    m = evaluate_overall(y_true, y_pred)
    yt = combine_hourly(y_true)
    yp = combine_hourly(y_pred)
    yt, yp = yt.align(yp, join="inner")
    total = yt.to_numpy().sum()
    m["bias_pct"] = float(100 * (yp.to_numpy().sum() - total) / total) if total else 0.0
    return m


def train_region_backtest(config: dict, code: int, label: str) -> dict:
    """Backtest set: held-out test, metrics + persistence."""
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
    sur_frozen_test, kalman_pred_test = kalman.predict(test)  # advance the state over the test
    logger.info("region %s (%s) backtest: SURE+Kalman in %.1fs", code, label, time.time() - t0)

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
    """Production set: trained through the latest data, no hold-out."""
    rm = config["regional_models"]
    df = load_dataset(_dataset_path(config, code))
    per_hour = build_hourly_equations(df)
    # empty test -> train = everything after train_start
    train, _ = split_train_test(per_hour, _FAR_FUTURE, _FAR_FUTURE, train_start=rm["train_start"])

    t0 = time.time()
    sure, kalman = _fit_sure_kalman(train)
    logger.info("region %s (%s) production: SURE+Kalman in %.1fs", code, label, time.time() - t0)

    save_artifact(sure, regional_artifact_name(code, "sure", "production"))
    save_artifact(kalman, regional_artifact_name(code, "kalman", "production"))


def _national_sanity_check(per_region: dict[int, dict]) -> dict:
    """Sum the 12 regions' test forecasts and compare with the sum of the
    regional truths — the regional models must collectively reconstruct the
    national total."""
    def _sum(key: str) -> pd.Series:
        return pd.concat([r[key] for r in per_region.values()], axis=1).sum(axis=1, min_count=len(per_region))

    y = _sum("y_test")
    return {
        "sure_sum_vs_truth_sum": evaluate(y, _sum("sure_test")),
        "kalman_sum_vs_truth_sum": evaluate(y, _sum("kalman_test")),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", type=int, default=None, help="region codes to train")
    parser.add_argument("--set", choices=["backtest", "production", "both"], default="both")
    args = parser.parse_args(argv)

    config = load_config()
    codes = args.only if args.only else _region_codes(config)
    names = {int(k): v for k, v in config["gas_regional"]["regions"].items()}
    (ARTIFACTS_DIR / REGIONAL_SUBDIR).mkdir(parents=True, exist_ok=True)

    per_region: dict[int, dict] = {}
    for code in codes:
        logger.info("=== Region %s (%s) ===", code, names[code])
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

        print("\n=== Test summary (RMSE MW / MAPE %) per region ===")
        print(f"{'reg':>4} {'label':<26} {'SURE rmse':>10} {'SURE mape':>10} {'KAL rmse':>10} {'KAL mape':>10}")
        for c in sorted(per_region):
            m = per_region[c]["metrics"]
            s, k = m["sure"]["test"], m["kalman"]["test"]
            print(f"{c:>4} {m['region_label'][:26]:<26} {s['rmse']:>10.1f} {s['mape']:>10.2f} {k['rmse']:>10.1f} {k['mape']:>10.2f}")
        if "_national_sanity_check" in all_metrics:
            sc = all_metrics["_national_sanity_check"]
            print(f"\nNational sanity check (sum of regions): SURE MAPE={sc['sure_sum_vs_truth_sum']['mape']:.2f}%  "
                  f"Kalman MAPE={sc['kalman_sum_vs_truth_sum']['mape']:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
