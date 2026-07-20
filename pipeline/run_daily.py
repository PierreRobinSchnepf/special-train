"""Entry point of the "5 PM job": run the live pipeline and record its
result. Meant to be called manually or by a scheduler (cron/scheduled task) —
that system configuration is not set up here (an infra action beyond this
repository's scope).

Usage:
    python -m pipeline.run_daily
"""
from __future__ import annotations

import logging
import sys

from pipeline.real_forecast import run_real_forecast
from pipeline.tracking_store import save_forecast

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline.run_daily")


def main() -> int:
    logger.info("Starting the live pipeline...")
    result = run_real_forecast()

    for w in result.warnings:
        logger.warning(w)

    run_id = save_forecast(result)
    logger.info(
        "Forecast recorded (run_id=%s): day J=%s, day G (last ground truth)=%s, %d points.",
        run_id, result.day_j, result.day_g, len(result.horizon),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
