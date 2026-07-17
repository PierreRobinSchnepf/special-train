"""Point d'entrée du "job de 17h" : exécute le pipeline réel et enregistre
son résultat. Pensé pour être appelé manuellement ou via un ordonnanceur
(cron/tâche planifiée) — cette configuration système n'est pas mise en
place ici (action d'infra qui dépasse le périmètre de ce dépôt).

Usage :
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
    logger.info("Démarrage du pipeline réel...")
    result = run_real_forecast()

    for w in result.warnings:
        logger.warning(w)

    run_id = save_forecast(result)
    logger.info(
        "Prévision enregistrée (run_id=%s) : jour J=%s, jour G (dernière vérité terrain)=%s, %d points.",
        run_id, result.day_j, result.day_g, len(result.horizon),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
