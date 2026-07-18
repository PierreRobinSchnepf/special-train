"""Ingestion brute (étape 1) : télécharge et met en cache localement les 4 sources.

Idempotent : toute ressource déjà présente dans data/raw/ est sautée, sauf
--force. L'année en cours des séries gaz est toujours retéléchargée (données
encore provisoires côté ODRÉ), sauf --no-refresh-current-year.

Usage :
    python fetch_data.py                      # tout
    python fetch_data.py --only gas meteo      # sous-ensemble
    python fetch_data.py --force               # ignore le cache
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys

from src.config import load_config, resolve_path
from src.http_utils import download, get_json
from src.regional_gas import fetch_regional_gas

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fetch_data")

SOURCES = ["gas", "gas_regional", "meteo", "holidays", "school_holidays"]


def fetch_gas(config: dict, force: bool = False, refresh_current_year: bool = True) -> None:
    gas_cfg = config["gas"]
    raw_dir = resolve_path(config["output"]["raw_dir"]) / "gas"

    start_year = dt.datetime.fromisoformat(config["date_range"]["start"].replace("Z", "+00:00")).year
    end = config["date_range"]["end"]
    end_year = (
        dt.datetime.fromisoformat(end.replace("Z", "+00:00")).year
        if end
        else dt.datetime.now(dt.timezone.utc).year
    )
    current_year = dt.datetime.now(dt.timezone.utc).year

    for year in range(start_year, end_year + 1):
        dest = raw_dir / f"gas_{year}.csv"
        year_force = force or (refresh_current_year and year == current_year)
        where = f"{gas_cfg['datetime_field']} in [date'{year}-01-01'..date'{year}-12-31']"
        params = {
            "where": where,
            "select": f"{gas_cfg['datetime_field']},{gas_cfg['value_field']}",
            "timezone": "UTC",
        }
        logger.info("fetching gas year=%d (force=%s)", year, year_force)
        download(gas_cfg["base_url"], dest, params=params, force=year_force)


def _list_meteo_resources(config: dict) -> dict:
    meteo_cfg = config["meteo"]
    dataset_id = meteo_cfg["datagouv_dataset_id"]
    url = f"https://www.data.gouv.fr/api/1/datasets/{dataset_id}/"
    logger.info("listing meteo resources for dataset %s", dataset_id)
    return get_json(url)


def fetch_meteo(config: dict, force: bool = False) -> None:
    meteo_cfg = config["meteo"]
    raw_dir = resolve_path(config["output"]["raw_dir"]) / "meteo"
    min_year = meteo_cfg["min_period_start_year"]

    catalog = _list_meteo_resources(config)
    resources = catalog.get("resources", [])

    departments = [s["department"] for s in meteo_cfg["stations"]]
    for dep in departments:
        prefix = f"HOR_departement_{dep}_periode_"
        matches = [r for r in resources if r.get("title", "").startswith(prefix)]
        kept = 0
        for r in matches:
            title = r["title"]
            period = title[len(prefix):]
            try:
                end_year = int(period.split("-")[-1])
            except ValueError:
                logger.warning("cannot parse period from title %r, skipping", title)
                continue
            if end_year < min_year:
                continue
            file_url = r["url"]
            filename = file_url.rsplit("/", 1)[-1]
            dest = raw_dir / filename
            download(file_url, dest, force=force)
            kept += 1
        if kept == 0:
            logger.warning("no meteo resources matched for department %s (prefix=%r)", dep, prefix)
        else:
            logger.info("department %s: %d period file(s) cached", dep, kept)


def fetch_holidays(config: dict, force: bool = False) -> None:
    dest = resolve_path(config["output"]["raw_dir"]) / "holidays" / "jours_feries_metropole.csv"
    download(config["holidays"]["url"], dest, force=force)


def fetch_school_holidays(config: dict, force: bool = False) -> None:
    sh_cfg = config["school_holidays"]
    dest = resolve_path(config["output"]["raw_dir"]) / "school_holidays" / "calendrier_scolaire.csv"

    if dest.exists() and dest.stat().st_size > 0 and not force:
        logger.info("skip (cache hit): %s", dest)
        return

    import csv

    page_size = sh_cfg["page_size"]
    start = 0
    all_records: list[dict] = []
    while True:
        params = {
            "dataset": sh_cfg["dataset_id"],
            "rows": page_size,
            "start": start,
        }
        payload = get_json(sh_cfg["api_url"], params=params)
        records = payload.get("records", [])
        if not records:
            break
        all_records.extend(records)
        logger.info("school_holidays: fetched %d records (start=%d)", len(records), start)
        start += page_size
        if len(records) < page_size:
            break

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".csv.part")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["location", "zones", "annee_scolaire", "description", "start_date", "end_date"])
        for rec in all_records:
            fields = rec.get("fields", {})
            writer.writerow(
                [
                    fields.get("location", ""),
                    fields.get("zones", ""),
                    fields.get("annee_scolaire", ""),
                    fields.get("description", ""),
                    fields.get("start_date", ""),
                    fields.get("end_date", ""),
                ]
            )
    tmp.replace(dest)
    logger.info("school_holidays: wrote %d rows to %s", len(all_records), dest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=SOURCES, default=SOURCES)
    parser.add_argument("--force", action="store_true", help="ignore le cache local")
    parser.add_argument(
        "--no-refresh-current-year",
        action="store_true",
        help="ne pas retélécharger systématiquement l'année gaz en cours",
    )
    args = parser.parse_args(argv)

    config = load_config()

    if "gas" in args.only:
        fetch_gas(config, force=args.force, refresh_current_year=not args.no_refresh_current_year)
    if "gas_regional" in args.only:
        fetch_regional_gas(config, force=args.force, refresh_current_year=not args.no_refresh_current_year)
    if "meteo" in args.only:
        fetch_meteo(config, force=args.force)
    if "holidays" in args.only:
        fetch_holidays(config, force=args.force)
    if "school_holidays" in args.only:
        fetch_school_holidays(config, force=args.force)

    logger.info("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
