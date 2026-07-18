"""Charge (ou entraîne+sauvegarde une fois) les modèles et expose les méthodes
de lookup du dashboard : prévision d'une heure d'un jour J avec la bonne règle
"quelles données étaient déjà connues au moment de la prévision".

Une seule classe `ModelStore` sert deux périmètres :
  - **National** (`region_code=None`) : dataset_final + 3 modèles (Kalman, OLS,
    SURE), jeu backtest, test 2025. Comportement historique inchangé.
  - **Régional** (`region_code=<int>`) : dataset_region_<code> + 2 modèles
    (Kalman, SURE — l'OLS est exclu au régional), jeu backtest, fenêtre de test
    définie dans config.yaml § regional_models.

Le dashboard itère sur `store.models` (liste ordonnée (clé, libellé)) plutôt que
de coder en dur les 3 modèles : ajouter/retirer un modèle ne touche qu'ici.

Règle d'assimilation (cf. spec : à J 17h, on prévoit J[17h-23h] + J+1[0h-23h]) :
pour une heure cible `hour`, la dernière occurrence réellement connue à "J 17h"
est aujourd'hui (J) si hour <= 16, sinon hier (J-1).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from models.dataset import (
    PREDICTOR_COLUMNS,
    TARGET_COLUMN,
    build_hourly_equations,
    load_dataset,
    split_train_test,
)
from models.kalman import HourlyKalmanSURModel
from models.ols import HourlyOLSModel
from models.persistence import (
    KALMAN_TRAIN_START,
    OLS_TRAIN_START,
    SURE_TRAIN_START,
    load_artifact,
    regional_artifact_name,
    save_artifact,
)
from models.sure import HourlySUREModel
from src.config import load_config, resolve_path

NATIONAL_TEST_START = "2025-01-01"
NATIONAL_TEST_END = "2026-01-01"

# Libellés des modèles (l'ordre définit l'ordre d'affichage ; le premier est
# "notre prédiction" mise en avant).
_LABELS = {
    "kalman": "Kalman (notre prédiction)",
    "ols": "OLS (statique)",
    "sure": "SURE (statique)",
}

# Regroupement des prédicteurs par bloc du Tableau 1 (décomposition explicable).
BLOCKS: dict[str, list[str]] = {
    "Thermique": ["temp_smo", "X1_heating", "X2_smo_heating"],
    "Saisonnier (Fourier)": [
        f"{trig}{s}_{grp}" for trig in ("cos", "sin") for s in (1, 2, 3, 4) for grp in ("WD", "WE")
    ],
    "Calendaire": ["is_monday", "is_friday", "is_saturday", "is_sunday", "is_end_of_year", "is_off_peak_period"],
}


def _assimilated_date(day_j: dt.date, hour: int) -> dt.date:
    return day_j if hour <= 16 else day_j - dt.timedelta(days=1)


@dataclass
class ModelPred:
    value: float
    lo: float
    hi: float


@dataclass
class HourPrediction:
    date: dt.date
    hour: int
    actual: float | None
    preds: dict[str, ModelPred]          # clé modèle -> (valeur, IC bas, IC haut)
    decomposition_log: dict[str, float] = field(default_factory=dict)


class ModelStore:
    def __init__(self, region_code: int | None = None) -> None:
        self.region_code = region_code
        config = load_config()

        if region_code is None:
            # --- périmètre national ---
            self.label = "National"
            self.model_keys = ["kalman", "ols", "sure"]
            self.test_start, self.test_end = NATIONAL_TEST_START, NATIONAL_TEST_END
            self.df = load_dataset()
            self._artifact = lambda key: f"backtest_{key}"
            self._train_start = {"ols": OLS_TRAIN_START, "sure": SURE_TRAIN_START, "kalman": KALMAN_TRAIN_START}
        else:
            # --- périmètre régional (OLS exclu) ---
            rm = config["regional_models"]
            names = {int(k): v for k, v in config["gas_regional"]["regions"].items()}
            self.label = f"{region_code} — {names[region_code]}"
            self.model_keys = ["kalman", "sure"]
            self.test_start, self.test_end = rm["test_start"], rm["test_end"]
            ds = resolve_path(config["output"]["processed_dir"]) / rm["dataset_subdir"] / f"dataset_region_{region_code}.parquet"
            self.df = load_dataset(ds)
            self._artifact = lambda key: regional_artifact_name(region_code, key, "backtest")
            ts = rm["train_start"]
            self._train_start = {"sure": ts, "kalman": ts}

        self.models = [(k, _LABELS[k]) for k in self.model_keys]
        self.has_ols = "ols" in self.model_keys

        self.per_hour_all = build_hourly_equations(self.df)
        self.train, self.test = split_train_test(self.per_hour_all, test_start=self.test_start, test_end=self.test_end)
        self.full_per_hour = {h: pd.concat([self.train[h], self.test[h]]).sort_index() for h in range(24)}

        self._load_or_train()

        # Attributs de commodité pour les IC (variance résiduelle en niveau).
        self._ols_mse = self.ols.mse_resid_ if self.has_ols else None
        self._sure_mse = self.sure.stage1_resid_var_
        self.state_cols = self.kalman.state_cols

    # ------------------------------------------------------------------
    def _load_or_train(self) -> None:
        """Charge les artefacts ; entraîne+sauvegarde uniquement ceux qui manquent."""
        self.ols = self.sure = self.kalman = None

        if self.has_ols:
            self.ols = load_artifact(self._artifact("ols"))
            if self.ols is None:
                print(f"[model_store {self.label}] entraînement OLS + sauvegarde...")
                train, _ = split_train_test(self.per_hour_all, self.test_start, self.test_end, train_start=self._train_start["ols"])
                self.ols = HourlyOLSModel().fit(train)
                save_artifact(self.ols, self._artifact("ols"))

        self.sure = load_artifact(self._artifact("sure"))
        if self.sure is None:
            print(f"[model_store {self.label}] entraînement SURE + sauvegarde...")
            train, _ = split_train_test(self.per_hour_all, self.test_start, self.test_end, train_start=self._train_start["sure"])
            self.sure = HourlySUREModel().fit(train)
            save_artifact(self.sure, self._artifact("sure"))

        self.kalman = load_artifact(self._artifact("kalman"))
        if self.kalman is None:
            print(f"[model_store {self.label}] entraînement Kalman + sauvegarde...")
            train, test = split_train_test(self.per_hour_all, self.test_start, self.test_end, train_start=self._train_start["kalman"])
            self.kalman = HourlyKalmanSURModel().fit(train)
            self.kalman.predict(test)  # avance l'état sur le test
            save_artifact(self.kalman, self._artifact("kalman"))
        print(f"[model_store {self.label}] ready.")

    # ------------------------------------------------------------------
    def selectable_days(self) -> list[str]:
        """Jours J sélectionnables : jours de la fenêtre de test présents dans le
        panel, en laissant un jour de marge pour que J+1 reste couvert."""
        test_dates = sorted(self.test[0].index)
        if not test_dates:
            return []
        return [d.isoformat() for d in test_dates[:-1]]

    def _predictor_row(self, hour: int, date: dt.date) -> pd.Series | None:
        frame = self.full_per_hour[hour]
        if date not in frame.index:
            return None
        row = frame.loc[date]
        if isinstance(row, pd.DataFrame):  # garde-fou (panel dédupliqué)
            row = row.iloc[0]
        return row

    def _apply_what_if(self, row: pd.Series, temp_delta: float) -> pd.Series:
        """Décale la température prévue de `temp_delta` °C (approximation : décale
        directement les 3 variables thermiques, sans recalcul de l'EWMA)."""
        if temp_delta == 0.0:
            return row
        row = row.copy()
        row["temp_smo"] = row["temp_smo"] + temp_delta
        row["X1_heating"] = max(0.0, row["X1_heating"] - temp_delta)
        row["X2_smo_heating"] = max(0.0, row["X2_smo_heating"] - temp_delta)
        return row

    # ------------------------------------------------------------------
    def _static_pred(self, beta: np.ndarray, mse_h: float, x_all: np.ndarray) -> ModelPred:
        pred = float(x_all @ beta)
        se = float(np.sqrt(mse_h))
        return ModelPred(pred, pred - 1.96 * se, pred + 1.96 * se)

    def _kalman_pred(self, row: pd.Series, hour: int, day_j: dt.date) -> tuple[ModelPred, dict[str, float]]:
        assim_date = _assimilated_date(day_j, hour)
        traj = self.kalman.full_beta_trajectory(hour)
        lookup_date = min(max(assim_date, traj.index.min()), traj.index.max())
        state = traj.loc[lookup_date].to_numpy(dtype=float)

        x_state = row[self.state_cols].to_numpy(dtype=float)
        contrib = x_state * self.kalman.sur_beta_[hour]     # contribution structurelle SUR (log)
        pred_log = self.kalman.intercept_[hour] + float(contrib @ state)
        kalman_pred = float(np.expm1(pred_log))

        P = self.kalman.P_state_[hour]
        var_log = float(contrib @ P @ contrib.T + self.kalman.V_[hour])
        se_log = np.sqrt(max(var_log, 0.0))
        pred = ModelPred(
            kalman_pred,
            float(np.expm1(pred_log - 1.96 * se_log)),
            float(np.expm1(pred_log + 1.96 * se_log)),
        )

        decomposition_log = {"Fond de roulement (beta0)": self.kalman.intercept_[hour]}
        for block_name, cols in BLOCKS.items():
            idx = [self.state_cols.index(c) for c in cols]
            decomposition_log[block_name] = float(np.sum(contrib[idx] * state[idx]))
        return pred, decomposition_log

    def predict_hour(
        self, day_j: dt.date, target_date: dt.date, hour: int, temp_delta: float = 0.0
    ) -> HourPrediction | None:
        row = self._predictor_row(hour, target_date)
        if row is None:
            return None
        row = self._apply_what_if(row, temp_delta)
        x_all = row[PREDICTOR_COLUMNS].to_numpy(dtype=float)

        preds: dict[str, ModelPred] = {}
        kalman_pred, decomposition_log = self._kalman_pred(row, hour, day_j)
        preds["kalman"] = kalman_pred
        if self.has_ols:
            preds["ols"] = self._static_pred(self.ols.beta_[hour], self._ols_mse[hour], x_all)
        preds["sure"] = self._static_pred(self.sure.beta_[hour], self._sure_mse[hour], x_all)

        actual = None
        ts_candidates = self.full_per_hour[hour]
        if target_date in ts_candidates.index:
            actual_val = ts_candidates.loc[target_date, TARGET_COLUMN]
            actual = float(actual_val) if pd.notna(actual_val) else None

        return HourPrediction(
            date=target_date, hour=hour, actual=actual, preds=preds, decomposition_log=decomposition_log
        )

    def forecast_horizon(self, day_j: dt.date, temp_delta: float = 0.0) -> list[HourPrediction]:
        """31 points : J[17h-23h] puis J+1[0h-23h], comme une entreprise le ferait à J 17h."""
        results = []
        day_j1 = day_j + dt.timedelta(days=1)
        for hour in range(17, 24):
            pred = self.predict_hour(day_j, day_j, hour, temp_delta)
            if pred is not None:
                results.append(pred)
        for hour in range(24):
            pred = self.predict_hour(day_j, day_j1, hour, temp_delta)
            if pred is not None:
                results.append(pred)
        return results

    def rolling_performance(self, end_date: dt.date, window_days: int = 30) -> pd.DataFrame:
        """Pour chaque jour d de [end_date-window, end_date], la prévision des 24h
        de d telle qu'elle aurait été faite la veille à 17h, comparée au réel.
        Retourne un DataFrame long (date, modele, rmse, mape) pour chaque modèle."""
        from models.metrics import mape as mape_fn
        from models.metrics import rmse as rmse_fn

        rows = []
        start = end_date - dt.timedelta(days=window_days)
        for d in pd.date_range(start, end_date, freq="D").date:
            j = d - dt.timedelta(days=1)
            preds = [self.predict_hour(j, d, h) for h in range(24)]
            preds = [p for p in preds if p is not None and p.actual is not None]
            if not preds:
                continue
            actual = pd.Series([p.actual for p in preds])
            for key, label in self.models:
                pred_series = pd.Series([p.preds[key].value for p in preds])
                rows.append({
                    "date": d, "modele": label,
                    "rmse": rmse_fn(actual, pred_series), "mape": mape_fn(actual, pred_series),
                })
        return pd.DataFrame(rows)
