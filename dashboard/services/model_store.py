"""Charge et entraîne une seule fois (au démarrage de Flask) les 3 modèles de
la phase R&D, et expose les méthodes de lookup dont le dashboard a besoin :
prédiction d'une heure donnée pour un jour J avec la bonne règle "quelles
données étaient déjà connues au moment de la prévision".

Règle d'assimilation (cf. spec : à J 17h, on prévoit J[17h-23h] + J+1[0h-23h]) :
pour une heure cible `hour`, la dernière occurrence RÉELLEMENT déjà connue à
"J 17h" est :
  - aujourd'hui (J) si hour <= 16 (déjà passée dans la journée)
  - hier (J-1) si hour >= 17 (pas encore arrivée aujourd'hui)
Cette règle est la même que la cible du jour soit J (heures 17-23) ou J+1
(toutes les heures) : ce qui compte, c'est uniquement si l'heure `hour` a déjà
eu lieu AUJOURD'HUI (jour J, l'instant où on se place pour prévoir).
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
from models.persistence import KALMAN_TRAIN_START, OLS_TRAIN_START, SURE_TRAIN_START, load_artifact, save_artifact
from models.sure import HourlySUREModel

TEST_START = "2025-01-01"
TEST_END = "2026-01-01"

# Regroupement des prédicteurs par bloc du Tableau 1, pour la décomposition
# explicable de la prévision (onglet Forecast).
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
class HourPrediction:
    date: dt.date
    hour: int
    kalman: float
    kalman_lo: float
    kalman_hi: float
    ols: float
    ols_lo: float
    ols_hi: float
    sure: float
    sure_lo: float
    sure_hi: float
    actual: float | None
    decomposition_log: dict[str, float] = field(default_factory=dict)


class ModelStore:
    def __init__(self) -> None:
        self.df = load_dataset()
        self.per_hour_all = build_hourly_equations(self.df)
        self.train, self.test = split_train_test(self.per_hour_all, test_start=TEST_START, test_end=TEST_END)

        # Panel combiné (train+test), pour retrouver la ligne de prédicteurs
        # ou la valeur observée à n'importe quelle date de la période étudiée.
        self.full_per_hour = {
            h: pd.concat([self.train[h], self.test[h]]).sort_index() for h in range(24)
        }

        self.ols = load_artifact("backtest_ols")
        self.sure = load_artifact("backtest_sure")
        self.kalman = load_artifact("backtest_kalman")

        if self.ols is None:
            print("[model_store] pas d'artefact 'backtest_ols' — entraînement (train >= 2020) + sauvegarde...")
            ols_train, _ = split_train_test(self.per_hour_all, TEST_START, TEST_END, train_start=OLS_TRAIN_START)
            self.ols = HourlyOLSModel().fit(ols_train)
            save_artifact(self.ols, "backtest_ols")
        if self.sure is None:
            print("[model_store] pas d'artefact 'backtest_sure' — entraînement (train >= 2020) + sauvegarde...")
            sure_train, _ = split_train_test(self.per_hour_all, TEST_START, TEST_END, train_start=SURE_TRAIN_START)
            self.sure = HourlySUREModel().fit(sure_train)
            save_artifact(self.sure, "backtest_sure")
        if self.kalman is None:
            print("[model_store] pas d'artefact 'backtest_kalman' — entraînement (train >= 2018) + sauvegarde...")
            kalman_train, kalman_test = split_train_test(self.per_hour_all, TEST_START, TEST_END, train_start=KALMAN_TRAIN_START)
            self.kalman = HourlyKalmanSURModel().fit(kalman_train)
            self.kalman.predict(kalman_test)
            save_artifact(self.kalman, "backtest_kalman")
        print("[model_store] ready (modèles chargés depuis data/models/).")

        self.state_cols = self.kalman.state_cols
        # Résidu (variance) de l'OLS/SURE en niveau, par heure — pour les IC.
        self._ols_mse = self.ols.mse_resid_
        self._sure_mse = self.sure.stage1_resid_var_

    # ------------------------------------------------------------------

    def selectable_days(self) -> list[str]:
        """Jours J sélectionnables : année de test 2025, en laissant assez de
        marge pour que J+1 reste dans l'année (cf. choix utilisateur)."""
        start = pd.Timestamp("2025-01-01").date()
        end = pd.Timestamp("2025-12-30").date()
        return [d.isoformat() for d in pd.date_range(start, end, freq="D").date]

    def _predictor_row(self, hour: int, date: dt.date) -> pd.Series | None:
        frame = self.full_per_hour[hour]
        if date not in frame.index:
            return None
        row = frame.loc[date]
        if isinstance(row, pd.DataFrame):  # garde-fou (ne devrait pas arriver, panel dédupliqué)
            row = row.iloc[0]
        return row

    def _apply_what_if(self, row: pd.Series, temp_delta: float) -> pd.Series:
        """Décale la température prévue de `temp_delta` °C sur l'horizon de
        prévision. Approximation documentée : on décale directement les 3
        variables thermiques déjà calculées (pas de recalcul de l'EWMA complet
        de temp_smo à partir de la série brute) — suffisant pour explorer une
        sensibilité, pas pour une prévision opérationnelle."""
        if temp_delta == 0.0:
            return row
        row = row.copy()
        row["temp_smo"] = row["temp_smo"] + temp_delta
        row["X1_heating"] = max(0.0, row["X1_heating"] - temp_delta)
        row["X2_smo_heating"] = max(0.0, row["X2_smo_heating"] - temp_delta)
        return row

    # ------------------------------------------------------------------

    def predict_hour(
        self, day_j: dt.date, target_date: dt.date, hour: int, temp_delta: float = 0.0
    ) -> HourPrediction | None:
        row = self._predictor_row(hour, target_date)
        if row is None:
            return None
        row = self._apply_what_if(row, temp_delta)

        x_all = row[PREDICTOR_COLUMNS].to_numpy(dtype=float)

        # --- OLS (niveau, statique) ---
        ols_beta = self.ols.beta_[hour]
        ols_pred = float(x_all @ ols_beta)
        ols_se = float(np.sqrt(self._ols_mse[hour]))
        ols_lo, ols_hi = ols_pred - 1.96 * ols_se, ols_pred + 1.96 * ols_se

        # --- SURE (niveau, statique) ---
        sure_beta = self.sure.beta_[hour]
        sure_pred = float(x_all @ sure_beta)
        sure_se = float(np.sqrt(self._sure_mse[hour]))
        sure_lo, sure_hi = sure_pred - 1.96 * sure_se, sure_pred + 1.96 * sure_se

        # --- Kalman-adjusted SUR (log, dynamique) : "notre prédiction" ---
        assim_date = _assimilated_date(day_j, hour)
        traj = self.kalman.full_beta_trajectory(hour)
        lookup_date = min(assim_date, traj.index.max())
        lookup_date = max(lookup_date, traj.index.min())
        state = traj.loc[lookup_date].to_numpy(dtype=float)

        x_state = row[self.state_cols].to_numpy(dtype=float)
        sur_beta_h = self.kalman.sur_beta_[hour]
        contrib = x_state * sur_beta_h  # contribution structurelle SUR (log), par variable
        pred_log = self.kalman.intercept_[hour] + float(contrib @ state)
        kalman_pred = float(np.expm1(pred_log))

        # IC approximatif : P "convergée" (dernière connue) plutôt que P exact
        # à `assim_date` (non stockée jour par jour) — cf. docstring du module.
        P = self.kalman.P_state_[hour]
        H_t = contrib
        var_log = float(H_t @ P @ H_t.T + self.kalman.V_[hour])
        se_log = np.sqrt(max(var_log, 0.0))
        kalman_lo = float(np.expm1(pred_log - 1.96 * se_log))
        kalman_hi = float(np.expm1(pred_log + 1.96 * se_log))

        # décomposition (espace log, additive) par bloc + intercept
        decomposition_log = {"Fond de roulement (beta0)": self.kalman.intercept_[hour]}
        for block_name, cols in BLOCKS.items():
            idx = [self.state_cols.index(c) for c in cols]
            decomposition_log[block_name] = float(np.sum(contrib[idx] * state[idx]))

        actual = None
        ts_candidates = self.full_per_hour[hour]
        if target_date in ts_candidates.index:
            actual_val = ts_candidates.loc[target_date, TARGET_COLUMN]
            actual = float(actual_val) if pd.notna(actual_val) else None

        return HourPrediction(
            date=target_date, hour=hour,
            kalman=kalman_pred, kalman_lo=kalman_lo, kalman_hi=kalman_hi,
            ols=ols_pred, ols_lo=ols_lo, ols_hi=ols_hi,
            sure=sure_pred, sure_lo=sure_lo, sure_hi=sure_hi,
            actual=actual, decomposition_log=decomposition_log,
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
        """Pour chaque jour d du fenêtre [end_date-window, end_date], la prévision
        des 24h de d telle qu'elle aurait été faite la veille à 17h (J=d-1),
        comparée au réel. Retourne un DataFrame long (date, modele, rmse, mape)."""
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
            for model_name, attr in (("Kalman", "kalman"), ("OLS", "ols"), ("SURE", "sure")):
                pred_series = pd.Series([getattr(p, attr) for p in preds])
                rows.append({
                    "date": d, "modele": model_name,
                    "rmse": rmse_fn(actual, pred_series), "mape": mape_fn(actual, pred_series),
                })
        return pd.DataFrame(rows)
