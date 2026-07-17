"""Benchmark 1 : 24 équations OLS indépendantes (une par heure locale).

Chaque heure h a ses propres coefficients, estimés séparément par
`statsmodels.OLS`, sans partage d'information entre équations. C'est la
référence naïve à laquelle comparer le gain (ou non) du système SURE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from models.dataset import PREDICTOR_COLUMNS, TARGET_COLUMN


class HourlyOLSModel:
    def __init__(self, predictor_cols: list[str] = PREDICTOR_COLUMNS):
        self.predictor_cols = list(predictor_cols)
        self.results_: dict[int, sm.regression.linear_model.RegressionResultsWrapper] = {}
        # Statistiques/coefficients dérivés, extraits à l'entraînement plutôt
        # que relus plus tard sur `results_` : `models.persistence` retire
        # les données brutes (exog/endog) des résultats avant sauvegarde
        # pour limiter la taille des artefacts, ce qui casse les propriétés
        # calculées à la demande (mse_resid, rsquared) ET fait perdre le
        # nom des colonnes de `.params` après un aller-retour pickle (il
        # retombe sur un index entier positionnel) — vérifié empiriquement.
        # `beta_` est donc la seule source utilisée pour prédire, jamais
        # `results_[h].params` directement.
        self.beta_: dict[int, np.ndarray] = {}
        self.mse_resid_: dict[int, float] = {}
        self.rsquared_: dict[int, float] = {}

    def fit(self, train_per_hour: dict[int, pd.DataFrame], target_col: str = TARGET_COLUMN) -> "HourlyOLSModel":
        for h, frame in train_per_hour.items():
            y = frame[target_col]
            X = frame[self.predictor_cols]
            res = sm.OLS(y, X).fit()
            self.results_[h] = res
            self.beta_[h] = res.params[self.predictor_cols].to_numpy(dtype=float)
            self.mse_resid_[h] = float(res.mse_resid)
            self.rsquared_[h] = float(res.rsquared)
        return self

    def predict(self, per_hour: dict[int, pd.DataFrame]) -> dict[int, pd.Series]:
        preds = {}
        for h, frame in per_hour.items():
            x = frame[self.predictor_cols].to_numpy(dtype=float)
            pred = x @ self.beta_[h]
            preds[h] = pd.Series(pred, index=frame["utc_ts"].to_numpy(), name=f"ols_pred_h{h}")
        return preds

    def coefficients(self) -> pd.DataFrame:
        """Coefficients (heures x prédicteurs), pour inspection."""
        return pd.DataFrame(self.beta_, index=self.predictor_cols).T.sort_index()

    def r_squared_by_hour(self) -> pd.Series:
        return pd.Series(self.rsquared_).sort_index()
