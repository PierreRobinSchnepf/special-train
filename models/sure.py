"""Benchmark 2 : système SURE (Seemingly Unrelated Regressions, Zellner 1962)
sur les 24 équations horaires, estimé par FGLS avec `statsmodels`.

Principe : les 24 équations partagent le même jeu de prédicteurs mais des
coefficients propres à chaque heure (comme le modèle OLS). La différence est
que leurs résidus, pris à un même jour, sont corrélés (chocs journaliers non
observés communs à toutes les heures — météo fine, comportement inhabituel,
jour férié mal capté, etc.). SURE exploite cette corrélation contemporaine
pour gagner en efficacité par rapport à 24 OLS indépendants (Aitken/FGLS).

Implémentation — pourquoi le blanchiment est fait à la main plutôt que via
`statsmodels.GLS(sigma=...)` :
    La matrice de covariance complète du système empilé fait (24T) x (24T)
    (T = nombre de jours). Avec T de l'ordre de quelques milliers, cette
    matrice est bien trop grande pour être construite ou inversée en dense
    (des dizaines de Go). Mais en ordonnant les observations "jour d'abord,
    heure ensuite", cette covariance est bloc-diagonale avec T blocs
    identiques égaux à Sigma (24x24, la covariance contemporaine entre
    équations). On peut donc blanchir le système jour par jour avec
    P = chol(Sigma)^-1 (P Sigma P' = I), sans jamais former la matrice
    complète, puis appeler `statsmodels.OLS` sur le système empilé blanchi :
    c'est exactement l'estimateur FGLS de Zellner, calculé efficacement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.linalg import cholesky, solve_triangular

from models.dataset import PREDICTOR_COLUMNS, TARGET_COLUMN

N_HOURS = 24


class HourlySUREModel:
    def __init__(self, predictor_cols: list[str] = PREDICTOR_COLUMNS):
        self.predictor_cols = list(predictor_cols)
        self.k = len(self.predictor_cols)
        self.sigma_: np.ndarray | None = None          # (24, 24) covariance contemporaine des résidus
        self.beta_: np.ndarray | None = None            # (24, k) coefficients par heure
        self.stage1_results_: dict[int, sm.regression.linear_model.RegressionResultsWrapper] = {}
        self.whitened_result_ = None                     # résultat statsmodels du système empilé/blanchi
        # Variance des résidus stage-1, extraite à l'entraînement (cf.
        # HourlyOLSModel.mse_resid_ : `models.persistence` retire les
        # données brutes des résultats statsmodels avant sauvegarde).
        self.stage1_resid_var_: dict[int, float] = {}

    def fit(self, train_per_hour: dict[int, pd.DataFrame], target_col: str = TARGET_COLUMN) -> "HourlySUREModel":
        hours = list(range(N_HOURS))
        dates = train_per_hour[0].index
        for h in hours:
            if not train_per_hour[h].index.equals(dates):
                raise ValueError("panel non équilibré entre équations : utiliser build_hourly_equations()")
        T = len(dates)

        Y = np.column_stack([train_per_hour[h][target_col].to_numpy() for h in hours])          # (T, 24)
        X = np.stack([train_per_hour[h][self.predictor_cols].to_numpy() for h in hours], axis=1)  # (T, 24, k)

        # --- Étape 1 : OLS équation par équation -> résidus -> Sigma ---
        resid = np.empty((T, N_HOURS))
        for h in hours:
            res = sm.OLS(Y[:, h], X[:, h, :]).fit()
            self.stage1_results_[h] = res
            resid[:, h] = res.resid
            self.stage1_resid_var_[h] = float(np.var(resid[:, h]))
        self.sigma_ = (resid.T @ resid) / T

        # --- Étape 2 : blanchiment P = chol(Sigma, lower)^-1, puis OLS empilé ---
        L = cholesky(self.sigma_, lower=True)
        P = solve_triangular(L, np.eye(N_HOURS), lower=True)  # P Sigma P' = I

        Y_star = np.einsum("ij,tj->ti", P, Y)                      # (T, 24)
        X_star = np.einsum("ij,tjk->tijk", P, X)                    # (T, 24, 24, k)
        X_star = X_star.reshape(T, N_HOURS, N_HOURS * self.k)        # (T, 24, 24k)

        y_stacked = Y_star.reshape(T * N_HOURS)
        X_stacked = X_star.reshape(T * N_HOURS, N_HOURS * self.k)

        self.whitened_result_ = sm.OLS(y_stacked, X_stacked).fit()
        self.beta_ = self.whitened_result_.params.reshape(N_HOURS, self.k)
        return self

    def predict(self, per_hour: dict[int, pd.DataFrame]) -> dict[int, pd.Series]:
        if self.beta_ is None:
            raise RuntimeError("fit() doit être appelé avant predict()")
        preds = {}
        for h, frame in per_hour.items():
            X = frame[self.predictor_cols].to_numpy()
            pred = X @ self.beta_[h]
            preds[h] = pd.Series(pred, index=frame["utc_ts"].to_numpy(), name=f"sure_pred_h{h}")
        return preds

    def coefficients(self) -> pd.DataFrame:
        return pd.DataFrame(self.beta_, index=range(N_HOURS), columns=self.predictor_cols)

    def sigma_frame(self) -> pd.DataFrame:
        """Matrice de covariance contemporaine (24x24) estimée en étape 1, pour inspection
        (ex. corrélation résiduelle entre heures proches vs. heures éloignées)."""
        return pd.DataFrame(self.sigma_, index=range(N_HOURS), columns=range(N_HOURS))
