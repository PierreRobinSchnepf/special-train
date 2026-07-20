"""Benchmark 2: SURE system (Seemingly Unrelated Regressions, Zellner 1962)
over the 24 hourly equations, estimated by FGLS with `statsmodels`.

Principle: the 24 equations share the same predictor set but hour-specific
coefficients (like the OLS model). The difference is that their residuals,
taken on the same day, are correlated (unobserved daily shocks common to all
hours — fine-grained weather, unusual behavior, poorly captured holidays,
etc.). SURE exploits this contemporaneous correlation to gain efficiency over
24 independent OLS regressions (Aitken/FGLS).

Implementation — why the whitening is done by hand rather than through
`statsmodels.GLS(sigma=...)`:
    The full covariance matrix of the stacked system is (24T) x (24T)
    (T = number of days). With T in the thousands, that matrix is far too
    large to build or invert densely (tens of GB). But when observations are
    ordered "day first, hour second", this covariance is block-diagonal with
    T identical blocks equal to Sigma (24x24, the contemporaneous covariance
    across equations). The system can therefore be whitened day by day with
    P = chol(Sigma)^-1 (P Sigma P' = I), without ever forming the full
    matrix, then `statsmodels.OLS` is called on the stacked whitened system:
    this is exactly Zellner's FGLS estimator, computed efficiently.
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
        self.sigma_: np.ndarray | None = None          # (24, 24) contemporaneous residual covariance
        self.beta_: np.ndarray | None = None            # (24, k) coefficients per hour
        self.stage1_results_: dict[int, sm.regression.linear_model.RegressionResultsWrapper] = {}
        self.whitened_result_ = None                     # statsmodels result of the stacked/whitened system
        # Stage-1 residual variance, extracted at fit time (cf.
        # HourlyOLSModel.mse_resid_: `models.persistence` strips the raw data
        # from statsmodels results before saving).
        self.stage1_resid_var_: dict[int, float] = {}

    def fit(self, train_per_hour: dict[int, pd.DataFrame], target_col: str = TARGET_COLUMN) -> "HourlySUREModel":
        hours = list(range(N_HOURS))
        dates = train_per_hour[0].index
        for h in hours:
            if not train_per_hour[h].index.equals(dates):
                raise ValueError("panel not balanced across equations: use build_hourly_equations()")
        T = len(dates)

        Y = np.column_stack([train_per_hour[h][target_col].to_numpy() for h in hours])          # (T, 24)
        X = np.stack([train_per_hour[h][self.predictor_cols].to_numpy() for h in hours], axis=1)  # (T, 24, k)

        # --- Stage 1: equation-by-equation OLS -> residuals -> Sigma ---
        resid = np.empty((T, N_HOURS))
        for h in hours:
            res = sm.OLS(Y[:, h], X[:, h, :]).fit()
            self.stage1_results_[h] = res
            resid[:, h] = res.resid
            self.stage1_resid_var_[h] = float(np.var(resid[:, h]))
        self.sigma_ = (resid.T @ resid) / T

        # --- Stage 2: whitening P = chol(Sigma, lower)^-1, then stacked OLS ---
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
            raise RuntimeError("fit() must be called before predict()")
        preds = {}
        for h, frame in per_hour.items():
            X = frame[self.predictor_cols].to_numpy()
            pred = X @ self.beta_[h]
            preds[h] = pd.Series(pred, index=frame["utc_ts"].to_numpy(), name=f"sure_pred_h{h}")
        return preds

    def coefficients(self) -> pd.DataFrame:
        return pd.DataFrame(self.beta_, index=range(N_HOURS), columns=self.predictor_cols)

    def sigma_frame(self) -> pd.DataFrame:
        """Contemporaneous covariance matrix (24x24) estimated in stage 1, for
        inspection (e.g. residual correlation between nearby vs distant hours)."""
        return pd.DataFrame(self.sigma_, index=range(N_HOURS), columns=range(N_HOURS))
