"""Benchmark 3: Kalman filter applied to the coefficients of a SURE system.

Theoretical reference:
- Slide "Final equation: dynamically adjusted SUR" (ENGIE / ENSAE):
      true coefficient = average structural effect x dynamic adjustment
      β^true_{t,h,j} = β^SUR_{h,j} · β^Kalman_{t,h,j}
      y_{t,h} = β_0 + Σ_j (β^SUR_{h,j} β^Kalman_{t,h,j}) x_{t,h,j}
                     + Σ_j (β^SUR_j   β^Kalman_{t,j})   z_{t,j} + ε_{t,h}
  → the model keeps the explainable SUR structure, but each effect can drift
    over time through a multiplicative scale factor.
- Filter mechanics: reference notebook
  github.com/PierreRobinSchnepf/Applied-Statistics-ENGIE (kalman_sur1h.ipynb).
  Reproduced here identically:
    * log target (`log1p(y)` / `expm1`) — required so that scale factors are
      comparable in magnitude across variables/hours despite very different
      structural contributions;
    * state = random walk (β_pred = β_{t-1}, P_pred = P_{t-1} + W);
    * scalar observation y_t = H_t β_t + ε_t, where H_t is the SUR structural
      contribution (β^SUR_{h,j} · x_{t,h,j}), not raw x_t — this is what
      makes the state interpretable as an adjustment factor around 1;
    * standard Kalman update (gain, innovation, covariance).

Deliberate deviation from the reference notebook: the intercept β_0 is kept
FIXED (not multiplied by a Kalman factor), following the slide's formula. The
reference notebook absorbed a dynamic intercept into the state vector — not
reproduced here because it is absent from the formula given as the
specification.

Unlike the reference notebook (a single hour, hard-coded V/W
hyperparameters), this module runs **24 independent filters** (one per local
hour, same scheme as HourlyOLSModel / HourlySUREModel) and estimates the
observation noise V_h automatically from the SUR's log-space residual
variance for that hour, rather than copying a constant tuned for a single
equation of another dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models.dataset import PREDICTOR_COLUMNS, TARGET_COLUMN
from models.sure import HourlySUREModel

N_HOURS = 24
LOG_TARGET_COLUMN = "_log_target"


def _with_log_target(per_hour: dict[int, pd.DataFrame], target_col: str) -> dict[int, pd.DataFrame]:
    out = {}
    for h, frame in per_hour.items():
        frame = frame.copy()
        frame[LOG_TARGET_COLUMN] = np.log1p(frame[target_col].to_numpy())
        out[h] = frame
    return out


def _run_kalman(
    H: np.ndarray,
    y: np.ndarray,
    V: float,
    W: float | np.ndarray,
    p: int,
    beta_init: np.ndarray | None = None,
    P_init: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standard scalar-observation Kalman filter, random-walk state.

    H: (T, p) design (SUR structural contributions per time step).
    y: (T,) observations (log target minus the fixed intercept).
    W: process noise, scalar (uniform) or length-p vector (e.g. a different
       component for a "dynamic intercept" state).
    Returns (beta_history, y_pred_pre_update, beta_final, P_final):
    `y_pred_pre_update[t]` is the prediction made BEFORE assimilating
    observation t (honest one-step-ahead forecast), `beta_history[t]` is the
    state AFTER assimilating observation t.
    """
    T = H.shape[0]
    Wmat = np.diag(np.broadcast_to(W, p).astype(float))
    beta = np.ones(p) if beta_init is None else beta_init.copy()
    P = np.eye(p) if P_init is None else P_init.copy()

    beta_history = np.empty((T, p))
    y_pred = np.empty(T)

    for t in range(T):
        beta_pred = beta
        P_pred = P + Wmat
        H_t = H[t]

        y_hat = H_t @ beta_pred
        y_pred[t] = y_hat

        innov = y[t] - y_hat
        S = float(H_t @ P_pred @ H_t.T + V)
        K = (P_pred @ H_t) / S

        beta = beta_pred + K * innov
        P = P_pred - np.outer(K, H_t @ P_pred)
        beta_history[t] = beta

    return beta_history, y_pred, beta, P


class HourlyKalmanSURModel:
    def __init__(
        self,
        predictor_cols: list[str] = PREDICTOR_COLUMNS,
        target_col: str = TARGET_COLUMN,
        process_noise_var: float = 1e-4,
        obs_noise_var: float | None = None,
    ):
        self.predictor_cols = list(predictor_cols)
        self.state_cols = [c for c in self.predictor_cols if c != "beta_0"]
        self.p = len(self.state_cols)
        self.target_col = target_col
        self.process_noise_var = process_noise_var
        self._obs_noise_var_arg = obs_noise_var

        self.sure_: HourlySUREModel | None = None
        self.intercept_: dict[int, float] = {}
        self.sur_beta_: dict[int, np.ndarray] = {}
        self.V_: dict[int, float] = {}

        self.beta_state_: dict[int, np.ndarray] = {}   # current state (advances with each predict())
        self.P_state_: dict[int, np.ndarray] = {}

        self.train_beta_history_: dict[int, pd.DataFrame] = {}
        self.test_beta_history_: dict[int, pd.DataFrame] = {}

        # In-sample one-step-ahead predictions, computed once during fit()
        # (the true training trajectory, starting from the initial state = 1)
        # — not to be confused with a predict(train_per_hour) call, which
        # would resume from the state already converged at the end of training.
        self.train_sur_pred_: dict[int, pd.Series] = {}
        self.train_kalman_pred_: dict[int, pd.Series] = {}

    # -- training -------------------------------------------------------

    def fit(self, train_per_hour: dict[int, pd.DataFrame]) -> "HourlyKalmanSURModel":
        log_train = _with_log_target(train_per_hour, self.target_col)

        self.sure_ = HourlySUREModel(predictor_cols=self.predictor_cols).fit(
            log_train, target_col=LOG_TARGET_COLUMN
        )
        coefs = self.sure_.coefficients()

        for h in range(N_HOURS):
            frame = log_train[h]
            self.intercept_[h] = float(coefs.loc[h, "beta_0"])
            beta_sur_h = coefs.loc[h, self.state_cols].to_numpy(dtype=float)
            self.sur_beta_[h] = beta_sur_h

            X = frame[self.state_cols].to_numpy(dtype=float)
            H = X * beta_sur_h[None, :]
            y_resid = frame[LOG_TARGET_COLUMN].to_numpy() - self.intercept_[h]

            V_h = (
                self._obs_noise_var_arg
                if self._obs_noise_var_arg is not None
                else self.sure_.stage1_resid_var_[h]
            )
            self.V_[h] = V_h

            beta_hist, y_pred_resid, beta_final, P_final = _run_kalman(H, y_resid, V_h, self.process_noise_var, self.p)

            self.train_beta_history_[h] = pd.DataFrame(beta_hist, index=frame.index, columns=self.state_cols)
            self.beta_state_[h] = beta_final
            self.P_state_[h] = P_final

            utc_ts = frame["utc_ts"].to_numpy()
            kalman_log_train = self.intercept_[h] + y_pred_resid
            sur_log_train = self.intercept_[h] + H.sum(axis=1)
            self.train_kalman_pred_[h] = pd.Series(np.expm1(kalman_log_train), index=utc_ts, name=f"kalman_pred_h{h}")
            self.train_sur_pred_[h] = pd.Series(np.expm1(sur_log_train), index=utc_ts, name=f"sur_pred_h{h}")

        return self

    # -- sequential prediction (one step ahead) -------------------------

    def predict(
        self, per_hour: dict[int, pd.DataFrame], update_state: bool = True
    ) -> tuple[dict[int, pd.Series], dict[int, pd.Series]]:
        """One-step-ahead prediction over `per_hour`, continuing the Kalman
        recursion from the current state (end of training, or end of the last
        predict() call when `update_state=True`).

        Returns (sur_pred, kalman_pred), in LEVEL (MW), each a
        dict[hour -> pd.Series indexed by utc_ts]:
        - sur_pred    : pure SUR baseline (factors frozen at 1, no Kalman)
        - kalman_pred : SUR + dynamic Kalman-filter adjustment
        """
        if self.sure_ is None:
            raise RuntimeError("fit() must be called before predict()")

        sur_preds, kalman_preds = {}, {}
        for h in range(N_HOURS):
            frame = per_hour[h]
            X = frame[self.state_cols].to_numpy(dtype=float)
            H = X * self.sur_beta_[h][None, :]
            y_log = np.log1p(frame[self.target_col].to_numpy())
            y_resid = y_log - self.intercept_[h]

            beta_hist, y_pred_resid, beta_final, P_final = _run_kalman(
                H, y_resid, self.V_[h], self.process_noise_var, self.p,
                beta_init=self.beta_state_[h], P_init=self.P_state_[h],
            )

            self.test_beta_history_[h] = pd.DataFrame(beta_hist, index=frame.index, columns=self.state_cols)
            if update_state:
                self.beta_state_[h] = beta_final
                self.P_state_[h] = P_final

            kalman_log = self.intercept_[h] + y_pred_resid   # PRE-update prediction (one step ahead)
            sur_log = self.intercept_[h] + H.sum(axis=1)      # factors frozen at 1

            utc_ts = frame["utc_ts"].to_numpy()
            kalman_preds[h] = pd.Series(np.expm1(kalman_log), index=utc_ts, name=f"kalman_pred_h{h}")
            sur_preds[h] = pd.Series(np.expm1(sur_log), index=utc_ts, name=f"sur_pred_h{h}")

        return sur_preds, kalman_preds

    # -- inspection -----------------------------------------------------

    def full_beta_trajectory(self, hour: int) -> pd.DataFrame:
        """Full trajectory (train then test, if predict() was called) of the
        Kalman scale factors for a given hour, indexed by local date."""
        parts = [self.train_beta_history_[hour]]
        if hour in self.test_beta_history_:
            parts.append(self.test_beta_history_[hour])
        return pd.concat(parts)
