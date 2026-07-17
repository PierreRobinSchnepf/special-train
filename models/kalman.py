"""Benchmark 3 : filtre de Kalman appliqué aux coefficients d'un système SURE.

Référence théorique :
- Slide "Équation finale : SUR ajusté dynamiquement" (ENGIE / ENSAE) :
      vrai coefficient = effet structurel moyen x ajustement dynamique
      β^true_{t,h,j} = β^SUR_{h,j} · β^Kalman_{t,h,j}
      y_{t,h} = β_0 + Σ_j (β^SUR_{h,j} β^Kalman_{t,h,j}) x_{t,h,j}
                     + Σ_j (β^SUR_j   β^Kalman_{t,j})   z_{t,j} + ε_{t,h}
  → le modèle garde la structure explicable du SUR, mais chaque effet peut
    dériver dans le temps via un facteur d'échelle multiplicatif.
- Mécanique du filtre : notebook de référence
  github.com/PierreRobinSchnepf/Applied-Statistics-ENGIE (kalman_sur1h.ipynb).
  Reproduite ici à l'identique :
    * cible en log (`log1p(y)` / `expm1`) — nécessaire pour que les facteurs
      d'échelle soient comparables en ordre de grandeur d'une variable/heure
      à l'autre malgré des contributions structurelles très différentes ;
    * état = marche aléatoire (β_pred = β_{t-1}, P_pred = P_{t-1} + W) ;
    * observation scalaire y_t = H_t β_t + ε_t, où H_t est la contribution
      structurelle SUR (β^SUR_{h,j} · x_{t,h,j}), pas x_t brut — c'est ce
      qui rend l'état interprétable comme un facteur d'ajustement autour de 1 ;
    * mise à jour de Kalman standard (gain, innovation, covariance).

Écart assumé par rapport au notebook de référence : l'intercept β_0 est
gardé FIXE (non multiplié par un facteur Kalman), conformément à la
formule de la slide. Le notebook de référence, lui, absorbait un intercept
dynamique dans le vecteur d'état — non repris ici car absent de la formule
fournie comme spécification.

Contrairement au notebook de référence (une seule heure, hyperparamètres
V/W codés en dur), ce module fait tourner **24 filtres indépendants** (un
par heure locale, même schéma que HourlyOLSModel / HourlySUREModel) et
estime le bruit d'observation V_h automatiquement à partir de la variance
résiduelle du SUR en log pour cette heure, plutôt que de recopier une
constante réglée pour une seule équation d'un autre jeu de données.
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
    """Filtre de Kalman scalaire standard, état = marche aléatoire.

    H: (T, p) design (contributions structurelles SUR par pas de temps).
    y: (T,) observations (log-cible moins intercept fixe).
    W: bruit de process, scalaire (uniforme) ou vecteur de longueur p
       (ex. composante différente pour un état "intercept dynamique").
    Retourne (beta_history, y_pred_pre_update, beta_final, P_final) :
    `y_pred_pre_update[t]` est la prédiction faite AVANT d'assimiler
    l'observation t (prévision honnête à un pas), `beta_history[t]` est
    l'état APRÈS assimilation de l'observation t.
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

        self.beta_state_: dict[int, np.ndarray] = {}   # état courant (marche à mesure des predict())
        self.P_state_: dict[int, np.ndarray] = {}

        self.train_beta_history_: dict[int, pd.DataFrame] = {}
        self.test_beta_history_: dict[int, pd.DataFrame] = {}

        # Prédictions en-échantillon un-pas-en-avant, calculées une seule fois
        # pendant fit() (la vraie trajectoire d'entraînement, partant de l'état
        # initial =1) — à ne pas confondre avec un appel à predict(train_per_hour),
        # qui reprendrait l'état déjà convergé en fin d'entraînement.
        self.train_sur_pred_: dict[int, pd.Series] = {}
        self.train_kalman_pred_: dict[int, pd.Series] = {}

    # -- entraînement -------------------------------------------------

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

    # -- prédiction séquentielle (un pas en avant) ---------------------

    def predict(
        self, per_hour: dict[int, pd.DataFrame], update_state: bool = True
    ) -> tuple[dict[int, pd.Series], dict[int, pd.Series]]:
        """Prédiction un-pas-en-avant sur `per_hour`, en poursuivant la récursion
        de Kalman à partir de l'état courant (fin d'entraînement, ou fin du
        dernier appel à predict() si `update_state=True`).

        Retourne (sur_pred, kalman_pred), en NIVEAU (MW), chacun un
        dict[heure -> pd.Series indexée par utc_ts] :
        - sur_pred    : baseline SUR pure (facteurs figés à 1, pas de Kalman)
        - kalman_pred : SUR + ajustement dynamique du filtre de Kalman
        """
        if self.sure_ is None:
            raise RuntimeError("fit() doit être appelé avant predict()")

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

            kalman_log = self.intercept_[h] + y_pred_resid   # prédiction PRE-update (un pas en avant)
            sur_log = self.intercept_[h] + H.sum(axis=1)      # facteurs figés à 1

            utc_ts = frame["utc_ts"].to_numpy()
            kalman_preds[h] = pd.Series(np.expm1(kalman_log), index=utc_ts, name=f"kalman_pred_h{h}")
            sur_preds[h] = pd.Series(np.expm1(sur_log), index=utc_ts, name=f"sur_pred_h{h}")

        return sur_preds, kalman_preds

    # -- inspection -----------------------------------------------------

    def full_beta_trajectory(self, hour: int) -> pd.DataFrame:
        """Trajectoire complète (train puis test, si predict() a été appelé) des
        facteurs d'échelle de Kalman pour une heure donnée, indexée par date locale."""
        parts = [self.train_beta_history_[hour]]
        if hour in self.test_beta_history_:
            parts.append(self.test_beta_history_[hour])
        return pd.concat(parts)
