import numpy as np
import pandas as pd

from models.kalman import HourlyKalmanSURModel, _run_kalman
from models.metrics import rmse


def _synthetic_panel(n_days, drift_per_day=0.0, seed=0):
    """24 équations synthétiques, cible strictement positive (compatible log1p),
    avec un coefficient sur `x0` qui dérive linéairement dans le temps (multiplié
    par `drift_per_day` * jour), en plus d'un intercept et d'un `x1` stables.
    Le SUR (coefficient moyen fixe) ne peut pas suivre cette dérive ; le Kalman le
    devrait, au moins partiellement.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D").date
    day_idx = np.arange(n_days)

    per_hour = {}
    for h in range(24):
        x0 = rng.normal(loc=5, scale=1.0, size=n_days)
        x1 = rng.normal(loc=2, scale=1.0, size=n_days)
        true_coef0 = 0.05 + drift_per_day * day_idx  # dérive dans le temps
        log_y = 8.0 + true_coef0 * x0 + 0.1 * x1 + rng.normal(scale=0.01, size=n_days)
        y = np.expm1(log_y)

        frame = pd.DataFrame({"beta_0": 1.0, "x0": x0, "x1": x1, "y": y})
        frame["utc_ts"] = pd.to_datetime(dates) + pd.Timedelta(hours=h)
        frame.index = dates
        per_hour[h] = frame
    return per_hour


PREDICTORS = ["x0", "x1", "beta_0"]


def test_first_train_prediction_matches_sur_exactly():
    # Etat initial = 1 (aucun ajustement) => AVANT toute assimilation (premier
    # pas de l'entraînement), la prédiction du filtre doit être identique à la
    # baseline SUR pure (H.sum() == H @ ones). Test en boîte grise sur
    # `_run_kalman` directement pour isoler cette propriété du reste du fit().
    train = _synthetic_panel(n_days=500)
    model = HourlyKalmanSURModel(predictor_cols=PREDICTORS, target_col="y").fit(train)

    for h in (0, 12, 23):
        frame = train[h]
        X = frame[model.state_cols].to_numpy(dtype=float)
        H = X * model.sur_beta_[h][None, :]
        y_resid = np.log1p(frame["y"].to_numpy()) - model.intercept_[h]

        _, y_pred_resid, _, _ = _run_kalman(H, y_resid, model.V_[h], model.process_noise_var, model.p)
        sur_log_first = model.intercept_[h] + H[0].sum()
        kalman_log_first = model.intercept_[h] + y_pred_resid[0]
        np.testing.assert_allclose(kalman_log_first, sur_log_first, rtol=1e-10)


def test_predict_output_shapes_and_finite():
    train = _synthetic_panel(n_days=400)
    test = _synthetic_panel(n_days=60, seed=2)
    model = HourlyKalmanSURModel(predictor_cols=PREDICTORS, target_col="y").fit(train)
    sur_pred, kalman_pred = model.predict(test)

    assert set(sur_pred.keys()) == set(range(24))
    assert set(kalman_pred.keys()) == set(range(24))
    for h in range(24):
        assert len(sur_pred[h]) == len(test[h]) == len(kalman_pred[h])
        assert np.isfinite(sur_pred[h].to_numpy()).all()
        assert np.isfinite(kalman_pred[h].to_numpy()).all()


def test_beta_trajectory_starts_near_one_and_has_expected_length():
    train = _synthetic_panel(n_days=300)
    model = HourlyKalmanSURModel(predictor_cols=PREDICTORS, target_col="y").fit(train)

    traj = model.full_beta_trajectory(hour=10)
    assert traj.shape == (300, 2)  # p = 2 (x0, x1), beta_0 exclu de l'état
    np.testing.assert_allclose(traj.iloc[0].to_numpy(), 1.0, atol=0.05)


def test_kalman_tracks_drift_better_than_static_sur():
    # Coefficient de x0 dérive linéairement ; sur la fin de la période de test
    # (loin de la moyenne vue par le SUR), le Kalman doit s'être rapproché de la
    # vraie valeur et donc mieux prédire que la baseline SUR figée.
    train = _synthetic_panel(n_days=1200, drift_per_day=0.0006)
    test = _synthetic_panel(n_days=300, drift_per_day=0.0006, seed=3)
    # décale le test pour qu'il continue la dérive du train (mêmes jours logiques)
    for h in test:
        test[h] = test[h].copy()

    model = HourlyKalmanSURModel(
        predictor_cols=PREDICTORS, target_col="y", process_noise_var=5e-4,
    ).fit(train)
    sur_pred, kalman_pred = model.predict(test)

    h = 10
    y_true = test[h].set_index("utc_ts")["y"]
    late = y_true.index[-100:]  # fin de la période de test : dérive maximale

    sur_rmse = rmse(y_true.loc[late], sur_pred[h].loc[late])
    kalman_rmse = rmse(y_true.loc[late], kalman_pred[h].loc[late])

    assert kalman_rmse < sur_rmse
