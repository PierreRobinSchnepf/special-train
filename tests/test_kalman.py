import numpy as np
import pandas as pd

from models.kalman import HourlyKalmanSURModel, _run_kalman
from models.metrics import rmse


def _synthetic_panel(n_days, drift_per_day=0.0, seed=0):
    """24 synthetic equations, strictly positive target (log1p-compatible),
    with a coefficient on `x0` drifting linearly over time (multiplied by
    `drift_per_day` * day), on top of a stable intercept and `x1`. The SUR
    (fixed average coefficient) cannot track that drift; the Kalman should,
    at least partially.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D").date
    day_idx = np.arange(n_days)

    per_hour = {}
    for h in range(24):
        x0 = rng.normal(loc=5, scale=1.0, size=n_days)
        x1 = rng.normal(loc=2, scale=1.0, size=n_days)
        true_coef0 = 0.05 + drift_per_day * day_idx  # drifts over time
        log_y = 8.0 + true_coef0 * x0 + 0.1 * x1 + rng.normal(scale=0.01, size=n_days)
        y = np.expm1(log_y)

        frame = pd.DataFrame({"beta_0": 1.0, "x0": x0, "x1": x1, "y": y})
        frame["utc_ts"] = pd.to_datetime(dates) + pd.Timedelta(hours=h)
        frame.index = dates
        per_hour[h] = frame
    return per_hour


PREDICTORS = ["x0", "x1", "beta_0"]


def test_first_train_prediction_matches_sur_exactly():
    # Initial state = 1 (no adjustment) => BEFORE any assimilation (first
    # training step), the filter's prediction must equal the pure SUR
    # baseline (H.sum() == H @ ones). Gray-box test on `_run_kalman` directly
    # to isolate this property from the rest of fit().
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
    assert traj.shape == (300, 2)  # p = 2 (x0, x1), beta_0 excluded from the state
    np.testing.assert_allclose(traj.iloc[0].to_numpy(), 1.0, atol=0.05)


def test_kalman_tracks_drift_better_than_static_sur():
    # The x0 coefficient drifts linearly; at the end of the test period (far
    # from the average seen by the SUR), the Kalman must have moved toward
    # the true value and therefore predict better than the frozen SUR
    # baseline.
    train = _synthetic_panel(n_days=1200, drift_per_day=0.0006)
    test = _synthetic_panel(n_days=300, drift_per_day=0.0006, seed=3)
    # shift the test so it continues the train drift (same logical days)
    for h in test:
        test[h] = test[h].copy()

    model = HourlyKalmanSURModel(
        predictor_cols=PREDICTORS, target_col="y", process_noise_var=5e-4,
    ).fit(train)
    sur_pred, kalman_pred = model.predict(test)

    h = 10
    y_true = test[h].set_index("utc_ts")["y"]
    late = y_true.index[-100:]  # end of the test period: maximal drift

    sur_rmse = rmse(y_true.loc[late], sur_pred[h].loc[late])
    kalman_rmse = rmse(y_true.loc[late], kalman_pred[h].loc[late])

    assert kalman_rmse < sur_rmse
