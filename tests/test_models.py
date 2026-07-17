import numpy as np
import pandas as pd
import pytest

from models.ols import HourlyOLSModel
from models.sure import HourlySUREModel


def _synthetic_panel(n_days=400, k=3, seed=0, cross_eq_corr=0.0):
    """24 équations synthétiques y_h = X_h @ beta_h + e_h, avec corrélation
    contemporaine `cross_eq_corr` entre les résidus de toutes les équations
    (même jour). beta_h dépend de h pour que les 24 équations soient distinctes.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D").date

    true_betas = {h: rng.normal(size=k) for h in range(24)}

    sigma = np.full((24, 24), cross_eq_corr)
    np.fill_diagonal(sigma, 1.0)
    L = np.linalg.cholesky(sigma)
    raw_noise = rng.normal(size=(n_days, 24))
    correlated_noise = raw_noise @ L.T  # (n_days, 24), covariance == sigma

    per_hour = {}
    for h in range(24):
        X = rng.normal(size=(n_days, k))
        y = X @ true_betas[h] + correlated_noise[:, h]
        frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(k)])
        frame["y"] = y
        frame["utc_ts"] = pd.to_datetime(dates) + pd.Timedelta(hours=h)
        frame.index = dates
        per_hour[h] = frame
    return per_hour, true_betas


def test_ols_recovers_known_coefficients_on_synthetic_data():
    per_hour, true_betas = _synthetic_panel(n_days=2000, cross_eq_corr=0.0)
    model = HourlyOLSModel(predictor_cols=["x0", "x1", "x2"]).fit(per_hour, target_col="y")

    for h in range(24):
        np.testing.assert_allclose(model.results_[h].params.to_numpy(), true_betas[h], atol=0.15)


def test_sure_fit_predict_shapes():
    per_hour, _ = _synthetic_panel(n_days=300, cross_eq_corr=0.4)
    model = HourlySUREModel(predictor_cols=["x0", "x1", "x2"]).fit(per_hour, target_col="y")

    assert model.beta_.shape == (24, 3)
    assert model.sigma_.shape == (24, 24)

    preds = model.predict(per_hour)
    assert set(preds.keys()) == set(range(24))
    for h in range(24):
        assert len(preds[h]) == len(per_hour[h])
        assert np.isfinite(preds[h].to_numpy()).all()


def test_sure_matches_ols_when_equations_uncorrelated():
    # Quand les équations sont générées indépendamment, Sigma_hat est
    # *approximativement* diagonale (bruit d'échantillonnage sur les termes
    # hors-diagonale, jamais exactement 0) : le FGLS doit rester proche de
    # l'estimateur OLS équation par équation, sans coïncider exactement.
    per_hour, _ = _synthetic_panel(n_days=2000, cross_eq_corr=0.0)

    ols = HourlyOLSModel(predictor_cols=["x0", "x1", "x2"]).fit(per_hour, target_col="y")
    sure = HourlySUREModel(predictor_cols=["x0", "x1", "x2"]).fit(per_hour, target_col="y")

    off_diag = sure.sigma_ - np.diag(np.diag(sure.sigma_))
    assert np.abs(off_diag).max() < 0.1  # Sigma_hat quasi diagonale

    ols_coefs = ols.coefficients().to_numpy()
    sure_coefs = sure.coefficients().to_numpy()
    np.testing.assert_allclose(sure_coefs, ols_coefs, atol=0.02)


def test_sure_whitening_recovers_true_coefficients_with_correlated_errors():
    per_hour, true_betas = _synthetic_panel(n_days=3000, cross_eq_corr=0.6)
    sure = HourlySUREModel(predictor_cols=["x0", "x1", "x2"]).fit(per_hour, target_col="y")

    true_matrix = np.array([true_betas[h] for h in range(24)])
    np.testing.assert_allclose(sure.beta_, true_matrix, atol=0.15)
