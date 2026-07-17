"""Bloc thermique du Tableau 1 : X1_heating, temp_smo, X2_smo_heating.

Toutes les fonctions sont causales : elles ne consomment que le passé et le
présent de la série d'entrée (pas de fuite de données).
"""
from __future__ import annotations

import pandas as pd


def compute_x1_heating(temp: pd.Series, t_base: float) -> pd.Series:
    """X1_t = max(0, t_base - T_t) — réaction immédiate au froid."""
    return (t_base - temp).clip(lower=0.0).rename("X1_heating")


def compute_temp_smo(temp: pd.Series, kappa: float) -> pd.Series:
    """T_smo_t = kappa * T_smo_{t-1} + (1-kappa) * T_t.

    Équivalent à une EWMA causale de facteur de lissage alpha = 1 - kappa,
    avec adjust=False (récursion exacte, pas de renormalisation par les
    poids des observations passées). T_smo_0 = T_0 (choix "first_observation"
    documenté dans config.yaml, correspond au comportement natif de
    pandas.ewm(adjust=False)).

    Les NaN internes à `temp` doivent être comblés (ffill borné) par l'appelant
    avant l'appel : pandas.ewm propage silencieusement la dernière valeur non
    nulle à travers les NaN, ce qui masquerait des trous de données non
    documentés.
    """
    alpha = 1.0 - kappa
    return temp.ewm(alpha=alpha, adjust=False).mean().rename("temp_smo")


def compute_x2_smo_heating(temp_smo: pd.Series, t_base: float) -> pd.Series:
    """X2_t = max(0, t_base - T_smo_t) — inertie du chauffage."""
    return (t_base - temp_smo).clip(lower=0.0).rename("X2_smo_heating")
