"""Tests unitaires du parsing gaz régional (wide heure locale -> long UTC).

Purs (pas de réseau) : on fabrique des dataframes larges synthétiques qui
reproduisent les particularités constatées des deux datasets ODRÉ (noms de
colonnes horaires incohérents, plusieurs lignes par région pour opérateurs et
secteurs) et on vérifie l'agrégation et la conversion de fuseau.
"""
import numpy as np
import pandas as pd

from build_dataset import renormalized_weighted_mean
from src.regional_gas import _hour_columns, _mask_isolated_dips, _wide_to_regional_utc


def test_hour_columns_tolerates_naming_variants():
    df = pd.DataFrame(
        columns=["date", "code_region", "consommation_journaliere_mwh_pcs",
                 "00_00_00", "06_00", "11_00_00", "23_00"]
    )
    mapping = _hour_columns(df)
    # colonnes horaires reconnues quel que soit le suffixe, non-horaires ignorées
    assert mapping == {"00_00_00": 0, "06_00": 6, "11_00_00": 11, "23_00": 23}


def test_sums_over_operators_and_sectors_per_region():
    # deux opérateurs x deux secteurs pour la même région/jour : doivent se sommer
    df = pd.DataFrame(
        {
            "date": ["2025-12-01"] * 4,
            "code_region": [11, 11, 11, 11],
            "operateur": ["NaTran", "NaTran", "Terega", "Terega"],
            "secteur": ["A", "B", "A", "B"],
            "12_00": [10.0, 20.0, 1.0, 4.0],
        }
    )
    out = _wide_to_regional_utc(df, "code_region")
    # 12:00 heure locale Paris en hiver (CET=UTC+1) -> 11:00 UTC
    ts = pd.Timestamp("2025-12-01 11:00", tz="UTC")
    assert list(out.columns) == [11]
    assert out.loc[ts, 11] == 35.0


def test_local_to_utc_conversion_winter():
    df = pd.DataFrame(
        {"date": ["2025-12-01"], "code_region": [24], "00_00": [50.0]}
    )
    out = _wide_to_regional_utc(df, "code_region")
    # minuit heure locale Paris en hiver -> 23:00 UTC la veille
    assert out.index[0] == pd.Timestamp("2025-11-30 23:00", tz="UTC")
    assert out.iloc[0, 0] == 50.0


def test_regions_become_separate_columns():
    df = pd.DataFrame(
        {
            "date": ["2025-12-01", "2025-12-01"],
            "code_region": [11, 75],
            "08_00": [100.0, 200.0],
        }
    )
    out = _wide_to_regional_utc(df, "code_region")
    assert set(out.columns) == {11, 75}
    ts = pd.Timestamp("2025-12-01 07:00", tz="UTC")
    assert out.loc[ts, 11] == 100.0
    assert out.loc[ts, 75] == 200.0


# ---------------------------------------------------------------------------
# Pondération partagée national/régional (Étape B)
# ---------------------------------------------------------------------------

def test_weighted_mean_matches_manual():
    wide = pd.DataFrame({"a": [10.0, 20.0], "b": [0.0, 40.0]})
    out = renormalized_weighted_mean(wide, {"a": 3.0, "b": 1.0})
    # ligne 0 : (10*3 + 0*1)/4 = 7.5 ; ligne 1 : (20*3 + 40*1)/4 = 25
    np.testing.assert_allclose(out.to_numpy(), [7.5, 25.0])


def test_weighted_mean_renormalizes_on_missing():
    # quand une station manque, le poids se renormalise sur les disponibles
    wide = pd.DataFrame({"a": [10.0, np.nan], "b": [20.0, 30.0]})
    out = renormalized_weighted_mean(wide, {"a": 1.0, "b": 1.0})
    # ligne 0 : (10+20)/2 = 15 ; ligne 1 : b seule -> 30 (pas (0+30)/2)
    np.testing.assert_allclose(out.to_numpy(), [15.0, 30.0])


def test_weighted_mean_all_missing_is_nan():
    wide = pd.DataFrame({"a": [np.nan], "b": [np.nan]})
    out = renormalized_weighted_mean(wide, {"a": 1.0, "b": 2.0})
    assert np.isnan(out.iloc[0])


# ---------------------------------------------------------------------------
# Filtre des chutes isolées (artefact DST de la source régionale)
# ---------------------------------------------------------------------------

def test_isolated_dip_masked_but_smooth_trough_kept():
    # colonne A : collapse isolé d'une heure (1000 -> 5 -> 1000) => masqué
    # colonne B : creux lisse d'été (130 -> 104 -> 110) => conservé
    df = pd.DataFrame({
        "A": [1000.0, 5.0, 1000.0],
        "B": [130.0, 104.0, 110.0],
    })
    out = _mask_isolated_dips(df, dip_fraction=0.5)
    assert np.isnan(out.loc[1, "A"])       # anomalie masquée
    assert out.loc[1, "B"] == 104.0         # vrai creux intact
    assert out.loc[0, "A"] == 1000.0        # voisins intacts


def test_isolated_dip_edge_values_never_masked():
    # une valeur au bord (sans les deux voisins) n'est jamais masquée
    df = pd.DataFrame({"A": [1.0, 1000.0, 1000.0]})
    out = _mask_isolated_dips(df, dip_fraction=0.5)
    assert out.loc[0, "A"] == 1.0
