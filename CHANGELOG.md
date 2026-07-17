# Changelog

## v0.1.0 — 2026-07-17

Première version du projet : de la constitution du dataset jusqu'à un pipeline de prévision réel branché sur des données ouvertes, avec dashboard interactif et persistance des modèles.

### Dataset (R&D)

- Dataset horaire de consommation gaz française, 2018-01-01 → 2026-05-31 (73 750 lignes), reconstruit exclusivement à partir de sources ouvertes (ODRÉ, Météo-France via Open-Meteo).
- Variables du Tableau 1 : inertie thermique (`temp_smo`, `X1_heating`, `X2_smo_heating` par EWMA κ=0.98), 16 colonnes de saisonnalité de Fourier (jour ouvré / week-end), 6 variables calendaires, `beta_0`.
- Pipeline causal-only (aucune fuite d'information future), documenté dans `data_dictionary.md` et `README.md`.
- `fetch_data.py` / `build_dataset.py` : téléchargement + construction reproductibles.

### Exploration

- `notebooks/01_exploration_stats.ipynb` : 9 graphiques de cohérence du dataset.

### Benchmarks (24 équations horaires)

- `models/ols.py` — OLS heure par heure, indépendant.
- `models/sure.py` — SURE (FGLS manuel : résidus stage-1 → Σ 24×24 → blanchiment Cholesky → OLS empilé), statsmodels n'ayant pas de SUR natif.
- `notebooks/02_benchmark_ols_sure.ipynb` : entraînement 2018-2024 / test 2025, RMSE/MAPE, prévisions sur 2 semaines.

### Kalman-SUR

- `models/kalman.py` — filtre de Kalman ajustant dynamiquement les coefficients SUR (β_true = β_SUR × β_Kalman, β_0 fixe), généralisé aux 24 heures à partir du notebook de référence (une seule heure à l'origine).
- `notebooks/03_kalman_sur.ipynb`, `04_kalman_faithful_reproduction.ipynb` : comparaison SURE vs Kalman (métriques, prévisions, erreur cumulée, évolution des coefficients).

### Dashboard interactif (Streamlit)

- `dashboard/app.py`, 4 onglets : Forecast (Kalman + comparaison OLS/SURE), Benchmark (prévu vs réel), Suivi de performance glissant, Pipeline réel.
- Simulation du workflow J+1 d'une entreprise : prévision lancée à J 17h, horizon J[17h-23h] + J+1[0h-23h] (31 points).
- Extras : intervalles de confiance, monitoring de performance glissant, simulateur what-if météo, décomposition explicable par bloc (thermique / saisonnier / calendaire).

### Pipeline réel

- Recherche de fraîcheur des données gaz : agrégat national à ~45-50 j de délai ; reconstruction régionale (industriel + distribution) validée à 0,2 % près avec ~15-20 j de délai — retenue pour le pipeline réel.
- `pipeline/gas_freshness.py`, `pipeline/weather_forecast.py` (prévisions Open-Meteo, continuité EWMA sans redémarrage), `pipeline/real_forecast.py` (orchestration, prédiction Kalman directe sans corruption d'état sur l'horizon futur), `pipeline/tracking_store.py` (suivi SQLite prévu vs réel), `pipeline/run_daily.py` (point d'entrée CLI).

### Persistance des modèles

- `models/persistence.py`, `train_models.py` : les modèles ne sont plus ré-entraînés à chaque lancement du dashboard.
- Fenêtres d'entraînement différenciées : OLS/SURE ≥ 2020-01-01, Kalman ≥ 2018-01-01 ; jeu de test 2025 conservé pour les 3 premiers onglets.
- Jeu "production" (onglet Pipeline réel) : ré-entraînement incluant 2025.
- Artefacts allégés ~50x (2,4 Go → ~50 Mo) via `remove_data()`, avec extraction préalable des attributs nécessaires (`beta_`, `mse_resid_`, `rsquared_`, `stage1_resid_var_`) pour survivre au pickling.

### Tests

- 20/20 tests passants (`tests/test_features.py`, `test_models.py`, `test_kalman.py`).

### Limites connues (à traiter en v0.2)

- Pas de gestion d'erreur autour de l'appel météo Open-Meteo dans `pipeline/real_forecast.py` (contrairement au fetch gaz) : une panne du fournisseur ferait planter le pipeline réel plutôt que de dégrader proprement.
- Le pipeline réel ne produit aucun intervalle de confiance (ni incertitude modèle, ni propagation de l'incertitude météo), contrairement au backtest du dashboard.
- Pas de réconciliation Provisoire → Définitif dans le suivi de performance (une seule passe de rapprochement).
- Pas d'orchestrateur/cron (le déclenchement quotidien reste manuel, hors périmètre de cette version).
- Découpage régional et carte d'accueil non implémentés (données régionales identifiées et confirmées disponibles, mais aucun modèle ni visualisation construits).
