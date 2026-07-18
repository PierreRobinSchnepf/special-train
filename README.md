# Dataset horaire de consommation de gaz français (2018–présent)

Pipeline de construction d'un dataset horaire (France continentale) pour
prédire `y_gas_mw` à partir des variables explicatives du rapport source
(inertie thermique, saisonnalité de Fourier, calendaire, fond de roulement).
Voir `data_dictionary.md` pour le détail colonne par colonne.

## Sources

| Source | Fournisseur | Fréquence brute | Fuseau |
|---|---|---|---|
| Consommation gaz | ODRÉ (`consommation-quotidienne-brute`) | horaire (points :00 seulement, structure demi-horaire héritée de l'électricité) | UTC |
| Température | Météo-France (données climatologiques horaires, par département) | horaire | UTC (convention SYNOP) |
| Jours fériés | data.gouv.fr / etalab | quotidien | Europe/Paris (civil) |
| Vacances scolaires | data.education.gouv.fr (zones A/B/C métropole) | intervalles | Europe/Paris (civil) |

## Installation

```bash
python -m venv .venv
.venv/Scripts/activate      # ou source .venv/bin/activate sous Unix
pip install -r requirements.txt
```

## Usage

```bash
# Ingestion + construction complète (2018 -> aujourd'hui)
python build_dataset.py

# Test rapide sur 1 mois (réutilise le cache si déjà téléchargé)
python build_dataset.py --sample-months 1

# Ré-utiliser le cache local sans re-télécharger
python build_dataset.py --skip-fetch

# Ingestion seule (idempotente, peut être relancée sans risque)
python fetch_data.py --only gas meteo holidays school_holidays
```

Sortie : `data/processed/dataset_final.parquet` + `data/processed/qc_report.json`
(taux de NaN par colonne, doublons, trous dans l'index).

## Décisions de conception documentées

### 1. Fuseau horaire : UTC comme index maître

Le dataset est indexé en **UTC continu, sans trou** (`freq="h"`). C'est un
choix délibéré : les transitions DST d'Europe/Paris (heure dupliquée fin
octobre, heure inexistante fin mars) n'existent pas en UTC, ce qui élimine
structurellement le problème plutôt que de le gérer au cas par cas. Les
variables qui dépendent du calendrier civil français (jour de semaine,
jour de l'année pour Fourier, jours fériés, vacances, fenêtre de fin
d'année) sont calculées en convertissant l'horodatage UTC vers
`Europe/Paris` **uniquement pour en extraire la date locale**, sans jamais
ré-indexer sur cette base. Voir `src/calendar_features.py` et
`src/fourier_features.py`.

Vérifié empiriquement le 2026-07-16 :
- ODRÉ `date_heure` est déjà en UTC (23:30 heure locale Paris ↔ 21:30Z en
  période d'été, écart +2h = CEST — cohérent).
- Météo-France `AAAAMMJJHH` suit la convention SYNOP standard (UTC).

### 2. Endpoint ODRÉ retenu

Le champ demandé dans la spec initiale (`gaz_grtgaz`) n'existe plus : le
schéma a évolué vers un champ agrégé `consommation_brute_gaz_totale`
(GRTgaz + Teréga, France continentale). L'endpoint v1
`/api/records/1.0/download/` **tronque silencieusement autour de 4 Mo**
(vérifié : coupure en plein milieu d'une ligne CSV) — inutilisable tel
quel pour 8 ans d'historique. Le pipeline utilise donc l'endpoint v2
`/exports/csv` avec un filtre `where=date_heure in [date'YYYY-01-01'..date'YYYY-12-31']`,
chunké par année civile, qui ne présente ni la limite offset+limit≤10000
de `/records` ni la troncature de taille de `/download/`.

### 3. Sélection des stations météo et pondération nationale

Le rapport demande une pondération "par population/consommation de gaz par
zone" sans fournir de zonage GRTgaz public. Choix documenté (pas deviné en
silence) : panier de 15 départements représentatifs des grandes zones
climatiques/démographiques de France continentale, pondérés par un ordre de
grandeur de population départementale (INSEE ~2021, `config.yaml §
meteo.stations`). Pour chaque département, le poste météo effectivement
utilisé est choisi **au moment du build**, automatiquement, comme celui
ayant la meilleure couverture non-nulle de `T` — tracé dans
`data/raw/meteo/station_selection.json` à chaque run, pas figé dans le code.

### 4. Ambiguïtés du Tableau 1 — signalées, pas résolues en silence

- **Creux mi-août** : mentionné dans le texte du rapport mais absent du
  Tableau 1. **Non implémenté.** Pas de colonne `is_mid_august`.
- **`is_off_peak_period`** : pas de définition opérationnelle unique dans
  le rapport. Définition **proposée et documentée** :
  `holiday OR any_school_zone_on_break OR is_end_of_year`
  (voir `data_dictionary.md` pour la justification et le chevauchement
  volontaire avec `is_end_of_year`).

### 5. Caractère causal des features

- `X1_heating` dépend uniquement de `T_t` (présent).
- `temp_smo` / `X2_smo_heating` sont une récursion EWMA (`κ=0.98`) qui ne
  regarde que le passé et le présent (`temp_smo_0 = T_0`, pas de
  pré-remplissage à partir du futur).
- Les variables de Fourier et calendaires dépendent uniquement de la date
  du timestamp courant.
- Aucun lissage ou cumul du pipeline ne "voit" une observation future.

## Structure du projet

```
config.yaml               # seule source de vérité pour tout paramètre/seuil/URL
fetch_data.py              # étape 1 : ingestion brute, idempotente, data/raw/
build_dataset.py           # étapes 2->5 : alignement, features, QC, export
src/
  config.py                 # chargement config.yaml
  http_utils.py              # download/get_json avec retry + cache idempotent
  thermal_features.py        # X1_heating, temp_smo, X2_smo_heating
  fourier_features.py        # cos/sin harmoniques 1-4, masquage WD/WE
  calendar_features.py       # is_monday..is_sunday, is_end_of_year, is_off_peak_period
models/                    # modèles de benchmark (voir § Modélisation)
  dataset.py                 # panel jour x heure équilibré, split train/test
  ols.py                     # HourlyOLSModel  : 24 régressions indépendantes
  sure.py                    # HourlySUREModel : système FGLS (Zellner)
  kalman.py                   # HourlyKalmanSURModel : SUR + facteurs d'échelle par Kalman
  metrics.py                  # RMSE / MAPE / MAE, agrégation par heure
notebooks/
  01_exploration_stats.ipynb   # cohérence du dataset (distributions, heatmaps)
  02_benchmark_ols_sure.ipynb  # comparaison OLS vs SURE, train/test, métriques
  03_kalman_sur.ipynb           # SUR ajusté dynamiquement par Kalman vs SUR figé
tests/
  test_features.py            # tests unitaires purs sur les formules du Tableau 1
  test_models.py               # tests OLS/SURE sur données synthétiques
  test_kalman.py                # tests du filtre de Kalman sur données synthétiques
data/raw/                  # cache brut par source (gitignored)
data/processed/            # dataset_final.parquet + qc_report.json (gitignored)
data_dictionary.md         # dictionnaire de données détaillé
```

## Tests

```bash
pytest tests/ -v
```

`test_features.py` couvre : formule EWMA (comparaison à une récursion
manuelle), clipping à 0 des blocs thermiques, réactivité immédiate de `X1`
vs. inertie de `X2` sur un choc de température, masquage WD/WE des
harmoniques de Fourier (y compris le cas limite 23h30 UTC ↔ jour civil
Paris suivant), exclusivité mutuelle des indicatrices de jour, fenêtre de
fin d'année, et composition de `is_off_peak_period`.

`test_models.py` couvre, sur données synthétiques (coefficients connus) :
la récupération des coefficients par OLS, les formes de sortie de SURE,
l'équivalence approximative SURE≈OLS quand les équations sont non
corrélées, et la récupération des vrais coefficients par SURE quand les
résidus sont corrélés entre équations (validation du blanchiment FGLS).

Le pipeline de construction du dataset a été validé sur un sous-échantillon
d'un mois (2019-01) avant le run complet 2018–présent.

## Modélisation : benchmark OLS vs SURE

Les deux modèles de référence (`models/`) découpent la prévision horaire en
**24 équations indépendantes**, une par heure locale Europe/Paris, avec les
mêmes prédicteurs (Tableau 1) :

- **OLS horaire** (`HourlyOLSModel`) : 24 régressions `statsmodels.OLS`
  totalement indépendantes.
- **SURE** (`HourlySUREModel`, Zellner 1962) : le même système, mais estimé
  conjointement par FGLS en exploitant la corrélation contemporaine des
  résidus entre heures d'un même jour (chocs journaliers communs — météo
  fine, comportement inhabituel, jour férié mal capté...). Le blanchiment
  du système est fait à la main (voir docstring de `models/sure.py`) car la
  matrice de covariance complète du système empilé est bien trop grande
  pour être formée en dense ; l'estimation finale passe tout de même par
  `statsmodels.OLS` sur le système transformé — c'est l'algorithme FGLS
  standard de Zellner, juste implémenté sans construire la matrice géante.

Le panel jour x heure est équilibré : les rares jours de transition DST
(heure locale dupliquée ou manquante, ~2/an) sont exclus plutôt que
bricolés, car SURE a besoin d'observations contemporaines alignées entre
les 24 équations pour estimer leur covariance.

`notebooks/02_benchmark_ols_sure.ipynb` compare les deux modèles :
entraînement sur 2018-2024, test sur l'année civile 2025 (dernière année
complète du dataset ; le reliquat partiel 2026 est exclu des deux
ensembles). RMSE/MAPE globaux train/test, RMSE par heure sur le test, et
comparaison graphique prédictions vs. réel sur deux fenêtres de 2 semaines
(janvier et juillet 2025, pour couvrir le régime chauffe et hors-chauffe).

## Modélisation : SUR ajusté dynamiquement par filtre de Kalman

`HourlyKalmanSURModel` (`models/kalman.py`) ajoute un troisième benchmark :
un système SURE dont chaque coefficient reçoit un facteur d'ajustement
multiplicatif dynamique, estimé par filtre de Kalman. Deux sources de
référence, explorées avant l'implémentation :

- **Composition du modèle** — slide "Équation finale : SUR ajusté
  dynamiquement" : `β_true_{t,h,j} = β_SUR_{h,j} · β_Kalman_{t,h,j}`, avec
  `y_{t,h} = β_0 + Σ_j (β_SUR_{h,j} β_Kalman_{t,h,j}) x_{t,h,j} + ε_{t,h}`
  — le SUR garde sa structure explicable, chaque effet peut dériver dans
  le temps via un facteur d'échelle autour de 1.
- **Mécanique du filtre** — notebook de référence
  [`kalman_sur1h.ipynb`](https://github.com/PierreRobinSchnepf/Applied-Statistics-ENGIE/blob/main/notebooks/archive/kalman_sur1h.ipynb) :
  cible en `log1p(y)` / `expm1`, état = marche aléatoire initialisée à 1,
  observation scalaire `y_t = H_t β_t + ε_t` où `H_t` est la contribution
  structurelle SUR (`β_SUR · x`, pas `x` brut), mise à jour de Kalman
  standard (gain, innovation, covariance). Reproduite à l'identique dans
  `models/kalman._run_kalman`.

Écarts assumés par rapport au notebook de référence (documentés dans le
docstring de `models/kalman.py`) :
- l'intercept `β_0` reste **fixe** (pas de facteur Kalman dessus), conforme
  à la formule de la slide — le notebook de référence glissait un
  intercept dynamique dans l'état, absent de cette formule ;
- **24 filtres indépendants** (un par heure), pas un seul (le notebook de
  référence ne traite que l'heure 7) ;
- le bruit d'observation `V_h` est estimé automatiquement à partir de la
  variance résiduelle du SUR en log pour chaque heure, plutôt que recopié
  depuis une constante réglée pour une seule équation d'un autre dataset.

**À propos de l'évaluation** : le filtre prédit *un pas en avant* et
assimile la vraie observation à chaque pas avant de passer au suivant
(comme dans le notebook de référence) — un régime différent d'un OLS/SURE
purement statique qui ne voit jamais la période de test. `notebooks/
03_kalman_sur.ipynb` compare donc le Kalman à un SUR **figé** (facteurs
gardés à 1) évalué selon exactement le même protocole un-pas-en-avant, pas
aux métriques de `02_benchmark_ols_sure.ipynb`. Le notebook trace aussi
l'erreur absolue cumulée sur le test et l'évolution des facteurs
d'ajustement dans le temps.

## Modèles persistés (`train_models.py`, `models/persistence.py`)

Le dashboard et le pipeline réel ne ré-entraînent plus les modèles à chaque
lancement (~40s) — ils chargent des artefacts pré-entraînés depuis
`data/models/*.pkl` (repris depuis un run précédent si présent, sinon
entraînés puis sauvegardés une fois).

```bash
python train_models.py                 # (ré)génère les deux jeux d'artefacts
python train_models.py --only backtest    # ou juste l'un des deux
python train_models.py --only production
```

Deux jeux, deux fenêtres d'entraînement différentes :

| Jeu | OLS / SURE | Kalman | Test réservé | Utilisé par |
|---|---|---|---|---|
| `backtest_*` | [2020-01-01, 2025-01-01) — 2018-2019 exclus, trop anciens | [2018-01-01, 2025-01-01) | 2025 (onglets Forecast/Benchmark/Suivi) | `dashboard/services/model_store.py` |
| `production_*` | [2020-01-01, 2026-01-01) | [2018-01-01, 2026-01-01) | aucun — 2025 inclus dans le train | `pipeline/real_forecast.py` (onglet Pipeline réel) |

À relancer après tout changement du dataset (`build_dataset.py`) ou du code
des modèles. Si un artefact manque, `ModelStore`/`run_real_forecast()`
retombent sur un entraînement à la volée (+sauvegarde) plutôt que d'échouer.

Point d'implémentation à connaître si vous touchez à `models/ols.py` ou
`models/sure.py` : les résultats `statsmodels` embarquent leur matrice de
design complète par défaut (des centaines de Mo une fois picklés pour 24
heures). `models/persistence.py` appelle `remove_data()` dessus avant
sauvegarde pour réduire la taille (~50x), mais ça casse `.params` (perd le
nom des colonnes après un aller-retour pickle) et les propriétés calculées
à la demande (`.mse_resid`, `.rsquared`) si elles n'ont pas déjà été mises
en cache. D'où `HourlyOLSModel.beta_`/`.mse_resid_`/`.rsquared_` et
`HourlySUREModel.stage1_resid_var_` : tout ce dont on a besoin après
chargement est extrait en attributs simples (numpy/dict) **pendant**
`fit()`, jamais relu sur l'objet `statsmodels` après coup.

## Prévision région par région (12 régions gazières)

Extension du modèle national à une prévision **par région**. La cible nationale
`consommation-quotidienne-brute` n'étant pas ventilée par région, la
consommation régionale est reconstituée en sommant deux datasets ODRÉ régionaux
(les mêmes que le pipeline réel, mais en **gardant** la maille région au lieu de
tout sommer) : `conso-journa-industriel-grtgazterega` (industriel) +
`courbe-de-charge-eldgrd-regional-grtgaz-terega` (distribution publique). Somme
des 12 régions ≈ total national à ~0,03 % (2024+).

**Contrainte de données documentée** (`src/regional_gas.py`) : avant **juin
2023**, le profil horaire intra-journalier de ces datasets est déphasé (~+6h ;
pic distribution à 12h UTC en 2018-2022 vs 06h UTC en 2024+), alors que les
totaux journaliers restent corrects. Les modèles horaires régionaux ne
s'entraînent donc que sur l'historique propre `>= gas_regional.hourly_valid_start`
(2023-06-01, `config.yaml`). Un artefact récurrent au samedi 01:00 UTC précédant
la bascule DST de mars (valeurs aberrantes 0→~100 MW) est filtré par un critère
de **chute isolée** (`_mask_isolated_dips` : une heure < 0,5× min(voisins) est un
collapse impossible vu l'inertie thermique) plutôt qu'un seuil de magnitude (le
vrai minimum d'été frôle l'anomalie). Température par région : moyenne pondérée
population des stations Météo-France de la région (`src/regional_meteo.py`,
`region_code` en config ; dept 45/Orléans ajouté pour couvrir Centre-Val de Loire).

```bash
# 1. Ingestion gaz régional (2 datasets ODRÉ, chunké par année)
python fetch_data.py --only gas_regional
# 2. Dataset régional : 12 parquet mono-région, même schéma que le national
python build_regional_dataset.py
# 3. Modèles par région (SURE + Kalman ; OLS exclu), 2 jeux backtest+production
python train_regional_models.py
```

`train_regional_models.py` réutilise **sans les modifier** `HourlySUREModel` et
`HourlyKalmanSURModel`, en boucle sur les 12 régions. Deux jeux d'artefacts par
région dans `data/models/regional/region<code>_<model>_<set>.pkl`
(`set ∈ {backtest, production}`, cf. `regional_artifact_name`). Fenêtres dans
`config.yaml § regional_models` : train 2023-06→2025-07, test 2025-07→2026-07.
Résultats (test) : le Kalman bat le SURE dans les 12 régions (RMSE −16 % en
moyenne) ; somme des 12 prévisions vs national : Kalman MAPE 7,6 %, SURE 12,0 %.
Métriques détaillées dans `data/processed/regional/metrics_regional.json`.

## Dashboard (`dashboard/`)

```bash
streamlit run dashboard/app.py
```

Un sélecteur **Périmètre** (barre latérale) bascule entre le **National** et
chacune des **12 régions** — le même `ModelStore` sert les deux (paramétré par
`region_code`), et les tracés sont pilotés par `store.models`, donc ajouter ou
retirer un modèle ne touche qu'un endroit.

Onglets : **Forecast** (courbe + IC + décomposition + what-if météo),
**Benchmark** (prévu vs réel sur un jour choisi), **Suivi de performance**
(RMSE/MAPE glissants sur fenêtre configurable), et **Pipeline réel** (national
uniquement, cf. ci-dessous). Les 3 premiers rejouent l'année de test (backtest —
tout est déjà connu) : 2025 au national, 2025-07→2026-06 au régional. Le détail
du calcul jour J → J+1 (17h→23h+1) est dans
`dashboard/services/model_store.py`.

## Pipeline réel (`pipeline/`)

Contrairement au dashboard de backtest, ce pipeline appelle de vraies
sources externes à l'instant présent. Recherche de fraîcheur des données
menée avant implémentation (résumé) :

| Source | Périmètre | Retard constaté |
|---|---|---|
| `consommation-quotidienne-brute` (notre cible d'entraînement) | National, total | **~45-50 jours** |
| `conso-journa-industriel-grtgazterega` + `courbe-de-charge-eldgrd-regional-grtgaz-terega` | Régional (13 régions), industriel + distribution | **~15-20 jours** |
| Météo-France (déjà utilisée) | National (15 stations) | quasi temps réel |

Les deux datasets régionaux, sommés sur toutes les régions et les 2
opérateurs (NaTran + Teréga), reconstituent notre cible nationale à ~0.2%
près (vérifié empiriquement) — c'est la meilleure fraîcheur atteignable
pour la cible complète. Point non documenté ailleurs et vérifié
empiriquement : leurs colonnes horaires sont en heure **locale
Europe/Paris** (pas UTC, et pas de décalage "jour gazier" malgré l'ordre
d'export 06h→05h du dataset industriel) — voir `pipeline/gas_freshness.py`.

Comme la météo reste fraîche (quasi temps réel) alors que le gaz plafonne
à ~15-20 jours de retard, le pipeline découple les deux plutôt que d'étaler
toute la prévision sur 2 semaines :
- **L'état du Kalman** n'est réassimilé qu'à partir de la vérité terrain
  disponible, jusqu'au **jour G** (~15-20 jours en arrière) — dataset
  régional ci-dessus.
- **La prévision livrée reste J (17h) → J+1 (23h)**, comme demandé : l'état
  figé au jour G est propagé (marche aléatoire, sans mise à jour, cf.
  `pipeline.real_forecast._direct_kalman_predict`) à travers la météo
  **réelle** pour combler jour G → aujourd'hui, puis la prévision météo
  **Open-Meteo** (modèle Météo-France AROME, gratuit, sans clé — l'API
  officielle Météo-France demande une inscription manuelle qu'un agent ne
  peut pas faire à la place de l'utilisateur) seulement pour
  aujourd'hui → J+1, un horizon largement dans la plage fiable d'une
  prévision courte échéance.

```bash
python -m pipeline.run_daily     # exécute le pipeline et enregistre le résultat
```

Chaque run est persisté dans `data/processed/tracking.sqlite3`
(`pipeline/tracking_store.py`) avec le jour J, le jour G, et la source
météo (observée/prévue) par heure. `reconcile_with_actuals()` rejoint ces
prévisions passées avec les données réelles une fois publiées (~15-20
jours plus tard) pour construire un vrai historique de performance — pas
un backtest rejoué, un vrai suivi de prévisions faites à l'aveugle sur le
futur. Visible dans l'onglet **Pipeline réel** du dashboard.

Non fait ici (hors périmètre d'un dépôt de code) : la configuration d'un
ordonnanceur (cron / tâche planifiée) pour déclencher `run_daily` tous les
jours à 17h — action d'infrastructure sur la machine de déploiement.
