# Dictionnaire de données — `dataset_final.parquet`

Index : horaire, UTC, continu, sans trou (`pandas.date_range(freq="h", tz="UTC")`).
Une ligne = une heure UTC. Les variables calendaires/saisonnières sont calculées
sur la date civile **Europe/Paris** dérivée de cet index (voir `config.yaml §
timezone` et le README pour la justification).

| Colonne | Unité | Source | Formule / définition |
|---|---|---|---|
| `y_gas_mw` | MW PCS 0°C | ODRÉ `consommation-quotidienne-brute`, champ `consommation_brute_gaz_totale` (GRTgaz+Terega, France continentale) | Valeur brute horaire, cible à prédire. |
| `temp_raw_c` | °C | Météo-France, données horaires par département, moyenne pondérée nationale | Moyenne pondérée (poids = population départementale approx. INSEE, cf. `config.yaml § meteo.stations`) des températures `T` des postes sélectionnés, un poste par département retenu automatiquement (meilleure couverture non-nulle sur la période). |
| `X1_heating` | °C (déficit) | Dérivée de `temp_raw_c` | `max(0, T_base - T_t)`, `T_base = 15°C`. Réaction immédiate au froid. |
| `temp_smo` | °C | Dérivée de `temp_raw_c` | `T_smo_t = κ·T_smo_{t-1} + (1-κ)·T_t`, `κ = 0.98`, `T_smo_0 = T_0`. Inertie thermique globale (Dordonnat et al.). |
| `X2_smo_heating` | °C (déficit) | Dérivée de `temp_smo` | `max(0, T_base - T_smo_t)`. Inertie du chauffage. |
| `cos1_WD`…`cos4_WD`, `sin1_WD`…`sin4_WD` | sans unité, [-1,1] | Calendaire (généré) | `cos(2π·s·d/365.25)` / `sin(...)`, `d` = jour de l'année (Europe/Paris), valeur conservée si jour ouvré, **0 sinon**. |
| `cos1_WE`…`cos4_WE`, `sin1_WE`…`sin4_WE` | sans unité, [-1,1] | Calendaire (généré) | Identique, valeur conservée si week-end, **0 sinon**. |
| `is_monday` | {0,1} | Calendaire (généré) | Jour civil Europe/Paris = lundi. |
| `is_friday` | {0,1} | Calendaire (généré) | Jour civil Europe/Paris = vendredi. |
| `is_saturday` | {0,1} | Calendaire (généré) | Jour civil Europe/Paris = samedi. |
| `is_sunday` | {0,1} | Calendaire (généré) | Jour civil Europe/Paris = dimanche. |
| `is_end_of_year` | {0,1} | Calendaire (généré) | Jour civil Europe/Paris ∈ [24 déc., 31 déc.] (fenêtre configurable, `config.yaml § calendar.end_of_year_window`). |
| `is_off_peak_period` | {0,1} | Jours fériés (etalab) + vacances scolaires (data.education.gouv.fr) + `is_end_of_year` | **Définition proposée, non fixée par le rapport source (ambiguïté signalée) :** `holiday OR any_school_zone_on_break OR is_end_of_year`. Zones scolaires A/B/C métropole uniquement. Voir README § Ambiguïtés. |
| `beta_0` | constante | Généré | Toujours 1. Fond de roulement / consommation résiduelle incompressible dans le modèle. |

## Colonnes intermédiaires exclues du Tableau 1 mais conservées pour audit

Aucune — `temp_raw_c` est la seule variable "brute" additionnelle conservée
au-delà du Tableau 1, car `X1_heating` et `X2_smo_heating` en dérivent
directement et sa présence permet de vérifier les deux formules a posteriori.

## Température nationale : méthode d'agrégation

Le rapport source demande une "moyenne pondérée par population/consommation
de gaz par zone" sans fournir de zonage GRTgaz officiel en open data. Choix
documenté : pondération par la population départementale (ordre de grandeur
INSEE ~2021) des 15 départements du panier (`config.yaml § meteo.stations`),
couvrant les principales zones climatiques et démographiques de la France
continentale. Si un département manque à une heure donnée (station hors
ligne au-delà de la tolérance de comblement `max_ffill_gap_hours`), les poids
des départements restants sont renormalisés pour cette heure (loggé en QC).

## Ambiguïtés signalées (non résolues silencieusement)

1. **Creux mi-août** : le texte du rapport source évoque un possible creux de
   consommation mi-août (fermetures industrielles estivales), non repris dans
   le Tableau 1. Aucune colonne n'a été ajoutée pour cet effet — à confirmer
   avant implémentation.
2. **`is_off_peak_period`** : le Tableau 1 ne donne pas de définition
   opérationnelle univoque ("vacances, jours fériés et trêve de fin d'année").
   La définition retenue ci-dessus est un choix explicite et documenté, avec
   chevauchement volontaire avec `is_end_of_year` (les deux colonnes sont
   conservées séparément pour laisser un modèle estimer un effet
   différentiel).

## Traçabilité des sélections de stations météo

Le poste météo retenu par département (parmi tous les postes présents dans
le fichier départemental Météo-France) est choisi automatiquement au moment
du build comme celui ayant le plus d'observations `T` non nulles sur la
période. Le détail par département (poste retenu, nombre d'observations,
couverture temporelle) est écrit dans
`data/raw/meteo/station_selection.json` à chaque exécution de
`build_dataset.py`, pour audit.
