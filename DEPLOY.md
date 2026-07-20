# Déploiement du dashboard sur Streamlit Community Cloud

Le dashboard lit des artefacts (datasets `data/processed/` + modèles
`data/models/`, ~51 Mo) qui **ne sont pas versionnés dans git** et que le
système de fichiers **éphémère** de Streamlit Cloud ne peut pas héberger
durablement. On les stocke donc sur le **stockage objet S3 (MinIO) de SSP
Cloud**, et l'app les télécharge à son démarrage.

Les 1,3 Go de `data/raw/` ne sont **jamais** envoyés : ils ne servent qu'à
l'actualisation des données, opération réservée à l'admin et exécutée en local.

```
   Local (toi)                    S3 / MinIO (SSP Cloud)          Streamlit Cloud
 ┌──────────────┐   upload_       ┌────────────────────┐  sync    ┌──────────────┐
 │ data/processed│──artifacts.py─▶│ gaz-dashboard/      │──boot──▶ │ dashboard     │
 │ data/models   │                │  data/processed/... │          │ (lecture seule│
 └──────────────┘                │  data/models/...     │          │  + admin gate)│
                                  └────────────────────┘          └──────────────┘
```

---

## Mode local vs mode cloud (automatique)

L'app détecte son mode via la présence d'une section `[s3]` dans `st.secrets` :

| | Local (pas de secrets S3) | Cloud (secrets S3 présents) |
|---|---|---|
| Lecture des données | disque (`data/`) | téléchargées depuis S3 au boot |
| Bouton « Actualiser » | visible | masqué, sauf mot de passe admin |
| Bouton « Prévision réelle » | visible | masqué, sauf mot de passe admin |

En développement local, **rien ne change** : aucun secret, aucun appel réseau.

---

## Étapes de déploiement

### 1. Générer les artefacts (une fois, en local)

Le national existe déjà. Pour le régional (carte de France + pages régionales) :

```bash
python fetch_data.py --only meteo          # complète la météo (dept 45 inclus)
python build_regional_dataset.py           # data/processed/regional/*.parquet
python train_regional_models.py --set both # data/models/regional/*.pkl
```

### 2. Créer le stockage sur SSP Cloud

1. Sur ton datalab Onyxia, ouvre **« Mon compte → Connexion au stockage »**.
2. Note `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`,
   l'`endpoint` (ex. `https://minio.lab.sspcloud.fr`) et ton `bucket`
   (souvent ton nom d'utilisateur).

> ⚠️ **Identifiants temporaires (~7 jours).** Ceux affichés par Onyxia
> expirent : le déploiement tomberait en panne au bout d'une semaine. Pour un
> déploiement durable, crée un **compte de service permanent** dans la console
> MinIO : `Access Keys → Create access key` (laisse vide la date d'expiration),
> et utilise ces clés-là (sans `aws_session_token`).

### 3. Configurer les secrets en local et uploader

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# éditer .streamlit/secrets.toml : endpoint, clés, bucket, prefix, admin_password
python scripts/upload_artifacts.py         # pousse ~51 Mo vers S3
```

`.streamlit/secrets.toml` est **gitignoré** — il ne part jamais sur GitHub.

### 4. Déployer sur Streamlit Community Cloud

1. Pousse le repo sur GitHub (le code, pas les données).
2. Sur <https://share.streamlit.io> : **New app**, pointe sur ce repo,
   fichier principal **`dashboard/app.py`**.
3. **App settings → Secrets** : colle le contenu de ton `secrets.toml`
   (mêmes clés : `admin_password` + section `[s3]`).
4. Deploy. Au premier démarrage, l'app synchronise les artefacts depuis S3.

### 5. Accès admin en ligne

Dans la barre latérale de l'app déployée, ouvre **« 🔒 Accès admin »**, saisis
le `admin_password` : les boutons d'actualisation et de prévision réelle
réapparaissent pour ta session uniquement.

---

## Mettre à jour les données publiées

Après un ré-entraînement local (ou une actualisation) :

```bash
python scripts/upload_artifacts.py
```

puis, dans l'app déployée : **⋮ → Reboot** (pour vider le cache `@st.cache_resource`
et re-synchroniser depuis S3).
