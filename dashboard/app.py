"""Dashboard de prévision de la consommation de gaz français — national & régional.

Lancer avec :
    streamlit run dashboard/app.py

Écran d'accueil = **carte de France** : chaque région gazière est colorée par
l'écart entre consommation réelle et consommation prévue (biais du modèle Kalman
sur la fenêtre de test). Un clic sur une région ouvre sa page de prévision (mêmes
onglets Forecast / Benchmark / Suivi que le national). Le sélecteur "Périmètre"
de la barre latérale fait la même bascule. À J 17h, on prévoit J[17h-23h] +
J+1[0h-23h], comme le ferait une entreprise avant l'échéance du lendemain.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.services.auth import admin_gate
from dashboard.services.gas_refresh import check_freshness, load_tracking, run_refresh
from dashboard.services.map_data import load_geojson, regional_error_table
from dashboard.services.model_store import BLOCKS, ModelStore
from src.config import load_config
from src.storage import admin_unlocked, download_runtime_artifacts

# ---------------------------------------------------------------------------
# Palette (cf. skill dataviz — thème catégoriel + diverging validés)
# ---------------------------------------------------------------------------
COLOR_ACTUAL = "#0b0b0b"
COLOR_KALMAN = "#2a78d6"
COLOR_OLS = "#eb6834"
COLOR_SURE = "#4a3aa7"
COLOR_KALMAN_BAND = "rgba(42, 120, 214, 0.15)"

MODEL_COLORS = {"kalman": COLOR_KALMAN, "ols": COLOR_OLS, "sure": COLOR_SURE}
MODEL_DASH = {"kalman": None, "ols": "dot", "sure": "dot"}

# Échelle SÉQUENTIELLE pour l'erreur (MAPE) : pâle = bien prédit, rouge profond =
# forte erreur. Intuitif et accessible (une seule teinte, luminance décroissante).
ERROR_SCALE = [
    [0.0, "#fff7ec"], [0.25, "#fdd49e"], [0.5, "#fdbb84"],
    [0.75, "#ef6548"], [1.0, "#990000"],
]
# Échelle DIVERGENTE pour le biais signé : sous-prévision (réel > prévu) en rouge,
# sur-prévision en bleu, 0 (calibré) en clair.
BIAS_SCALE = [
    [0.00, "#b2182b"], [0.25, "#ef8a62"], [0.5, "#f7f7f7"],
    [0.75, "#67a9cf"], [1.00, "#2166ac"],
]

BLOCK_COLORS = {
    "Fond de roulement (beta0)": "#008300",
    "Thermique": "#2a78d6",
    "Saisonnier (Fourier)": "#eda100",
    "Calendaire": "#eb6834",
}

st.set_page_config(page_title="Prévision gaz — France & régions", layout="wide", page_icon="🔥")


@st.cache_resource(show_spinner="Téléchargement des données depuis le stockage…")
def _bootstrap_artifacts() -> dict:
    """En déploiement (config S3 présente), synchronise les artefacts runtime
    depuis S3 vers le disque local éphémère. No-op en local. Caché : ne tourne
    qu'une fois par conteneur."""
    return download_runtime_artifacts()


_bootstrap_artifacts()

_cfg = load_config()
_region_names = {int(k): v for k, v in _cfg["gas_regional"]["regions"].items()}
_scope_options: list[int | None] = [None] + sorted(_region_names)


def _scope_label(code: int | None) -> str:
    return "🇫🇷 National (France)" if code is None else f"{code} — {_region_names[code]}"


@st.cache_resource(show_spinner="Chargement des modèles…")
def get_store(region_code: int | None) -> ModelStore:
    return ModelStore(region_code)


@st.cache_data(show_spinner=False)
def get_geojson() -> dict:
    return load_geojson()


def get_error_table() -> pd.DataFrame:
    # non caché : doit refléter les métriques après une actualisation
    return regional_error_table(_cfg)


# ---------------------------------------------------------------------------
# État : périmètre courant (None = National). La carte et le sélecteur le pilotent.
# ---------------------------------------------------------------------------
if "scope" not in st.session_state:
    st.session_state.scope = None


def _set_scope(code: int | None) -> None:
    if st.session_state.scope != code:
        # En repassant au national, on efface la sélection persistée de la carte
        # pour ne pas re-naviguer aussitôt vers la dernière région cliquée.
        if code is None:
            st.session_state.pop("france_map", None)
        st.session_state.scope = code
        st.rerun()


# ---------------------------------------------------------------------------
# Barre latérale
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Paramètres")

    chosen = st.selectbox(
        "Périmètre", _scope_options,
        index=_scope_options.index(st.session_state.scope),
        format_func=_scope_label,
    )
    if chosen != st.session_state.scope:
        _set_scope(chosen)

    scope = st.session_state.scope
    store = get_store(scope)

    days = store.selectable_days()
    default_idx = min(14, len(days) - 1)
    day_str = st.selectbox("Jour de prévision J (17h)", days, index=default_idx)
    day_j = dt.date.fromisoformat(day_str)
    day_j1 = day_j + dt.timedelta(days=1)
    st.caption(f"Horizon : **{day_j.isoformat()} 17h → {day_j1.isoformat()} 23h**")

    st.divider()
    temp_delta = st.slider(
        "What-if température (°C)", -5.0, 5.0, 0.0, 0.5,
        help="Décale temp_smo/X1/X2 sur l'horizon (approximation, sans recalcul de l'EWMA).",
    )
    window_days = st.slider("Fenêtre de suivi (jours)", 7, 90, 30, 7)

    st.divider()
    # Verrou admin : en cloud, l'actualisation (téléchargement lourd + écriture)
    # est masquée tant que le mot de passe admin n'est pas saisi.
    _is_admin = admin_gate()
    st.subheader("🔄 Données gaz")
    st.caption("Les données régionales ODRÉ se publient ~tous les 15-20 jours.")
    if _is_admin:
        _do_refresh = st.button("Vérifier & actualiser", width="stretch")
    else:
        _do_refresh = False
        st.caption("🔒 Actualisation réservée à l'admin.")


# ---------------------------------------------------------------------------
# Actualisation des données (fetch -> rebuild -> retrain -> tracking)
# ---------------------------------------------------------------------------
if _do_refresh:
    with st.status("Vérification de la fraîcheur des données ODRÉ…", expanded=True) as status:
        fresh = check_freshness(_cfg)
        st.write(
            f"Dernier jour en ligne : **{fresh['source_day']}** · "
            f"dernier jour construit localement : **{fresh['local_day']}**"
        )
        if not fresh["has_new"]:
            status.update(label="Données déjà à jour ✅", state="complete")
            st.info("Aucune donnée plus récente à télécharger.")
        else:
            st.write(f"🆕 **{fresh['gap_days']} jours** de nouvelles données disponibles — actualisation…")
            try:
                result = run_refresh(_cfg, log=lambda m: st.write(f"• {m}"))
                get_store.clear()
                status.update(
                    label=f"Actualisé jusqu'au {result['after']} ✅ (modèles ré-entraînés)",
                    state="complete",
                )
                st.session_state["_refreshed"] = True
            except Exception as exc:  # noqa: BLE001 — on veut afficher toute erreur à l'utilisateur
                status.update(label="Échec de l'actualisation ❌", state="error")
                st.error(f"Erreur : {exc}")
    if st.session_state.pop("_refreshed", False):
        st.rerun()


# ---------------------------------------------------------------------------
# En-tête
# ---------------------------------------------------------------------------
scope = st.session_state.scope
store = get_store(scope)

if scope is not None:
    if st.button("← Retour à la carte de France"):
        _set_scope(None)

st.title("🔥 Prévision horaire de consommation de gaz")
st.caption(
    f"Périmètre : **{store.label}**. "
    + ("Cliquez une région sur la carte pour ouvrir sa prévision."
       if scope is None else "Prévision J[17h-23h] + J+1[0h-23h].")
)

# ---------------------------------------------------------------------------
# Calculs partagés (pour les onglets prévision) — pilotés par store.models
# ---------------------------------------------------------------------------
primary_key = store.models[0][0]
horizon = store.forecast_horizon(day_j, temp_delta)

records = []
for p in horizon:
    ts = pd.Timestamp(p.date) + pd.Timedelta(hours=p.hour)
    rec = {"timestamp": ts, "date": p.date, "heure": p.hour, "Réel": p.actual}
    for key, _label in store.models:
        rec[key] = p.preds[key].value
    rec[f"{primary_key}_lo"] = p.preds[primary_key].lo
    rec[f"{primary_key}_hi"] = p.preds[primary_key].hi
    for k, v in p.decomposition_log.items():
        rec[f"decomp_{k}"] = v
    records.append(rec)
horizon_df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)


# ===========================================================================
# Rendu des onglets
# ===========================================================================
def render_map() -> None:
    st.subheader("Consommation réelle vs prévue, par région")
    tbl = get_error_table()
    if tbl.empty:
        st.info("Métriques régionales absentes — lancez `python train_regional_models.py`.")
        return

    metric_label = st.radio(
        "Colorer par", ["Erreur (MAPE %)", "Biais signé (%)"], horizontal=True, label_visibility="collapsed",
    )
    if metric_label.startswith("Erreur"):
        color_col, scale, midpoint = "mape", ERROR_SCALE, None
        rng = [0.0, float(tbl["mape"].max())]
        cbar_title = "MAPE<br>(%)"
        st.caption("Couleur = ampleur de l'erreur de prévision (Kalman, fenêtre de test) : "
                   "🟡 pâle = bien prédit, 🔴 rouge profond = erreur élevée. Corse exclue (pas de réseau gaz).")
    else:
        color_col, scale, midpoint = "bias_pct", BIAS_SCALE, 0.0
        bound = float(max(1e-6, tbl["bias_pct"].abs().max()))
        rng = [-bound, bound]
        cbar_title = "Biais<br>(%)"
        st.caption("Couleur = biais signé : 🔴 sous-prévision (réel > prévu) · "
                   "🔵 sur-prévision (prévu > réel) · clair = calibré. Corse exclue.")

    fig = px.choropleth(
        tbl, geojson=get_geojson(), locations="code", featureidkey="properties.code",
        color=color_col, color_continuous_scale=scale, color_continuous_midpoint=midpoint,
        range_color=rng, hover_name="region",
        custom_data=["code", "region", "bias_pct", "mape", "rmse"],
    )
    fig.update_traces(
        marker_line_width=0.6, marker_line_color="white",
        hovertemplate="<b>%{customdata[1]}</b><br>MAPE : %{customdata[3]:.1f}%"
                      "<br>Biais : %{customdata[2]:+.1f}%<br>RMSE : %{customdata[4]:.0f} MW<extra></extra>",
    )
    fig.update_geos(visible=False, fitbounds="locations", projection_type="mercator")
    fig.update_layout(
        height=560, margin=dict(t=0, r=0, b=0, l=0), dragmode=False,
        coloraxis_colorbar=dict(title=cbar_title, thickness=14, len=0.7),
    )

    col_map, col_side = st.columns([3, 1], gap="medium")
    with col_map:
        event = st.plotly_chart(
            fig, width="stretch", key="france_map", on_select="rerun",
            selection_mode="points", config={"displayModeBar": False},
        )
    with col_side:
        st.markdown("**Classement (MAPE test)**")
        ranked = tbl.sort_values("mape")[["region", "mape"]].reset_index(drop=True)
        ranked.index += 1
        st.dataframe(
            ranked.rename(columns={"region": "Région", "mape": "MAPE %"}).round({"MAPE %": 1}),
            width="stretch", height=460,
        )
        st.caption("Cliquez une région (carte ou tableau) pour sa prévision.")
        picked = st.selectbox(
            "Ouvrir une région", [None] + list(tbl["code"].astype(int)),
            format_func=lambda c: "—" if c is None else _region_names[c],
            key="map_region_picker",
        )
        if picked is not None:
            _set_scope(int(picked))

    # clic sur la carte -> navigation
    sel = getattr(event, "selection", None)
    pts = (sel or {}).get("points") if isinstance(sel, dict) else getattr(sel, "points", None)
    if pts:
        loc = pts[0].get("location") or (pts[0].get("customdata") or [None])[0]
        if loc is not None:
            _set_scope(int(loc))


def render_forecast() -> None:
    st.subheader("Prévision par heure")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=horizon_df["timestamp"], y=horizon_df[f"{primary_key}_hi"],
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=horizon_df["timestamp"], y=horizon_df[f"{primary_key}_lo"],
                             line=dict(width=0), fill="tonexty", fillcolor=COLOR_KALMAN_BAND,
                             showlegend=True, name="IC 95% (Kalman)", hoverinfo="skip"))
    for i, (key, label) in enumerate(store.models):
        if i == 0:
            fig.add_trace(go.Scatter(x=horizon_df["timestamp"], y=horizon_df[key], mode="lines+markers",
                                     name=label, line=dict(color=MODEL_COLORS[key], width=2.5)))
        else:
            fig.add_trace(go.Scatter(x=horizon_df["timestamp"], y=horizon_df[key], mode="lines",
                                     name=label, line=dict(color=MODEL_COLORS[key], width=1.5, dash=MODEL_DASH[key])))
    if horizon_df["Réel"].notna().any():
        fig.add_trace(go.Scatter(x=horizon_df["timestamp"], y=horizon_df["Réel"], mode="lines",
                                 name="Réel (référence)", line=dict(color=COLOR_ACTUAL, width=1.5, dash="dash"), opacity=0.6))
    fig.update_layout(hovermode="x unified", height=460, yaxis_title="y_gas_mw",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                      margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")

    with st.expander("Table des valeurs"):
        model_cols = [k for k, _ in store.models]
        display_df = horizon_df[["timestamp", *model_cols, f"{primary_key}_lo", f"{primary_key}_hi", "Réel"]].rename(
            columns={k: lbl for k, lbl in store.models} | {f"{primary_key}_lo": "IC bas", f"{primary_key}_hi": "IC haut"})
        num = display_df.columns.drop("timestamp")
        display_df[num] = display_df[num].round(0)
        st.dataframe(display_df, width="stretch", hide_index=True)

    st.subheader("Décomposition de la prévision (espace log, additive)")
    st.caption("Chaque bloc du Tableau 1 contribue une part au log de la prévision Kalman-SUR.")
    decomp_fig = go.Figure()
    for block_name, color in BLOCK_COLORS.items():
        col = f"decomp_{block_name}"
        if col in horizon_df.columns:
            decomp_fig.add_trace(go.Bar(x=horizon_df["timestamp"], y=horizon_df[col], name=block_name, marker_color=color))
    decomp_fig.update_layout(barmode="relative", height=380, yaxis_title="contribution (log)",
                             legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                             margin=dict(t=10, b=10))
    st.plotly_chart(decomp_fig, width="stretch")


def render_benchmark() -> None:
    st.subheader(f"Prévision vs réel — {day_j.isoformat()} 17h → {day_j1.isoformat()} 23h")
    bench_df = horizon_df.dropna(subset=["Réel"])
    if bench_df.empty:
        st.warning("Aucune donnée réelle disponible sur cet horizon.")
        return
    fig_b = go.Figure()
    fig_b.add_trace(go.Scatter(x=bench_df["timestamp"], y=bench_df["Réel"], mode="lines+markers",
                               name="Réel", line=dict(color=COLOR_ACTUAL, width=2.5)))
    for i, (key, label) in enumerate(store.models):
        fig_b.add_trace(go.Scatter(x=bench_df["timestamp"], y=bench_df[key],
                                   mode="lines+markers" if i == 0 else "lines", name=label,
                                   line=dict(color=MODEL_COLORS[key], width=2 if i == 0 else 1.5, dash=MODEL_DASH[key])))
    fig_b.update_layout(hovermode="x unified", height=420, yaxis_title="y_gas_mw",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                        margin=dict(t=10, b=10))
    st.plotly_chart(fig_b, width="stretch")

    from models.metrics import evaluate
    cols = st.columns(len(store.models))
    for col, (key, label) in zip(cols, store.models):
        m = evaluate(bench_df["Réel"], bench_df[key])
        with col:
            st.metric(f"{label.split(' (')[0]} — RMSE", f"{m['rmse']:,.0f} MW")
            st.metric(f"{label.split(' (')[0]} — MAPE", f"{m['mape']:.1f} %")

    st.subheader("Erreur absolue par heure")
    err_fig = go.Figure()
    for key, label in store.models:
        err_fig.add_trace(go.Bar(x=bench_df["timestamp"], y=(bench_df[key] - bench_df["Réel"]).abs(),
                                 name=label.split(" (")[0], marker_color=MODEL_COLORS[key]))
    err_fig.update_layout(barmode="group", height=320, yaxis_title="|erreur| (MW)", margin=dict(t=10, b=10))
    st.plotly_chart(err_fig, width="stretch")


def render_monitoring() -> None:
    st.subheader(f"Suivi glissant — {window_days} jours au {day_j.isoformat()}")
    st.caption("Prévision quotidienne des 24h telle qu'elle aurait été faite la veille à 17h, vs réel.")
    with st.spinner("Calcul du suivi glissant…"):
        perf_df = store.rolling_performance(day_j, window_days=window_days)
    if perf_df.empty:
        st.warning("Pas assez de données pour cette fenêtre.")
        return
    model_colors = {label: MODEL_COLORS[key] for key, label in store.models}
    c1, c2 = st.columns(2)
    for col, metric, title, unit in [(c1, "rmse", "RMSE quotidien", "RMSE (MW)"), (c2, "mape", "MAPE quotidien", "MAPE (%)")]:
        with col:
            f = go.Figure()
            for label, color in model_colors.items():
                sub = perf_df[perf_df["modele"] == label]
                f.add_trace(go.Scatter(x=sub["date"], y=sub[metric], mode="lines+markers", name=label, line=dict(color=color)))
            f.update_layout(title=title, height=380, yaxis_title=unit, margin=dict(t=40, b=10))
            st.plotly_chart(f, width="stretch")
    st.subheader("Moyenne sur la fenêtre")
    st.dataframe(perf_df.groupby("modele")[["rmse", "mape"]].mean().round(2), width="stretch")


def render_tracking() -> None:
    """Historique de qualité des modèles à chaque actualisation des données."""
    st.subheader("Suivi de qualité dans le temps")
    st.caption("Un point par actualisation des données (bouton « Vérifier & actualiser ») : "
               "MAPE Kalman moyenne des 12 régions au moment du ré-entraînement.")
    hist = load_tracking(_cfg)
    if not hist:
        st.info("Aucune actualisation enregistrée. Cliquez « Vérifier & actualiser » pour créer un point de suivi.")
        return
    h = pd.DataFrame(hist)
    h["refreshed_at"] = pd.to_datetime(h["refreshed_at"])
    f = go.Figure()
    f.add_trace(go.Scatter(x=h["refreshed_at"], y=h["mean_mape_kalman"], mode="lines+markers",
                           line=dict(color=COLOR_KALMAN, width=2)))
    f.update_layout(height=340, yaxis_title="MAPE moyenne (%)", xaxis_title=None, margin=dict(t=10, b=10))
    st.plotly_chart(f, width="stretch")
    st.dataframe(h[["refreshed_at", "data_last_day_after", "mean_mape_kalman"]].round(2), width="stretch", hide_index=True)


# --- Aiguillage des onglets selon le périmètre -----------------------------
if scope is None:
    tabs = st.tabs(["🗺️ Carte de France", "📈 Forecast", "🎯 Benchmark",
                    "🩺 Suivi", "📊 Qualité dans le temps", "🌐 Pipeline réel"])
    with tabs[0]:
        render_map()
    with tabs[1]:
        render_forecast()
    with tabs[2]:
        render_benchmark()
    with tabs[3]:
        render_monitoring()
    with tabs[4]:
        render_tracking()
    real_tab = tabs[5]
else:
    tabs = st.tabs(["📈 Forecast", "🎯 Benchmark", "🩺 Suivi"])
    with tabs[0]:
        render_forecast()
    with tabs[1]:
        render_benchmark()
    with tabs[2]:
        render_monitoring()
    real_tab = None


# ---------------------------------------------------------------------------
# Onglet Pipeline réel (national uniquement : données live)
# ---------------------------------------------------------------------------
if real_tab is not None:
    with real_tab:
        st.subheader("Pipeline réel")
        st.caption(
            "Contrairement aux autres onglets (rejeu de l'historique), ce pipeline appelle de vraies "
            "sources externes à l'instant présent : ODRÉ (reconstruction régionale) pour rattraper l'état "
            "du modèle jusqu'au jour G (~15-20 j en arrière), puis Open-Meteo pour la météo à venir."
        )
        _can_run_real = admin_unlocked()
        if not _can_run_real:
            st.caption("🔒 Le déclenchement d'une prévision réelle (appels externes live) "
                       "est réservé à l'admin. Les prévisions déjà enregistrées restent visibles.")
        if _can_run_real and st.button("🚀 Lancer une prévision réelle maintenant (30-60s)"):
            with st.spinner("Récupération ODRÉ régional + Open-Meteo + calcul…"):
                from pipeline.real_forecast import run_real_forecast
                from pipeline.tracking_store import save_forecast
                real_result = run_real_forecast()
                run_id = save_forecast(real_result)
                st.session_state["last_real_run_id"] = run_id
            st.success(f"Prévision générée : jour J = {real_result.day_j.isoformat()}, "
                       f"jour G = {real_result.day_g.isoformat()}.")
            for w in real_result.warnings:
                st.warning(w)

        from pipeline.tracking_store import load_all_forecasts, reconcile_with_actuals
        all_real_forecasts = load_all_forecasts()
        if all_real_forecasts.empty:
            st.info("Aucune prévision réelle enregistrée — cliquez sur le bouton ci-dessus.")
        else:
            latest_run_id = all_real_forecasts.sort_values("generated_at")["run_id"].iloc[-1]
            latest = all_real_forecasts[all_real_forecasts["run_id"] == latest_run_id].sort_values("timestamp")
            st.markdown(f"**Dernière prévision** — générée le {latest['generated_at'].iloc[0]:%Y-%m-%d %H:%M UTC} "
                        f"· jour J = {latest['day_j'].iloc[0]} · jour G = {latest['day_g'].iloc[0]}")
            fig_real = go.Figure()
            for key, label, color, dash in [("kalman", "Kalman (notre prédiction)", COLOR_KALMAN, None),
                                            ("ols", "OLS (statique)", COLOR_OLS, "dot"),
                                            ("sure", "SURE (statique)", COLOR_SURE, "dot")]:
                fig_real.add_trace(go.Scatter(x=latest["timestamp"], y=latest[key],
                                              mode="lines+markers" if key == "kalman" else "lines",
                                              name=label, line=dict(color=color, width=2.5 if key == "kalman" else 1.5, dash=dash)))
            fig_real.update_layout(hovermode="x unified", height=420, yaxis_title="y_gas_mw prévu",
                                   legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                                   margin=dict(t=10, b=10))
            st.plotly_chart(fig_real, width="stretch")

            with st.expander("Table des valeurs (dernier run)"):
                table_real = latest[["timestamp", "heure_locale", "kalman", "ols", "sure", "source_meteo"]].copy()
                table_real[["kalman", "ols", "sure"]] = table_real[["kalman", "ols", "sure"]].round(0)
                st.dataframe(table_real, width="stretch", hide_index=True)

            st.subheader("Suivi réel — prévisions passées vs réel")
            st.caption("Une prévision n'est vérifiable qu'une fois la donnée régionale publiée (~15-20 j après).")
            reconciled = reconcile_with_actuals()
            verified = reconciled[~reconciled["a_verifier"]]
            if verified.empty:
                st.info("Pas encore de prévision réconciliable avec des données réelles publiées.")
            else:
                fig_track = go.Figure()
                fig_track.add_trace(go.Scatter(x=verified["timestamp"], y=verified["reel"], mode="lines+markers",
                                               name="Réel", line=dict(color=COLOR_ACTUAL, width=2)))
                fig_track.add_trace(go.Scatter(x=verified["timestamp"], y=verified["kalman"], mode="lines+markers",
                                               name="Kalman (prévu)", line=dict(color=COLOR_KALMAN, width=1.5, dash="dot")))
                fig_track.update_layout(height=380, yaxis_title="y_gas_mw", margin=dict(t=10, b=10))
                st.plotly_chart(fig_track, width="stretch")
                track_metrics = verified[["erreur_abs_kalman", "erreur_abs_ols", "erreur_abs_sure"]].mean().round(0)
                cols_track = st.columns(3)
                for col, (label, key) in zip(cols_track, [("Kalman", "kalman"), ("OLS", "ols"), ("SURE", "sure")]):
                    with col:
                        st.metric(f"{label} — MAE (vérifiées)", f"{track_metrics[f'erreur_abs_{key}']:,.0f} MW")
