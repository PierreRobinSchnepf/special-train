"""Dashboard de test : prévision J+1 de la consommation de gaz français.

Lancer avec :
    streamlit run dashboard/app.py

Simule la pratique d'une entreprise : à J 17h, on prévoit les heures
17h-23h de J puis 0h-23h de J+1 (31 points), à partir des 3 modèles de la
phase R&D (OLS, SURE statiques ; Kalman-adjusted SUR = "notre prédiction").
Tourne sur la base locale (dataset_final.parquet) : le jour J est choisi
dans l'année de test 2025, pour laquelle les 3 modèles ont déjà un
entraînement/état validé.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.services.model_store import BLOCKS, ModelStore

# ---------------------------------------------------------------------------
# Palette (cf. skill dataviz — thème catégoriel validé, ordre fixe)
# ---------------------------------------------------------------------------
COLOR_ACTUAL = "#0b0b0b"   # encre primaire — la vérité terrain n'est pas une "série" au sens catégoriel
COLOR_KALMAN = "#2a78d6"   # slot 1 (bleu) — notre prédiction
COLOR_OLS = "#eb6834"      # slot 6 (orange)
COLOR_SURE = "#4a3aa7"     # slot 7 (violet)
COLOR_KALMAN_BAND = "rgba(42, 120, 214, 0.15)"

BLOCK_COLORS = {
    "Fond de roulement (beta0)": "#008300",  # slot 2 vert
    "Thermique": "#2a78d6",                    # slot 1 bleu
    "Saisonnier (Fourier)": "#eda100",          # slot 4 jaune
    "Calendaire": "#eb6834",                     # slot 6 orange
}

st.set_page_config(page_title="Prévision gaz J+1", layout="wide", page_icon="🔥")


@st.cache_resource(show_spinner="Entraînement des modèles (OLS, SURE, Kalman)...")
def get_store() -> ModelStore:
    return ModelStore()


store = get_store()

st.title("🔥 Prévision horaire de consommation de gaz — J+1")
st.caption(
    "Dashboard de test sur base locale (2025). À J 17h, on prévoit J[17h-23h] + J+1[0h-23h], "
    "comme le ferait une entreprise avant l'échéance de nomination du lendemain."
)

# ---------------------------------------------------------------------------
# Barre latérale : sélection du jour + what-if
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Paramètres")
    days = store.selectable_days()
    default_idx = min(14, len(days) - 1)
    day_str = st.selectbox("Jour de prévision J (17h)", days, index=default_idx)
    day_j = dt.date.fromisoformat(day_str)
    day_j1 = day_j + dt.timedelta(days=1)
    st.caption(f"Horizon prévu : **{day_j.isoformat()} 17h → {day_j1.isoformat()} 23h**")

    st.divider()
    st.subheader("Simulateur what-if")
    temp_delta = st.slider(
        "Écart de température vs. prévu (°C)", -5.0, 5.0, 0.0, 0.5,
        help="Décale temp_smo/X1_heating/X2_smo_heating sur l'horizon de prévision. "
             "Approximation : ne recalcule pas l'EWMA complet, adaptée à l'exploration "
             "de sensibilité, pas à une prévision opérationnelle.",
    )

    st.divider()
    window_days = st.slider("Fenêtre de suivi (onglet Suivi), en jours", 7, 90, 30, 7)


# ---------------------------------------------------------------------------
# Calculs partagés
# ---------------------------------------------------------------------------
horizon = store.forecast_horizon(day_j, temp_delta)

records = []
for p in horizon:
    ts = pd.Timestamp(p.date) + pd.Timedelta(hours=p.hour)
    records.append({
        "timestamp": ts, "date": p.date, "heure": p.hour,
        "Kalman (notre prédiction)": p.kalman, "kalman_lo": p.kalman_lo, "kalman_hi": p.kalman_hi,
        "OLS": p.ols, "SURE": p.sure, "Réel": p.actual,
        **{f"decomp_{k}": v for k, v in p.decomposition_log.items()},
    })
horizon_df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)

tab_forecast, tab_benchmark, tab_monitoring, tab_real = st.tabs(
    ["📈 Forecast", "🎯 Benchmark", "🩺 Suivi de performance", "🌐 Pipeline réel"]
)

# ---------------------------------------------------------------------------
# Onglet 1 — Forecast
# ---------------------------------------------------------------------------
with tab_forecast:
    st.subheader("Prévision par heure")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=horizon_df["timestamp"], y=horizon_df["kalman_hi"],
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=horizon_df["timestamp"], y=horizon_df["kalman_lo"],
        line=dict(width=0), fill="tonexty", fillcolor=COLOR_KALMAN_BAND,
        showlegend=True, name="IC 95% (Kalman)", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=horizon_df["timestamp"], y=horizon_df["Kalman (notre prédiction)"],
        mode="lines+markers", name="Kalman (notre prédiction)",
        line=dict(color=COLOR_KALMAN, width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=horizon_df["timestamp"], y=horizon_df["OLS"],
        mode="lines", name="OLS (statique)", line=dict(color=COLOR_OLS, width=1.5, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=horizon_df["timestamp"], y=horizon_df["SURE"],
        mode="lines", name="SURE (statique)", line=dict(color=COLOR_SURE, width=1.5, dash="dot"),
    ))
    if horizon_df["Réel"].notna().any():
        fig.add_trace(go.Scatter(
            x=horizon_df["timestamp"], y=horizon_df["Réel"],
            mode="lines", name="Réel (historique, pour référence)",
            line=dict(color=COLOR_ACTUAL, width=1.5, dash="dash"), opacity=0.6,
        ))
    fig.update_layout(
        hovermode="x unified", height=460,
        yaxis_title="y_gas_mw", xaxis_title=None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig, width="stretch")

    with st.expander("Table des valeurs"):
        display_df = horizon_df[["timestamp", "Kalman (notre prédiction)", "kalman_lo", "kalman_hi", "OLS", "SURE", "Réel"]].rename(
            columns={"kalman_lo": "IC bas", "kalman_hi": "IC haut"}
        )
        numeric_cols = display_df.columns.drop("timestamp")
        display_df[numeric_cols] = display_df[numeric_cols].round(0)
        st.dataframe(display_df, width="stretch", hide_index=True)

    st.subheader("Décomposition de la prévision (espace log, additive)")
    st.caption(
        "Le modèle Kalman-adjusted SUR est multiplicatif en niveau mais additif en log : "
        "chaque bloc du Tableau 1 contribue une part au log de la prévision. "
        "Fond de roulement domine (niveau de base) ; les autres blocs sont les écarts autour."
    )
    decomp_fig = go.Figure()
    for block_name, color in BLOCK_COLORS.items():
        col = f"decomp_{block_name}"
        if col in horizon_df.columns:
            decomp_fig.add_trace(go.Bar(
                x=horizon_df["timestamp"], y=horizon_df[col], name=block_name, marker_color=color,
            ))
    decomp_fig.update_layout(
        barmode="relative", height=380, yaxis_title="contribution (log)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(decomp_fig, width="stretch")

# ---------------------------------------------------------------------------
# Onglet 2 — Benchmark
# ---------------------------------------------------------------------------
with tab_benchmark:
    st.subheader(f"Prévision vs réel — {day_j.isoformat()} 17h → {day_j1.isoformat()} 23h")

    bench_df = horizon_df.dropna(subset=["Réel"])
    if bench_df.empty:
        st.warning("Aucune donnée réelle disponible sur cet horizon pour comparer.")
    else:
        fig_b = go.Figure()
        fig_b.add_trace(go.Scatter(
            x=bench_df["timestamp"], y=bench_df["Réel"], mode="lines+markers",
            name="Réel", line=dict(color=COLOR_ACTUAL, width=2.5),
        ))
        fig_b.add_trace(go.Scatter(
            x=bench_df["timestamp"], y=bench_df["Kalman (notre prédiction)"], mode="lines+markers",
            name="Kalman (notre prédiction)", line=dict(color=COLOR_KALMAN, width=2),
        ))
        fig_b.add_trace(go.Scatter(
            x=bench_df["timestamp"], y=bench_df["OLS"], mode="lines",
            name="OLS", line=dict(color=COLOR_OLS, width=1.5, dash="dot"),
        ))
        fig_b.add_trace(go.Scatter(
            x=bench_df["timestamp"], y=bench_df["SURE"], mode="lines",
            name="SURE", line=dict(color=COLOR_SURE, width=1.5, dash="dot"),
        ))
        fig_b.update_layout(
            hovermode="x unified", height=420, yaxis_title="y_gas_mw",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_b, width="stretch")

        from models.metrics import evaluate

        cols = st.columns(3)
        for col, (name, key) in zip(cols, [("Kalman", "Kalman (notre prédiction)"), ("OLS", "OLS"), ("SURE", "SURE")]):
            m = evaluate(bench_df["Réel"], bench_df[key])
            with col:
                st.metric(f"{name} — RMSE", f"{m['rmse']:,.0f} MW")
                st.metric(f"{name} — MAPE", f"{m['mape']:.1f} %")

        st.subheader("Erreur absolue par heure")
        err_fig = go.Figure()
        for name, key, color in [("Kalman", "Kalman (notre prédiction)", COLOR_KALMAN), ("OLS", "OLS", COLOR_OLS), ("SURE", "SURE", COLOR_SURE)]:
            err_fig.add_trace(go.Bar(
                x=bench_df["timestamp"], y=(bench_df[key] - bench_df["Réel"]).abs(),
                name=name, marker_color=color,
            ))
        err_fig.update_layout(barmode="group", height=320, yaxis_title="|erreur| (MW)", margin=dict(t=10, b=10))
        st.plotly_chart(err_fig, width="stretch")

# ---------------------------------------------------------------------------
# Onglet 3 — Suivi de performance glissant
# ---------------------------------------------------------------------------
with tab_monitoring:
    st.subheader(f"Suivi glissant — {window_days} jours se terminant le {day_j.isoformat()}")
    st.caption(
        "Pour chaque jour d de la fenêtre, la prévision des 24h de d telle qu'elle aurait "
        "été faite la veille à 17h (même protocole que l'onglet Forecast), comparée au réel."
    )

    with st.spinner("Calcul du suivi glissant..."):
        perf_df = store.rolling_performance(day_j, window_days=window_days)

    if perf_df.empty:
        st.warning("Pas assez de données pour cette fenêtre.")
    else:
        model_colors = {"Kalman": COLOR_KALMAN, "OLS": COLOR_OLS, "SURE": COLOR_SURE}

        col1, col2 = st.columns(2)
        with col1:
            fig_rmse = go.Figure()
            for name, color in model_colors.items():
                sub = perf_df[perf_df["modele"] == name]
                fig_rmse.add_trace(go.Scatter(x=sub["date"], y=sub["rmse"], mode="lines+markers", name=name, line=dict(color=color)))
            fig_rmse.update_layout(title="RMSE quotidien", height=380, yaxis_title="RMSE (MW)", margin=dict(t=40, b=10))
            st.plotly_chart(fig_rmse, width="stretch")
        with col2:
            fig_mape = go.Figure()
            for name, color in model_colors.items():
                sub = perf_df[perf_df["modele"] == name]
                fig_mape.add_trace(go.Scatter(x=sub["date"], y=sub["mape"], mode="lines+markers", name=name, line=dict(color=color)))
            fig_mape.update_layout(title="MAPE quotidien", height=380, yaxis_title="MAPE (%)", margin=dict(t=40, b=10))
            st.plotly_chart(fig_mape, width="stretch")

        st.subheader("Moyenne sur la fenêtre")
        summary = perf_df.groupby("modele")[["rmse", "mape"]].mean().round(2)
        st.dataframe(summary, width="stretch")

# ---------------------------------------------------------------------------
# Onglet 4 — Pipeline réel (données live, pas un rejeu de l'historique)
# ---------------------------------------------------------------------------
with tab_real:
    st.subheader("Pipeline réel")
    st.caption(
        "Contrairement aux 3 autres onglets (qui rejouent l'historique où tout est déjà connu), "
        "ce pipeline appelle de vraies sources externes à l'instant présent : ODRÉ (reconstruction "
        "régionale industriel + distribution) pour rattraper l'état du modèle, et Open-Meteo pour "
        "la météo à venir. La cible officielle a ~45-50 jours de retard : l'état n'est donc rafraîchi "
        "que jusqu'au **jour G** (~15-20 jours en arrière), puis figé et propagé jusqu'à la prévision."
    )

    if st.button("🚀 Lancer une prévision réelle maintenant (30-60s)"):
        with st.spinner("Récupération ODRÉ régional + Open-Meteo + calcul..."):
            from pipeline.real_forecast import run_real_forecast
            from pipeline.tracking_store import save_forecast

            real_result = run_real_forecast()
            run_id = save_forecast(real_result)
            st.session_state["last_real_run_id"] = run_id
        st.success(
            f"Prévision générée : jour J = {real_result.day_j.isoformat()}, "
            f"jour G (dernière vérité terrain gaz) = {real_result.day_g.isoformat()}."
        )
        for w in real_result.warnings:
            st.warning(w)

    from pipeline.tracking_store import load_all_forecasts, reconcile_with_actuals

    all_real_forecasts = load_all_forecasts()
    if all_real_forecasts.empty:
        st.info("Aucune prévision réelle enregistrée pour l'instant — cliquez sur le bouton ci-dessus.")
    else:
        latest_run_id = all_real_forecasts.sort_values("generated_at")["run_id"].iloc[-1]
        latest = all_real_forecasts[all_real_forecasts["run_id"] == latest_run_id].sort_values("timestamp")

        st.markdown(
            f"**Dernière prévision** — générée le {latest['generated_at'].iloc[0]:%Y-%m-%d %H:%M UTC} "
            f"· jour J = {latest['day_j'].iloc[0]} · jour G = {latest['day_g'].iloc[0]}"
        )

        fig_real = go.Figure()
        fig_real.add_trace(go.Scatter(
            x=latest["timestamp"], y=latest["kalman"], mode="lines+markers",
            name="Kalman (notre prédiction)", line=dict(color=COLOR_KALMAN, width=2.5),
        ))
        fig_real.add_trace(go.Scatter(
            x=latest["timestamp"], y=latest["ols"], mode="lines",
            name="OLS (statique)", line=dict(color=COLOR_OLS, width=1.5, dash="dot"),
        ))
        fig_real.add_trace(go.Scatter(
            x=latest["timestamp"], y=latest["sure"], mode="lines",
            name="SURE (statique)", line=dict(color=COLOR_SURE, width=1.5, dash="dot"),
        ))
        fig_real.update_layout(
            hovermode="x unified", height=420, yaxis_title="y_gas_mw prévu",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_real, width="stretch")

        with st.expander("Table des valeurs (dernier run)"):
            table_real = latest[["timestamp", "heure_locale", "kalman", "ols", "sure", "source_meteo"]].copy()
            table_real[["kalman", "ols", "sure"]] = table_real[["kalman", "ols", "sure"]].round(0)
            st.dataframe(table_real, width="stretch", hide_index=True)

        st.subheader("Suivi réel — prévisions passées vs réel")
        st.caption(
            "Une prévision ne peut être vérifiée qu'une fois la donnée régionale correspondante "
            "publiée (~15-20 jours après le run). Les points encore trop récents sont marqués "
            "'à vérifier'."
        )
        reconciled = reconcile_with_actuals()
        verified = reconciled[~reconciled["a_verifier"]]
        if verified.empty:
            st.info("Pas encore de prévision réconciliable avec des données réelles publiées.")
        else:
            fig_track = go.Figure()
            fig_track.add_trace(go.Scatter(
                x=verified["timestamp"], y=verified["reel"], mode="lines+markers",
                name="Réel", line=dict(color=COLOR_ACTUAL, width=2),
            ))
            fig_track.add_trace(go.Scatter(
                x=verified["timestamp"], y=verified["kalman"], mode="lines+markers",
                name="Kalman (prévu)", line=dict(color=COLOR_KALMAN, width=1.5, dash="dot"),
            ))
            fig_track.update_layout(height=380, yaxis_title="y_gas_mw", margin=dict(t=10, b=10))
            st.plotly_chart(fig_track, width="stretch")

            track_metrics = verified[["erreur_abs_kalman", "erreur_abs_ols", "erreur_abs_sure"]].mean().round(0)
            cols_track = st.columns(3)
            for col, (label, key) in zip(cols_track, [("Kalman", "kalman"), ("OLS", "ols"), ("SURE", "sure")]):
                with col:
                    st.metric(f"{label} — MAE (vérifiées)", f"{track_metrics[f'erreur_abs_{key}']:,.0f} MW")
