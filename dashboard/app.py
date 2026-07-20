"""French gas consumption forecasting dashboard — national & regional.

Run with:
    streamlit run dashboard/app.py

Home screen = **map of France**: each gas region is colored by the gap
between actual and forecast consumption (Kalman model bias over the test
window). Clicking a region opens its forecast page (same Forecast /
Benchmark / Monitoring tabs as the national view). The sidebar "Scope"
selector performs the same switch. At J 17:00, we forecast J[17-23h] +
J+1[0-23h], as a company would ahead of the next-day deadline.
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
# Palette (cf. dataviz skill — validated categorical + diverging themes)
# ---------------------------------------------------------------------------
COLOR_ACTUAL = "#0b0b0b"
COLOR_KALMAN = "#2a78d6"
COLOR_OLS = "#eb6834"
COLOR_SURE = "#4a3aa7"
COLOR_KALMAN_BAND = "rgba(42, 120, 214, 0.15)"

MODEL_COLORS = {"kalman": COLOR_KALMAN, "ols": COLOR_OLS, "sure": COLOR_SURE}
MODEL_DASH = {"kalman": None, "ols": "dot", "sure": "dot"}

# SEQUENTIAL scale for the error (MAPE): pale = well predicted, deep red =
# large error. Intuitive and accessible (single hue, decreasing luminance).
ERROR_SCALE = [
    [0.0, "#fff7ec"], [0.25, "#fdd49e"], [0.5, "#fdbb84"],
    [0.75, "#ef6548"], [1.0, "#990000"],
]
# DIVERGING scale for the signed bias: under-forecast (actual > predicted) in
# red, over-forecast in blue, 0 (calibrated) in light gray.
BIAS_SCALE = [
    [0.00, "#b2182b"], [0.25, "#ef8a62"], [0.5, "#f7f7f7"],
    [0.75, "#67a9cf"], [1.00, "#2166ac"],
]

BLOCK_COLORS = {
    "Base load (beta0)": "#008300",
    "Thermal": "#2a78d6",
    "Seasonal (Fourier)": "#eda100",
    "Calendar": "#eb6834",
}

st.set_page_config(page_title="Gas forecast — France & regions", layout="wide", page_icon="🔥")


@st.cache_resource(show_spinner="Downloading data from storage…")
def _bootstrap_artifacts() -> dict:
    """When deployed (S3 config present), sync the runtime artifacts from S3
    to the local ephemeral disk. No-op locally. Cached: runs once per
    container."""
    return download_runtime_artifacts()


_bootstrap_artifacts()

_cfg = load_config()
_region_names = {int(k): v for k, v in _cfg["gas_regional"]["regions"].items()}
_scope_options: list[int | None] = [None] + sorted(_region_names)


def _scope_label(code: int | None) -> str:
    return "🇫🇷 National (France)" if code is None else f"{code} — {_region_names[code]}"


@st.cache_resource(show_spinner="Loading models…")
def get_store(region_code: int | None) -> ModelStore:
    return ModelStore(region_code)


@st.cache_data(show_spinner=False)
def get_geojson() -> dict:
    return load_geojson()


def get_error_table() -> pd.DataFrame:
    # not cached: must reflect the metrics after a refresh
    return regional_error_table(_cfg)


# ---------------------------------------------------------------------------
# State: current scope (None = National). Driven by the map and the selector.
# ---------------------------------------------------------------------------
if "scope" not in st.session_state:
    st.session_state.scope = None


def _set_scope(code: int | None) -> None:
    if st.session_state.scope != code:
        # When switching back to national, clear the map's persisted selection
        # so we don't immediately navigate back to the last clicked region.
        if code is None:
            st.session_state.pop("france_map", None)
        st.session_state.scope = code
        st.rerun()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    chosen = st.selectbox(
        "Scope", _scope_options,
        index=_scope_options.index(st.session_state.scope),
        format_func=_scope_label,
    )
    if chosen != st.session_state.scope:
        _set_scope(chosen)

    scope = st.session_state.scope
    store = get_store(scope)

    days = store.selectable_days()
    default_idx = min(14, len(days) - 1)
    day_str = st.selectbox("Forecast day J (5 PM)", days, index=default_idx)
    day_j = dt.date.fromisoformat(day_str)
    day_j1 = day_j + dt.timedelta(days=1)
    st.caption(f"Horizon: **{day_j.isoformat()} 17:00 → {day_j1.isoformat()} 23:00**")

    st.divider()
    temp_delta = st.slider(
        "Temperature what-if (°C)", -5.0, 5.0, 0.0, 0.5,
        help="Shifts temp_smo/X1/X2 over the horizon (approximation, no EWMA recomputation).",
    )
    window_days = st.slider("Monitoring window (days)", 7, 90, 30, 7)

    st.divider()
    # Admin lock: in the cloud, the refresh (heavy download + writes) is
    # hidden until the admin password is entered.
    _is_admin = admin_gate()
    st.subheader("🔄 Gas data")
    st.caption("Regional ODRÉ data is published every ~15-20 days.")
    if _is_admin:
        _do_refresh = st.button("Check & refresh", width="stretch")
    else:
        _do_refresh = False
        st.caption("🔒 Refresh is restricted to the admin.")


# ---------------------------------------------------------------------------
# Data refresh (fetch -> rebuild -> retrain -> tracking)
# ---------------------------------------------------------------------------
if _do_refresh:
    with st.status("Checking ODRÉ data freshness…", expanded=True) as status:
        fresh = check_freshness(_cfg)
        st.write(
            f"Latest day online: **{fresh['source_day']}** · "
            f"latest day built locally: **{fresh['local_day']}**"
        )
        if not fresh["has_new"]:
            status.update(label="Data already up to date ✅", state="complete")
            st.info("No newer data to download.")
        else:
            st.write(f"🆕 **{fresh['gap_days']} days** of new data available — refreshing…")
            try:
                result = run_refresh(_cfg, log=lambda m: st.write(f"• {m}"))
                get_store.clear()
                status.update(
                    label=f"Refreshed through {result['after']} ✅ (models retrained)",
                    state="complete",
                )
                st.session_state["_refreshed"] = True
            except Exception as exc:  # noqa: BLE001 — any error must be shown to the user
                status.update(label="Refresh failed ❌", state="error")
                st.error(f"Error: {exc}")
    if st.session_state.pop("_refreshed", False):
        st.rerun()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
scope = st.session_state.scope
store = get_store(scope)

if scope is not None:
    if st.button("← Back to the map of France"):
        _set_scope(None)

st.title("🔥 Hourly gas consumption forecast")
st.caption(
    f"Scope: **{store.label}**. "
    + ("Click a region on the map to open its forecast."
       if scope is None else "Forecast J[17-23h] + J+1[0-23h].")
)

# ---------------------------------------------------------------------------
# Shared computations (for the forecast tabs) — driven by store.models
# ---------------------------------------------------------------------------
primary_key = store.models[0][0]
horizon = store.forecast_horizon(day_j, temp_delta)

records = []
for p in horizon:
    ts = pd.Timestamp(p.date) + pd.Timedelta(hours=p.hour)
    rec = {"timestamp": ts, "date": p.date, "hour": p.hour, "Actual": p.actual}
    for key, _label in store.models:
        rec[key] = p.preds[key].value
    rec[f"{primary_key}_lo"] = p.preds[primary_key].lo
    rec[f"{primary_key}_hi"] = p.preds[primary_key].hi
    for k, v in p.decomposition_log.items():
        rec[f"decomp_{k}"] = v
    records.append(rec)
horizon_df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)


# ===========================================================================
# Tab rendering
# ===========================================================================
def render_map() -> None:
    st.subheader("Actual vs forecast consumption, by region")
    tbl = get_error_table()
    if tbl.empty:
        st.info("Regional metrics missing — run `python scripts/train_regional_models.py`.")
        return

    metric_label = st.radio(
        "Color by", ["Error (MAPE %)", "Signed bias (%)"], horizontal=True, label_visibility="collapsed",
    )
    if metric_label.startswith("Error"):
        color_col, scale, midpoint = "mape", ERROR_SCALE, None
        rng = [0.0, float(tbl["mape"].max())]
        cbar_title = "MAPE<br>(%)"
        st.caption("Color = forecast error magnitude (Kalman, test window): "
                   "🟡 pale = well predicted, 🔴 deep red = high error. Corsica excluded (no gas grid).")
    else:
        color_col, scale, midpoint = "bias_pct", BIAS_SCALE, 0.0
        bound = float(max(1e-6, tbl["bias_pct"].abs().max()))
        rng = [-bound, bound]
        cbar_title = "Bias<br>(%)"
        st.caption("Color = signed bias: 🔴 under-forecast (actual > predicted) · "
                   "🔵 over-forecast (predicted > actual) · light = calibrated. Corsica excluded.")

    fig = px.choropleth(
        tbl, geojson=get_geojson(), locations="code", featureidkey="properties.code",
        color=color_col, color_continuous_scale=scale, color_continuous_midpoint=midpoint,
        range_color=rng, hover_name="region",
        custom_data=["code", "region", "bias_pct", "mape", "rmse"],
    )
    fig.update_traces(
        marker_line_width=0.6, marker_line_color="white",
        hovertemplate="<b>%{customdata[1]}</b><br>MAPE: %{customdata[3]:.1f}%"
                      "<br>Bias: %{customdata[2]:+.1f}%<br>RMSE: %{customdata[4]:.0f} MW<extra></extra>",
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
        st.markdown("**Ranking (test MAPE)**")
        ranked = tbl.sort_values("mape")[["region", "mape"]].reset_index(drop=True)
        ranked.index += 1
        st.dataframe(
            ranked.rename(columns={"region": "Region", "mape": "MAPE %"}).round({"MAPE %": 1}),
            width="stretch", height=460,
        )
        st.caption("Click a region (map or table) to open its forecast.")
        picked = st.selectbox(
            "Open a region", [None] + list(tbl["code"].astype(int)),
            format_func=lambda c: "—" if c is None else _region_names[c],
            key="map_region_picker",
        )
        if picked is not None:
            _set_scope(int(picked))

    # map click -> navigation
    sel = getattr(event, "selection", None)
    pts = (sel or {}).get("points") if isinstance(sel, dict) else getattr(sel, "points", None)
    if pts:
        loc = pts[0].get("location") or (pts[0].get("customdata") or [None])[0]
        if loc is not None:
            _set_scope(int(loc))


def render_forecast() -> None:
    st.subheader("Hour-by-hour forecast")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=horizon_df["timestamp"], y=horizon_df[f"{primary_key}_hi"],
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=horizon_df["timestamp"], y=horizon_df[f"{primary_key}_lo"],
                             line=dict(width=0), fill="tonexty", fillcolor=COLOR_KALMAN_BAND,
                             showlegend=True, name="95% CI (Kalman)", hoverinfo="skip"))
    for i, (key, label) in enumerate(store.models):
        if i == 0:
            fig.add_trace(go.Scatter(x=horizon_df["timestamp"], y=horizon_df[key], mode="lines+markers",
                                     name=label, line=dict(color=MODEL_COLORS[key], width=2.5)))
        else:
            fig.add_trace(go.Scatter(x=horizon_df["timestamp"], y=horizon_df[key], mode="lines",
                                     name=label, line=dict(color=MODEL_COLORS[key], width=1.5, dash=MODEL_DASH[key])))
    if horizon_df["Actual"].notna().any():
        fig.add_trace(go.Scatter(x=horizon_df["timestamp"], y=horizon_df["Actual"], mode="lines",
                                 name="Actual (reference)", line=dict(color=COLOR_ACTUAL, width=1.5, dash="dash"), opacity=0.6))
    fig.update_layout(hovermode="x unified", height=460, yaxis_title="y_gas_mw",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                      margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")

    with st.expander("Value table"):
        model_cols = [k for k, _ in store.models]
        display_df = horizon_df[["timestamp", *model_cols, f"{primary_key}_lo", f"{primary_key}_hi", "Actual"]].rename(
            columns={k: lbl for k, lbl in store.models} | {f"{primary_key}_lo": "CI low", f"{primary_key}_hi": "CI high"})
        num = display_df.columns.drop("timestamp")
        display_df[num] = display_df[num].round(0)
        st.dataframe(display_df, width="stretch", hide_index=True)

    st.subheader("Forecast decomposition (log space, additive)")
    st.caption("Each Table 1 block contributes a share of the log of the Kalman-SUR forecast.")
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
    st.subheader(f"Forecast vs actual — {day_j.isoformat()} 17:00 → {day_j1.isoformat()} 23:00")
    bench_df = horizon_df.dropna(subset=["Actual"])
    if bench_df.empty:
        st.warning("No actual data available over this horizon.")
        return
    fig_b = go.Figure()
    fig_b.add_trace(go.Scatter(x=bench_df["timestamp"], y=bench_df["Actual"], mode="lines+markers",
                               name="Actual", line=dict(color=COLOR_ACTUAL, width=2.5)))
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
        m = evaluate(bench_df["Actual"], bench_df[key])
        with col:
            st.metric(f"{label.split(' (')[0]} — RMSE", f"{m['rmse']:,.0f} MW")
            st.metric(f"{label.split(' (')[0]} — MAPE", f"{m['mape']:.1f} %")

    st.subheader("Absolute error by hour")
    err_fig = go.Figure()
    for key, label in store.models:
        err_fig.add_trace(go.Bar(x=bench_df["timestamp"], y=(bench_df[key] - bench_df["Actual"]).abs(),
                                 name=label.split(" (")[0], marker_color=MODEL_COLORS[key]))
    err_fig.update_layout(barmode="group", height=320, yaxis_title="|error| (MW)", margin=dict(t=10, b=10))
    st.plotly_chart(err_fig, width="stretch")


def render_monitoring() -> None:
    st.subheader(f"Rolling monitoring — {window_days} days up to {day_j.isoformat()}")
    st.caption("Daily 24-hour forecast as it would have been made the day before at 17:00, vs actual.")
    with st.spinner("Computing the rolling monitoring…"):
        perf_df = store.rolling_performance(day_j, window_days=window_days)
    if perf_df.empty:
        st.warning("Not enough data for this window.")
        return
    model_colors = {label: MODEL_COLORS[key] for key, label in store.models}
    c1, c2 = st.columns(2)
    for col, metric, title, unit in [(c1, "rmse", "Daily RMSE", "RMSE (MW)"), (c2, "mape", "Daily MAPE", "MAPE (%)")]:
        with col:
            f = go.Figure()
            for label, color in model_colors.items():
                sub = perf_df[perf_df["model"] == label]
                f.add_trace(go.Scatter(x=sub["date"], y=sub[metric], mode="lines+markers", name=label, line=dict(color=color)))
            f.update_layout(title=title, height=380, yaxis_title=unit, margin=dict(t=40, b=10))
            st.plotly_chart(f, width="stretch")
    st.subheader("Window average")
    st.dataframe(perf_df.groupby("model")[["rmse", "mape"]].mean().round(2), width="stretch")


def render_tracking() -> None:
    """Model-quality history at each data refresh."""
    st.subheader("Quality tracking over time")
    st.caption("One point per data refresh ('Check & refresh' button): "
               "mean Kalman MAPE over the 12 regions at retraining time.")
    hist = load_tracking(_cfg)
    if not hist:
        st.info("No refresh recorded yet. Click 'Check & refresh' to create a tracking point.")
        return
    h = pd.DataFrame(hist)
    h["refreshed_at"] = pd.to_datetime(h["refreshed_at"])
    f = go.Figure()
    f.add_trace(go.Scatter(x=h["refreshed_at"], y=h["mean_mape_kalman"], mode="lines+markers",
                           line=dict(color=COLOR_KALMAN, width=2)))
    f.update_layout(height=340, yaxis_title="Mean MAPE (%)", xaxis_title=None, margin=dict(t=10, b=10))
    st.plotly_chart(f, width="stretch")
    st.dataframe(h[["refreshed_at", "data_last_day_after", "mean_mape_kalman"]].round(2), width="stretch", hide_index=True)


# --- Tab routing by scope ---------------------------------------------------
if scope is None:
    tabs = st.tabs(["🗺️ Map of France", "📈 Forecast", "🎯 Benchmark",
                    "🩺 Monitoring", "📊 Quality over time", "🌐 Live pipeline"])
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
    tabs = st.tabs(["📈 Forecast", "🎯 Benchmark", "🩺 Monitoring"])
    with tabs[0]:
        render_forecast()
    with tabs[1]:
        render_benchmark()
    with tabs[2]:
        render_monitoring()
    real_tab = None


# ---------------------------------------------------------------------------
# Live pipeline tab (national only: live data)
# ---------------------------------------------------------------------------
if real_tab is not None:
    with real_tab:
        st.subheader("Live pipeline")
        st.caption(
            "Unlike the other tabs (history replay), this pipeline calls real external "
            "sources at the present moment: ODRÉ (regional reconstruction) to catch the model "
            "state up to day G (~15-20 days back), then Open-Meteo for the upcoming weather."
        )
        _can_run_real = admin_unlocked()
        if not _can_run_real:
            st.caption("🔒 Triggering a live forecast (live external calls) is restricted "
                       "to the admin. Previously recorded forecasts remain visible.")
        if _can_run_real and st.button("🚀 Run a live forecast now (30-60s)"):
            with st.spinner("Fetching regional ODRÉ + Open-Meteo + computing…"):
                from pipeline.real_forecast import run_real_forecast
                from pipeline.tracking_store import save_forecast
                real_result = run_real_forecast()
                run_id = save_forecast(real_result)
                st.session_state["last_real_run_id"] = run_id
            st.success(f"Forecast generated: day J = {real_result.day_j.isoformat()}, "
                       f"day G = {real_result.day_g.isoformat()}.")
            for w in real_result.warnings:
                st.warning(w)

        from pipeline.tracking_store import load_all_forecasts, reconcile_with_actuals
        all_real_forecasts = load_all_forecasts()
        if all_real_forecasts.empty:
            st.info("No live forecast recorded — click the button above.")
        else:
            latest_run_id = all_real_forecasts.sort_values("generated_at")["run_id"].iloc[-1]
            latest = all_real_forecasts[all_real_forecasts["run_id"] == latest_run_id].sort_values("timestamp")
            st.markdown(f"**Latest forecast** — generated {latest['generated_at'].iloc[0]:%Y-%m-%d %H:%M UTC} "
                        f"· day J = {latest['day_j'].iloc[0]} · day G = {latest['day_g'].iloc[0]}")
            fig_real = go.Figure()
            for key, label, color, dash in [("kalman", "Kalman (our prediction)", COLOR_KALMAN, None),
                                            ("ols", "OLS (static)", COLOR_OLS, "dot"),
                                            ("sure", "SURE (static)", COLOR_SURE, "dot")]:
                fig_real.add_trace(go.Scatter(x=latest["timestamp"], y=latest[key],
                                              mode="lines+markers" if key == "kalman" else "lines",
                                              name=label, line=dict(color=color, width=2.5 if key == "kalman" else 1.5, dash=dash)))
            fig_real.update_layout(hovermode="x unified", height=420, yaxis_title="forecast y_gas_mw",
                                   legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                                   margin=dict(t=10, b=10))
            st.plotly_chart(fig_real, width="stretch")

            with st.expander("Value table (latest run)"):
                table_real = latest[["timestamp", "local_hour", "kalman", "ols", "sure", "weather_source"]].copy()
                table_real[["kalman", "ols", "sure"]] = table_real[["kalman", "ols", "sure"]].round(0)
                st.dataframe(table_real, width="stretch", hide_index=True)

            st.subheader("Live tracking — past forecasts vs actuals")
            st.caption("A forecast only becomes verifiable once the regional data is published (~15-20 days later).")
            reconciled = reconcile_with_actuals()
            verified = reconciled[~reconciled["pending"]]
            if verified.empty:
                st.info("No forecast reconcilable with published actuals yet.")
            else:
                fig_track = go.Figure()
                fig_track.add_trace(go.Scatter(x=verified["timestamp"], y=verified["actual"], mode="lines+markers",
                                               name="Actual", line=dict(color=COLOR_ACTUAL, width=2)))
                fig_track.add_trace(go.Scatter(x=verified["timestamp"], y=verified["kalman"], mode="lines+markers",
                                               name="Kalman (forecast)", line=dict(color=COLOR_KALMAN, width=1.5, dash="dot")))
                fig_track.update_layout(height=380, yaxis_title="y_gas_mw", margin=dict(t=10, b=10))
                st.plotly_chart(fig_track, width="stretch")
                track_metrics = verified[["abs_error_kalman", "abs_error_ols", "abs_error_sure"]].mean().round(0)
                cols_track = st.columns(3)
                for col, (label, key) in zip(cols_track, [("Kalman", "kalman"), ("OLS", "ols"), ("SURE", "sure")]):
                    with col:
                        st.metric(f"{label} — MAE (verified)", f"{track_metrics[f'abs_error_{key}']:,.0f} MW")
