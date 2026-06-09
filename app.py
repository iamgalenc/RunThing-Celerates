"""
app.py
======
GPX Running Analytics Dashboard
================================
A complete Streamlit application for:
  - Loading & parsing GPX data (real or synthetic)
  - Interactive route maps, pace/elevation profiles, distributions
  - ML-powered pace prediction (regression) and run classification
  - Hypothetical run inference with adjustable sliders

Run with:
    streamlit run app.py
"""

import sys
import os

# Detect if running in Streamlit Cloud / Deployed environment
IS_CLOUD = (
    os.environ.get("STREAMLIT_SHARING_AUTHOR_REPO") is not None
    or os.environ.get("STREAMLIT_SECRETS_EXISTS") is not None
)

# Ensure local modules are importable regardless of working directory
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_loader import generate_synthetic_runs, parse_gpx_file, load_gpx_folder
from feature_engineering import (
    compute_point_features,
    build_feature_matrix,
    RUN_TYPE_MAP,
    RUN_TYPE_INV,
)
from models import (
    train_pace_regression,
    train_run_classifier,
    predict_pace,
    classify_run,
)
from gpx_exporter import run_to_gpx_bytes, runs_to_zip_bytes

# ---------------------------------------------------------------------------
# GenAI SDK imports
# ---------------------------------------------------------------------------
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# ---------------------------------------------------------------------------
# Page config & theme
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="GPX Running Analytics",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS — dark athletic aesthetic with neon accent
st.markdown("""
<style>
    /* ── Root variables ── */
    :root {
        --accent: #00E5A0;
        --accent2: #FF6B35;
        --bg-card: rgba(255,255,255,0.04);
        --radius: 12px;
    }

    /* ── Hide default Streamlit chrome ── */
    #MainMenu, footer, header {visibility: hidden;}

    /* ── App background ── */
    .stApp {
        background: linear-gradient(135deg, #0D0F14 0%, #111720 100%);
        color: #E8ECF0;
        font-family: 'DM Sans', sans-serif;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #0A0C10 !important;
        border-right: 1px solid rgba(0,229,160,0.12);
    }
    section[data-testid="stSidebar"] * {color: #CDD6E0 !important;}

    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: var(--bg-card);
        border: 1px solid rgba(0,229,160,0.15);
        border-radius: var(--radius);
        padding: 12px 16px;
    }
    [data-testid="metric-container"] label {color: #8899AA !important; font-size: 0.75rem;}
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: var(--accent) !important;
        font-size: 1.6rem !important;
        font-weight: 700;
    }

    /* ── Tabs ── */
    .stTabs [role="tablist"] {border-bottom: 1px solid rgba(0,229,160,0.2);}
    .stTabs [role="tab"][aria-selected="true"] {
        color: var(--accent) !important;
        border-bottom: 2px solid var(--accent) !important;
    }
    .stTabs [role="tab"] {color: #667788 !important;}

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #00E5A0, #00C080);
        color: #0D0F14 !important;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        letter-spacing: 0.03em;
        transition: opacity 0.2s;
    }
    .stButton > button:hover {opacity: 0.88;}

    /* ── Selectboxes & sliders ── */
    .stSelectbox > div > div {
        background: var(--bg-card) !important;
        border: 1px solid rgba(0,229,160,0.2) !important;
        border-radius: 8px;
    }

    /* ── Section headers ── */
    h1 {
        background: linear-gradient(90deg, #00E5A0, #00BFFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -0.03em;
    }
    h2, h3 {color: #DDE6EE !important; font-weight: 700;}

    /* ── Info / success boxes ── */
    .stAlert {border-radius: var(--radius);}

    /* ── Divider ── */
    hr {border-color: rgba(0,229,160,0.12);}

    /* ── Plotly chart backgrounds ── */
    .js-plotly-plot .plotly {background: transparent !important;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Plotly theme helper
# ---------------------------------------------------------------------------

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.03)",
    font=dict(color="#CDD6E0", family="DM Sans, sans-serif"),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)"),
)
ACCENT_COLOR = "#00E5A0"
ACCENT2 = "#FF6B35"
RUN_TYPE_COLORS = {
    "easy":     "#4FC3F7",
    "tempo":    "#FF6B35",
    "interval": "#E040FB",
    "long":     "#00E5A0",
}


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_synthetic_runs():
    return generate_synthetic_runs(n_runs=50)


@st.cache_data(show_spinner=False)
def get_enriched(_raw_df, fname):
    """Enrich a single run's GPX DataFrame with point features."""
    return compute_point_features(_raw_df)


@st.cache_data(show_spinner=False)
def get_feature_matrix(runs_raw):
    return build_feature_matrix(runs_raw)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "runs_raw" not in st.session_state:
    st.session_state.runs_raw = {}
if "reg_result" not in st.session_state:
    st.session_state.reg_result = None
if "clf_result" not in st.session_state:
    st.session_state.clf_result = None
if "last_data_source" not in st.session_state:
    st.session_state.last_data_source = None


# ---------------------------------------------------------------------------
# Sidebar — Data Loading
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🏃 GPX Analytics")
    st.markdown("---")

    st.markdown("### 📂 Data Source")
    data_sources = ["Upload GPX Files", "Synthetic Demo Data"]
    if not IS_CLOUD:
        data_sources.append("Local Folder")
    data_source = st.radio(
        "Choose source",
        data_sources,
        label_visibility="collapsed",
    )

    # Clear loaded runs if data source has changed
    if st.session_state.last_data_source != data_source:
        st.session_state.runs_raw = {}
        st.session_state.last_data_source = data_source

    if data_source == "Synthetic Demo Data":
        if st.button("Generate 50 Synthetic Runs", use_container_width=True):
            with st.spinner("Generating synthetic GPX data…"):
                st.session_state.runs_raw = get_synthetic_runs()
            st.success(f"Loaded {len(st.session_state.runs_raw)} runs!")

    elif data_source == "Upload GPX Files":
        uploaded = st.file_uploader(
            "Upload .gpx files", type=["gpx"], accept_multiple_files=True
        )
        if uploaded:
            # Sync session state runs with currently uploaded files
            uploaded_names = {f.name for f in uploaded}
            
            # Remove deleted runs
            keys_to_remove = [k for k in st.session_state.runs_raw if k not in uploaded_names]
            for k in keys_to_remove:
                del st.session_state.runs_raw[k]
                
            # Add new runs
            new_runs = {}
            for f in uploaded:
                if f.name not in st.session_state.runs_raw:
                    df = parse_gpx_file(f)
                    if df is not None:
                        new_runs[f.name] = df
            if new_runs:
                st.session_state.runs_raw.update(new_runs)
                st.success(f"Loaded {len(new_runs)} file(s).")
        else:
            if st.session_state.runs_raw:
                st.session_state.runs_raw = {}

    elif data_source == "Local Folder":
        if IS_CLOUD:
            st.error("Local Folder loading is disabled in cloud deployments for security and cloud container limitations.")
        else:
            folder = st.text_input("Folder path", placeholder="/path/to/gpx/files")
            if st.button("Load Folder", use_container_width=True) and folder:
                runs = load_gpx_folder(folder)
                if runs:
                    st.session_state.runs_raw.update(runs)
                    st.success(f"Loaded {len(runs)} GPX file(s).")
                else:
                    st.warning("No valid .gpx files found.")

    st.markdown("---")

    # Run type filter
    if st.session_state.runs_raw:
        st.markdown("### 🎯 Filter Runs")
        all_types = ["All", "easy", "tempo", "interval", "long"]
        selected_type = st.selectbox("Run Type", all_types)
    else:
        selected_type = "All"

    # ── Download Section ────────────────────────────────────────────────────
    if st.session_state.runs_raw:
        st.markdown("---")
        st.markdown("### 💾 Export GPX Files")

        dl_runs = st.session_state.runs_raw

        # Download ALL runs as a ZIP
        if len(dl_runs) > 0:
            zip_bytes = runs_to_zip_bytes(dl_runs)
            st.download_button(
                label=f"⬇ Download All ({len(dl_runs)} runs) as ZIP",
                data=zip_bytes,
                file_name="synthetic_gpx_runs.zip",
                mime="application/zip",
                use_container_width=True,
                key="dl_all_zip",
            )

        # Download a single selected run
        st.markdown("**Download single run:**")
        dl_run_name = st.selectbox(
            "Pick run",
            sorted(dl_runs.keys()),
            label_visibility="collapsed",
            key="dl_run_select",
        )
        if dl_run_name and dl_run_name in dl_runs:
            single_bytes = run_to_gpx_bytes(dl_run_name, dl_runs[dl_run_name])
            dl_fname = dl_run_name if dl_run_name.endswith(".gpx") else dl_run_name + ".gpx"
            st.download_button(
                label=f"⬇ Download {dl_run_name}",
                data=single_bytes,
                file_name=dl_fname,
                mime="application/gpx+xml",
                use_container_width=True,
                key="dl_single_gpx",
            )


# ---------------------------------------------------------------------------
# Filter runs_raw by selected type
# ---------------------------------------------------------------------------

runs_raw = st.session_state.runs_raw

if selected_type != "All":
    runs_raw = {k: v for k, v in runs_raw.items() if selected_type in k.lower()}


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("# 🏃 GPX Running Analytics")
st.markdown(
    "**End-to-end pipeline** — parse GPX → engineer features → interactive dashboards → ML models"
)

if not runs_raw:
    st.info("👈 Select a data source in the sidebar to get started.")
    st.stop()

# Build feature matrix (cached by run count to avoid recomputing every render)
feat_df = get_feature_matrix(runs_raw)

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------

total_runs  = len(feat_df)
total_km    = feat_df["total_dist_km"].sum()
avg_pace    = feat_df["avg_pace_min_km"].mean()
total_elev  = feat_df["total_elevation_gain_m"].sum()
total_hrs   = feat_df["total_duration_min"].sum() / 60

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Runs",        f"{total_runs}")
col2.metric("Total Distance",    f"{total_km:.1f} km")
col3.metric("Avg Pace",          f"{avg_pace:.2f} min/km")
col4.metric("Total Elev. Gain",  f"{total_elev:.0f} m")
col5.metric("Total Time",        f"{total_hrs:.1f} hrs")

st.markdown("---")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tabs = st.tabs([
    "🗺️ Route Map",
    "📈 Pace & Elevation",
    "📊 Distributions",
    "🤖 ML: Pace Prediction",
    "🏷️ ML: Run Classifier",
    "🔮 Predict New Run",
    "😴 Fatigue Trends",
    "💬 AI Running Coach",
])


# ============================================================
# TAB 0 — Route Map
# ============================================================
with tabs[0]:
    st.subheader("🗺️ Interactive Route Map")

    run_names = sorted(runs_raw.keys())
    selected_run = st.selectbox("Select Run", run_names, key="map_run")

    enriched = get_enriched(runs_raw[selected_run], selected_run)
    rtype_color = RUN_TYPE_COLORS.get(
        next((t for t in RUN_TYPE_COLORS if t in selected_run.lower()), "easy"), ACCENT_COLOR
    )

    # Route map
    fig_map = go.Figure()
    fig_map.add_trace(go.Scattermapbox(
        lat=enriched["lat"],
        lon=enriched["lon"],
        mode="lines+markers",
        marker=dict(
            size=4,
            color=enriched["pace_min_km"],
            colorscale="RdYlGn_r",
            colorbar=dict(title="Pace<br>(min/km)", thickness=12),
            cmin=enriched["pace_min_km"].quantile(0.05),
            cmax=enriched["pace_min_km"].quantile(0.95),
        ),
        line=dict(width=3, color=rtype_color),
        text=[f"Dist: {d:.2f} km<br>Pace: {p:.2f} min/km<br>Elev: {e:.1f} m"
              for d, p, e in zip(enriched["dist_km"], enriched["pace_min_km"], enriched["elevation"])],
        hoverinfo="text",
        name=selected_run,
    ))
    # Start & end markers
    fig_map.add_trace(go.Scattermapbox(
        lat=[enriched["lat"].iloc[0], enriched["lat"].iloc[-1]],
        lon=[enriched["lon"].iloc[0], enriched["lon"].iloc[-1]],
        mode="markers+text",
        marker=dict(size=14, color=[ACCENT_COLOR, ACCENT2]),
        text=["START", "FINISH"],
        textposition="top right",
        hoverinfo="text",
        name="Waypoints",
    ))

    center_lat = enriched["lat"].mean()
    center_lon = enriched["lon"].mean()
    fig_map.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=13,
        ),
        height=520,
        legend=dict(orientation="h", y=-0.05),
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")},
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Quick stats
    summary_cols = st.columns(4)
    total_d = enriched["dist_km"].iloc[-1]
    total_t = enriched["elapsed_s"].iloc[-1] / 60
    gain    = enriched["elev_gain"].iloc[-1]
    pace    = enriched["pace_min_km"].dropna().mean()
    summary_cols[0].metric("Distance", f"{total_d:.2f} km")
    summary_cols[1].metric("Duration", f"{total_t:.1f} min")
    summary_cols[2].metric("Elev. Gain", f"{gain:.0f} m")
    summary_cols[3].metric("Avg Pace", f"{pace:.2f} min/km")

    # ── Per-run GPX download ──────────────────────────────────────────────
    st.markdown("---")
    dl_col1, dl_col2 = st.columns([2, 3])
    with dl_col1:
        single_gpx = run_to_gpx_bytes(selected_run, runs_raw[selected_run])
        dl_name = selected_run if selected_run.endswith(".gpx") else selected_run + ".gpx"
        st.download_button(
            label=f"⬇ Download this run as GPX",
            data=single_gpx,
            file_name=dl_name,
            mime="application/gpx+xml",
            use_container_width=True,
            key="map_tab_dl_single",
        )
    with dl_col2:
        st.caption(
            f"File: **{dl_name}** · {len(runs_raw[selected_run])} track points · "
            f"Compatible with Garmin, Strava, Komoot, Google Maps"
        )


# ============================================================
# TAB 1 — Pace & Elevation Profile
# ============================================================
with tabs[1]:
    st.subheader("📈 Pace & Elevation Profile")

    run_names = sorted(runs_raw.keys())
    sel2 = st.selectbox("Select Run", run_names, key="profile_run")
    enriched2 = get_enriched(runs_raw[sel2], sel2)

    fig_pe = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Pace vs Distance", "Elevation Profile"),
    )

    # Pace
    fig_pe.add_trace(go.Scatter(
        x=enriched2["dist_km"],
        y=enriched2["pace_min_km"],
        mode="lines",
        line=dict(color="rgba(0,229,160,0.35)", width=1),
        name="Instant Pace",
        showlegend=True,
    ), row=1, col=1)
    fig_pe.add_trace(go.Scatter(
        x=enriched2["dist_km"],
        y=enriched2["rolling_pace"],
        mode="lines",
        line=dict(color=ACCENT_COLOR, width=2.5),
        name="Rolling Pace",
        showlegend=True,
    ), row=1, col=1)

    # Elevation
    fig_pe.add_trace(go.Scatter(
        x=enriched2["dist_km"],
        y=enriched2["elevation"],
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(255,107,53,0.15)",
        line=dict(color=ACCENT2, width=2),
        name="Elevation (m)",
        showlegend=True,
    ), row=2, col=1)

    fig_pe.update_yaxes(title_text="Pace (min/km)", row=1, col=1)
    fig_pe.update_yaxes(title_text="Elevation (m)", row=2, col=1)
    fig_pe.update_xaxes(title_text="Distance (km)", row=2, col=1)
    fig_pe.update_layout(height=560, **PLOTLY_LAYOUT)
    st.plotly_chart(fig_pe, use_container_width=True)

    # Pace variability bar chart
    st.markdown("#### Pace Variability per 1 km Segment")
    enriched2["km_bin"] = enriched2["dist_km"].apply(lambda x: int(x))
    var_df = enriched2.groupby("km_bin")["pace_min_km"].std().reset_index()
    var_df.columns = ["km_segment", "pace_std"]
    fig_var = px.bar(
        var_df, x="km_segment", y="pace_std",
        color="pace_std",
        color_continuous_scale=["#00E5A0", "#FF6B35"],
        labels={"pace_std": "Std Dev (min/km)", "km_segment": "km Segment"},
    )
    fig_var.update_layout(height=280, coloraxis_showscale=False, **PLOTLY_LAYOUT)
    st.plotly_chart(fig_var, use_container_width=True)


# ============================================================
# TAB 2 — Distributions
# ============================================================
with tabs[2]:
    st.subheader("📊 Fleet-wide Distributions")

    col_a, col_b = st.columns(2)

    # Pace distribution by run type
    with col_a:
        st.markdown("#### Pace Distribution by Run Type")
        fig_pace = px.violin(
            feat_df, y="avg_pace_min_km", x="run_type", color="run_type",
            color_discrete_map=RUN_TYPE_COLORS,
            box=True, points="all",
            labels={"avg_pace_min_km": "Avg Pace (min/km)", "run_type": "Run Type"},
        )
        fig_pace.update_layout(showlegend=False, height=350, **PLOTLY_LAYOUT)
        st.plotly_chart(fig_pace, use_container_width=True)

    # Distance distribution
    with col_b:
        st.markdown("#### Distance Distribution")
        fig_dist = px.histogram(
            feat_df, x="total_dist_km", color="run_type",
            color_discrete_map=RUN_TYPE_COLORS,
            barmode="overlay", nbins=15,
            labels={"total_dist_km": "Distance (km)", "run_type": "Run Type"},
        )
        fig_dist.update_layout(height=350, **PLOTLY_LAYOUT)
        st.plotly_chart(fig_dist, use_container_width=True)

    col_c, col_d = st.columns(2)

    # Duration distribution
    with col_c:
        st.markdown("#### Duration Distribution")
        fig_dur = px.box(
            feat_df, x="run_type", y="total_duration_min", color="run_type",
            color_discrete_map=RUN_TYPE_COLORS,
            labels={"total_duration_min": "Duration (min)", "run_type": "Run Type"},
        )
        fig_dur.update_layout(showlegend=False, height=350, **PLOTLY_LAYOUT)
        st.plotly_chart(fig_dur, use_container_width=True)

    # Elevation gain vs distance scatter
    with col_d:
        st.markdown("#### Elevation Gain vs Distance")
        fig_elev = px.scatter(
            feat_df.dropna(subset=["avg_pace_min_km"]),
            x="total_dist_km", y="total_elevation_gain_m",
            color="run_type", size="avg_pace_min_km",
            color_discrete_map=RUN_TYPE_COLORS,
            hover_data=["run_name", "avg_pace_min_km"],
            labels={
                "total_dist_km": "Distance (km)",
                "total_elevation_gain_m": "Elev. Gain (m)",
                "run_type": "Type",
            },
        )
        fig_elev.update_layout(height=350, **PLOTLY_LAYOUT)
        st.plotly_chart(fig_elev, use_container_width=True)

    # Correlation heatmap
    st.markdown("#### Feature Correlation Matrix")
    numeric_cols = [
        "total_dist_km", "total_elevation_gain_m", "avg_pace_min_km",
        "std_pace_min_km", "avg_speed_kmh", "total_duration_min",
        "fatigue_index", "avg_pace_variability",
    ]
    corr = feat_df[numeric_cols].corr().round(2)
    fig_corr = px.imshow(
        corr, text_auto=True,
        color_continuous_scale=["#FF6B35", "#111720", "#00E5A0"],
        zmin=-1, zmax=1,
        aspect="auto",
    )
    fig_corr.update_layout(height=400, **PLOTLY_LAYOUT)
    st.plotly_chart(fig_corr, use_container_width=True)


# ============================================================
# TAB 3 — ML: Pace Prediction
# ============================================================
with tabs[3]:
    st.subheader("🤖 Pace Prediction (Regression)")

    with st.expander("⚙️ Model Parameters", expanded=True):
        col1, col2, col3 = st.columns(3)
        reg_model  = col1.selectbox(
            "Model",
            ["random_forest", "linear_regression", "ridge", "gradient_boosting"],
            key="reg_model"
        )
        test_frac  = col2.slider("Test Split", 0.1, 0.4, 0.25, 0.05, key="reg_split")
        
        # Hyperparameters depending on model selection
        rf_trees = 100
        rf_depth = None
        gb_trees = 100
        gb_depth = 3
        gb_lr = 0.1
        ridge_alpha = 1.0
        
        if reg_model == "random_forest":
            rf_trees = col3.slider("RF Trees", 10, 300, 100, 10, key="reg_trees")
            rf_depth = col1.select_slider("RF Max Depth", [None, 2, 3, 5, 8, 10], value=None, key="reg_depth")
        elif reg_model == "ridge":
            ridge_alpha = col3.slider("Ridge Alpha (Regularisation)", 0.01, 10.0, 1.0, 0.1, key="reg_ridge_alpha")
        elif reg_model == "gradient_boosting":
            gb_trees = col3.slider("GB Trees", 10, 300, 100, 10, key="reg_gb_trees")
            gb_depth = col1.select_slider("GB Max Depth", [2, 3, 4, 5, 6, 8], value=3, key="reg_gb_depth")
            gb_lr = col2.slider("GB Learning Rate", 0.01, 0.5, 0.1, 0.01, key="reg_gb_lr")

    if st.button("🚀 Train Regression Model", use_container_width=True):
        if len(feat_df) < 4:
            st.error("Need at least 4 runs to train. Generate more data.")
        else:
            with st.spinner("Training…"):
                try:
                    result = train_pace_regression(
                        feat_df,
                        model_name=reg_model,
                        rf_n_estimators=rf_trees,
                        rf_max_depth=rf_depth,
                        gb_n_estimators=gb_trees,
                        gb_max_depth=gb_depth,
                        gb_learning_rate=gb_lr,
                        ridge_alpha=ridge_alpha,
                        test_size=test_frac,
                    )
                    st.session_state.reg_result = result
                except Exception as e:
                    st.error(f"Training failed: {e}")

    if st.session_state.reg_result:
        r = st.session_state.reg_result
        st.success(f"✅ Model saved → `{r['model_path']}`")

        # 6-column metrics layout
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("MAE",        f"{r['mae']:.4f} min/km")
        m2.metric("RMSE",       f"{r['rmse']:.4f} min/km")
        m3.metric("R² Score",   f"{r['r2']:.4f}")
        m4.metric("MAPE",       f"{r['mape']*100:.2f}%")
        m5.metric("Train Size", r["train_size"])
        m6.metric("Test Size",  r["test_size_n"])

        col_p, col_f = st.columns([3, 2])

        # Actual vs Predicted scatter
        with col_p:
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=r["y_test"], y=r["y_pred"], mode="markers",
                marker=dict(color=ACCENT_COLOR, size=10, opacity=0.8,
                            line=dict(color="white", width=1)),
                name="Predictions",
            ))
            mn = min(r["y_test"].min(), r["y_pred"].min()) - 0.1
            mx = max(r["y_test"].max(), r["y_pred"].max()) + 0.1
            fig_pred.add_trace(go.Scatter(
                x=[mn, mx], y=[mn, mx],
                mode="lines",
                line=dict(color=ACCENT2, dash="dash"),
                name="Perfect Fit",
            ))
            fig_pred.update_layout(
                title="Actual vs Predicted Pace",
                xaxis_title="Actual (min/km)",
                yaxis_title="Predicted (min/km)",
                height=360,
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_pred, use_container_width=True)

        # Feature importances (RF and GB)
        with col_f:
            if r["feature_importances"] is not None:
                fi = r["feature_importances"].reset_index()
                fi.columns = ["feature", "importance"]
                fig_fi = px.bar(
                    fi, x="importance", y="feature", orientation="h",
                    color="importance", color_continuous_scale=["#111720", ACCENT_COLOR],
                    labels={"importance": "Importance", "feature": ""},
                )
                fig_fi.update_layout(
                    title="Feature Importances",
                    coloraxis_showscale=False,
                    height=360,
                    **PLOTLY_LAYOUT,
                )
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.info("Feature importances available for Random Forest and Gradient Boosting only.")

        # Residual plots side-by-side
        col_res1, col_res2 = st.columns(2)
        residuals = r["y_pred"] - r["y_test"]

        with col_res1:
            fig_res = px.scatter(
                x=r["y_test"], y=residuals,
                labels={"x": "Actual Pace (min/km)", "y": "Residual (min/km)"},
                color_discrete_sequence=[ACCENT_COLOR],
            )
            fig_res.add_hline(y=0, line_dash="dash", line_color=ACCENT2)
            fig_res.update_layout(title="Residuals Scatter Plot", height=300, **PLOTLY_LAYOUT)
            st.plotly_chart(fig_res, use_container_width=True)

        with col_res2:
            fig_hist = px.histogram(
                x=residuals,
                nbins=12,
                labels={"x": "Residual (min/km)"},
                color_discrete_sequence=[ACCENT2],
            )
            fig_hist.update_layout(
                title="Residuals Distribution Histogram",
                xaxis_title="Residual (min/km)",
                yaxis_title="Count",
                height=300,
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_hist, use_container_width=True)


# ============================================================
# TAB 4 — ML: Run Classifier
# ============================================================
with tabs[4]:
    st.subheader("🏷️ Run Type Classifier")

    with st.expander("⚙️ Model Parameters", expanded=True):
        col1, col2, col3 = st.columns(3)
        clf_model  = col1.selectbox(
            "Model",
            ["random_forest", "logistic_regression", "gradient_boosting", "svm", "decision_tree"],
            key="clf_model"
        )
        clf_split  = col2.slider("Test Split", 0.1, 0.4, 0.25, 0.05, key="clf_split")
        
        # Hyperparameters depending on model selection
        rf_trees = 100
        rf_depth = None
        lr_C = 1.0
        gb_trees = 100
        gb_depth = 3
        gb_lr = 0.1
        svm_C = 1.0
        svm_kernel = "rbf"
        dt_depth = None
        
        if clf_model == "random_forest":
            rf_trees = col3.slider("RF Trees", 10, 300, 100, 10, key="clf_trees")
            rf_depth = col1.select_slider("RF Max Depth", [None, 2, 3, 5, 8], value=None, key="clf_depth")
        elif clf_model == "logistic_regression":
            lr_C = col3.slider("LR Regularisation (C)", 0.01, 10.0, 1.0, 0.01, key="lr_C")
        elif clf_model == "gradient_boosting":
            gb_trees = col3.slider("GB Trees", 10, 300, 100, 10, key="clf_gb_trees")
            gb_depth = col1.select_slider("GB Max Depth", [2, 3, 4, 5, 6, 8], value=3, key="clf_gb_depth")
            gb_lr = col2.slider("GB Learning Rate", 0.01, 0.5, 0.1, 0.01, key="clf_gb_lr")
        elif clf_model == "svm":
            svm_C = col3.slider("SVM Regularisation (C)", 0.01, 10.0, 1.0, 0.05, key="clf_svm_C")
            svm_kernel = col1.selectbox("SVM Kernel", ["rbf", "linear", "poly"], index=0, key="clf_svm_kernel")
        elif clf_model == "decision_tree":
            dt_depth = col3.select_slider("DT Max Depth", [None, 2, 3, 5, 8, 10], value=None, key="clf_dt_depth")

    if st.button("🚀 Train Classifier", use_container_width=True):
        if len(feat_df) < 4:
            st.error("Need at least 4 runs to train.")
        else:
            with st.spinner("Training…"):
                try:
                    result = train_run_classifier(
                        feat_df,
                        model_name=clf_model,
                        rf_n_estimators=rf_trees,
                        rf_max_depth=rf_depth,
                        lr_C=lr_C,
                        gb_n_estimators=gb_trees,
                        gb_max_depth=gb_depth,
                        gb_learning_rate=gb_lr,
                        svm_C=svm_C,
                        svm_kernel=svm_kernel,
                        dt_max_depth=dt_depth,
                        test_size=clf_split,
                    )
                    st.session_state.clf_result = result
                except Exception as e:
                    st.error(f"Training failed: {e}")

    if st.session_state.clf_result:
        r = st.session_state.clf_result
        st.success(f"✅ Model saved → `{r['model_path']}`")

        # 6-column metrics layout
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Accuracy",   f"{r['accuracy']*100:.1f}%")
        m2.metric("Macro F1",   f"{r['macro_f1']*100:.1f}%")
        m3.metric("Macro Prec", f"{r['macro_precision']*100:.1f}%")
        m4.metric("Macro Rec",  f"{r['macro_recall']*100:.1f}%")
        m5.metric("Train Size", r["train_size"])
        m6.metric("Test Size",  r["test_size_n"])

        col_cm, col_fi = st.columns([3, 2])

        # Confusion matrix
        with col_cm:
            conf = r["conf_matrix"]
            class_names = r["class_names"]
            fig_cm = px.imshow(
                conf,
                x=class_names, y=class_names,
                text_auto=True,
                color_continuous_scale=["#111720", ACCENT_COLOR],
                labels=dict(x="Predicted", y="Actual", color="Count"),
                aspect="equal",
            )
            fig_cm.update_layout(
                title="Confusion Matrix",
                height=380,
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        # Feature importances (RF, GB, and DT)
        with col_fi:
            if r["feature_importances"] is not None:
                fi = r["feature_importances"].reset_index()
                fi.columns = ["feature", "importance"]
                fig_fi = px.bar(
                    fi, x="importance", y="feature", orientation="h",
                    color="importance",
                    color_continuous_scale=["#111720", ACCENT_COLOR],
                    labels={"importance": "Importance", "feature": ""},
                )
                fig_fi.update_layout(
                    title="Feature Importances",
                    coloraxis_showscale=False,
                    height=380,
                    **PLOTLY_LAYOUT,
                )
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.info("Feature importances available for Random Forest, Gradient Boosting, and Decision Tree only.")

        # Per-class metrics table
        st.markdown("#### Per-class Metrics")
        report = r["class_report"]
        report_rows = []
        for cls in r["class_names"]:
            if cls in report:
                row = report[cls]
                report_rows.append({
                    "Class":     cls,
                    "Precision": f"{row['precision']:.2f}",
                    "Recall":    f"{row['recall']:.2f}",
                    "F1-Score":  f"{row['f1-score']:.2f}",
                    "Support":   int(row["support"]),
                })
        if report_rows:
            st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)

            # Class performance comparison bar chart
            perf_df = pd.DataFrame(report_rows)
            perf_df["Precision"] = perf_df["Precision"].astype(float)
            perf_df["Recall"] = perf_df["Recall"].astype(float)
            perf_df["F1-Score"] = perf_df["F1-Score"].astype(float)
            
            melt_df = perf_df.melt(id_vars="Class", value_vars=["Precision", "Recall", "F1-Score"],
                                   var_name="Metric", value_name="Score")
            
            fig_perf = px.bar(
                melt_df, x="Class", y="Score", color="Metric",
                barmode="group",
                color_discrete_sequence=[ACCENT_COLOR, ACCENT2, "#4FC3F7"],
                labels={"Score": "Value", "Class": "Run Type Class"},
            )
            fig_perf.update_layout(
                title="Class-wise Performance Metrics Comparison",
                yaxis_range=[0, 1.05],
                height=350,
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_perf, use_container_width=True)


# ============================================================
# TAB 5 — Predict New Run
# ============================================================
with tabs[5]:
    st.subheader("🔮 Predict a Hypothetical Run")

    st.markdown("Adjust the sliders below to describe a new run, then get pace prediction and type classification.")

    col1, col2, col3 = st.columns(3)
    hyp_dist   = col1.slider("Distance (km)",          1.0, 42.0, 10.0, 0.5)
    hyp_gain   = col2.slider("Elevation Gain (m)",     0, 1500,   100,  10)
    hyp_type   = col3.selectbox("Run Type", ["easy", "tempo", "interval", "long"])
    hyp_var    = col1.slider("Avg Pace Variability",   0.0, 2.0,   0.3, 0.05)
    hyp_fatigue= col2.slider("Fatigue Index (pace Δ)", -1.0, 2.0,  0.1, 0.1)
    hyp_dur    = col3.slider("Estimated Duration (min)", 10, 300,  60,   5)

    hyp_features = {
        "total_dist_km":           hyp_dist,
        "total_elevation_gain_m":  hyp_gain,
        "elev_gain_per_km":        hyp_gain / hyp_dist,
        "run_type_code":           RUN_TYPE_MAP.get(hyp_type, 0),
        "avg_pace_variability":    hyp_var,
        "fatigue_index":           hyp_fatigue,
        "avg_pace_min_km":         hyp_dur / hyp_dist,  # rough initial estimate
        "std_pace_min_km":         hyp_var,
        "total_duration_min":      hyp_dur,
    }

    col_reg, col_clf = st.columns(2)

    with col_reg:
        st.markdown("#### 📐 Pace Prediction")
        if st.session_state.reg_result:
            r = st.session_state.reg_result
            pred_pace = predict_pace(r["pipeline"], hyp_features, r["features"])
            mins = int(pred_pace)
            secs = int((pred_pace - mins) * 60)
            st.metric("Predicted Avg Pace", f"{mins}:{secs:02d} min/km")

            # Confidence band approximation from training RMSE
            low  = pred_pace - r["rmse"]
            high = pred_pace + r["rmse"]
            st.caption(f"±1 RMSE range: {low:.2f} – {high:.2f} min/km")
        else:
            st.info("Train a regression model in the **ML: Pace Prediction** tab first.")

    with col_clf:
        st.markdown("#### 🏷️ Run Type Classification")
        if st.session_state.clf_result:
            r = st.session_state.clf_result
            pred_label, proba = classify_run(r["pipeline"], hyp_features, r["features"])
            st.metric("Predicted Run Type", pred_label.upper())
            if proba:
                prob_df = pd.DataFrame(
                    list(proba.items()), columns=["Run Type", "Probability"]
                ).sort_values("Probability", ascending=False)
                fig_prob = px.bar(
                    prob_df, x="Run Type", y="Probability",
                    color="Run Type",
                    color_discrete_map=RUN_TYPE_COLORS,
                )
                fig_prob.update_layout(
                    showlegend=False, height=260,
                    yaxis_range=[0, 1],
                    **PLOTLY_LAYOUT,
                )
                st.plotly_chart(fig_prob, use_container_width=True)
        else:
            st.info("Train a classifier in the **ML: Run Classifier** tab first.")

    # Expected pace breakdown
    st.markdown("---")
    st.markdown("#### Estimated Pace Breakdown")
    if st.session_state.reg_result:
        pred_p = predict_pace(
            st.session_state.reg_result["pipeline"],
            hyp_features,
            st.session_state.reg_result["features"],
        )
        # Simulate a pace-over-distance profile
        n = 50
        km_pts = np.linspace(0, hyp_dist, n)
        # Introduce mild fatigue + noise
        sim_pace = (
            pred_p
            + hyp_fatigue * (km_pts / hyp_dist)
            + np.random.default_rng(7).normal(0, hyp_var * 0.5, n)
        ).clip(2, 20)

        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(
            x=km_pts, y=sim_pace,
            mode="lines", fill="tozeroy",
            fillcolor="rgba(0,229,160,0.1)",
            line=dict(color=ACCENT_COLOR, width=2),
            name="Simulated Pace",
        ))
        fig_sim.add_hline(y=pred_p, line_dash="dash", line_color=ACCENT2,
                          annotation_text=f"Avg {pred_p:.2f} min/km")
        fig_sim.update_layout(
            title="Simulated Pace Over Distance",
            xaxis_title="Distance (km)",
            yaxis_title="Pace (min/km)",
            height=300,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_sim, use_container_width=True)


# ============================================================
# TAB 6 — Fatigue Trends
# ============================================================
with tabs[6]:
    st.subheader("😴 Fatigue Trend Analysis")

    # Sort by inferred date order (using run_name ordering since synthetic)
    trend_df = feat_df.copy().reset_index(drop=True)
    trend_df["run_index"] = range(1, len(trend_df) + 1)

    # Rolling 3-run avg pace
    trend_df["rolling_avg_pace"] = trend_df["avg_pace_min_km"].rolling(3, min_periods=1).mean()
    trend_df["rolling_fatigue"]  = trend_df["fatigue_index"].rolling(3, min_periods=1).mean()

    # Pace over time
    fig_trend = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "Avg Pace per Run (+ 3-run rolling mean)",
            "Fatigue Index per Run (+ rolling mean)",
            "Distance per Run",
        ),
    )

    for rt, grp in trend_df.groupby("run_type"):
        color = RUN_TYPE_COLORS.get(rt, "#AABBCC")
        fig_trend.add_trace(go.Scatter(
            x=grp["run_index"], y=grp["avg_pace_min_km"],
            mode="markers", marker=dict(color=color, size=8),
            name=rt, legendgroup=rt, showlegend=True,
        ), row=1, col=1)

    fig_trend.add_trace(go.Scatter(
        x=trend_df["run_index"], y=trend_df["rolling_avg_pace"],
        mode="lines", line=dict(color="white", width=2, dash="dot"),
        name="Rolling Mean Pace", showlegend=True,
    ), row=1, col=1)

    fig_trend.add_trace(go.Bar(
        x=trend_df["run_index"], y=trend_df["fatigue_index"],
        marker_color=np.where(trend_df["fatigue_index"] > 0, ACCENT2, ACCENT_COLOR),
        name="Fatigue Index", showlegend=False,
    ), row=2, col=1)
    fig_trend.add_trace(go.Scatter(
        x=trend_df["run_index"], y=trend_df["rolling_fatigue"],
        mode="lines", line=dict(color="white", width=2, dash="dot"),
        name="Rolling Fatigue", showlegend=False,
    ), row=2, col=1)

    fig_trend.add_trace(go.Bar(
        x=trend_df["run_index"], y=trend_df["total_dist_km"],
        marker_color=[RUN_TYPE_COLORS.get(rt, "#888") for rt in trend_df["run_type"]],
        name="Distance", showlegend=False,
    ), row=3, col=1)

    fig_trend.update_yaxes(title_text="Pace (min/km)", row=1, col=1)
    fig_trend.update_yaxes(title_text="Fatigue Δ", row=2, col=1)
    fig_trend.update_yaxes(title_text="km", row=3, col=1)
    fig_trend.update_xaxes(title_text="Run #", row=3, col=1)
    fig_trend.update_layout(height=660, **PLOTLY_LAYOUT)
    st.plotly_chart(fig_trend, use_container_width=True)

    # Weekly volume
    st.markdown("#### Cumulative Distance Over Runs")
    trend_df["cumulative_km"] = trend_df["total_dist_km"].cumsum()
    fig_cum = go.Figure(go.Scatter(
        x=trend_df["run_index"], y=trend_df["cumulative_km"],
        mode="lines+markers",
        fill="tozeroy",
        fillcolor="rgba(0,229,160,0.08)",
        line=dict(color=ACCENT_COLOR, width=2.5),
        marker=dict(color=ACCENT_COLOR, size=6),
    ))
    fig_cum.update_layout(
        xaxis_title="Run #",
        yaxis_title="Cumulative Distance (km)",
        height=280,
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    # Effort breakdown donut
    st.markdown("#### Training Load Breakdown by Run Type")
    load_df = trend_df.groupby("run_type")["total_dist_km"].sum().reset_index()
    fig_pie = px.pie(
        load_df, names="run_type", values="total_dist_km",
        color="run_type",
        color_discrete_map=RUN_TYPE_COLORS,
        hole=0.55,
    )
    fig_pie.update_layout(height=340, **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")})
    st.plotly_chart(fig_pie, use_container_width=True)


# ============================================================
# TAB 7 — AI Running Coach
# ============================================================
with tabs[7]:
    st.subheader("💬 Chat with Antigravity Pace Coach")
    st.markdown(
        "Ask questions about your pacing, run classifications, fatigue levels, or training recommendations. "
        "The coach has full context of your uploaded runs."
    )

    if not HAS_GENAI:
        st.error("The `google-genai` SDK is not installed. Please check requirements.txt.")
    else:
        # Resolve Gemini API Key
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            try:
                api_key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
            except Exception:
                pass

        if "gemini_api_key" not in st.session_state:
            st.session_state.gemini_api_key = ""

        final_api_key = api_key or st.session_state.gemini_api_key

        # Clear Chat Button and API Key Input side-by-side
        col_clear, col_key = st.columns([1, 3])
        with col_clear:
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.messages = [
                    {
                        "role": "assistant",
                        "content": "Hi! I am **Antigravity Pace Coach** 🏃. I've analyzed your GPX running history and am ready to chat. Ask me about your pacing, run classifications, fatigue levels, or training recommendations!"
                    }
                ]
                st.rerun()

        with col_key:
            if not api_key:
                entered_key = st.text_input(
                    "Enter Gemini API Key to chat:",
                    value=st.session_state.gemini_api_key,
                    type="password",
                    placeholder="AIzaSy...",
                    help="Get an API key from Google AI Studio. It will only be stored in this session.",
                )
                if entered_key != st.session_state.gemini_api_key:
                    st.session_state.gemini_api_key = entered_key
                    st.rerun()
            else:
                st.success("🤖 API Key detected from environment/secrets!")

        if not final_api_key:
            st.info("👈 Please enter a valid Gemini API Key above or set `GEMINI_API_KEY` in your environment/secrets to activate the coach.")
        else:
            # Initialize messages if not present
            if "messages" not in st.session_state:
                st.session_state.messages = [
                    {
                        "role": "assistant",
                        "content": "Hi! I am **Antigravity Pace Coach** 🏃. I've analyzed your GPX running history and am ready to chat. Ask me about your pacing, run classifications, fatigue levels, or training recommendations!"
                    }
                ]

            # Render history
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # Format context table
            def get_runs_context_markdown(df: pd.DataFrame) -> str:
                if df.empty:
                    return "No running data available."
                cols_to_include = [
                    "run_name", "run_type", "total_dist_km", "total_duration_min",
                    "avg_pace_min_km", "total_elevation_gain_m", "fatigue_index", "avg_pace_variability"
                ]
                existing_cols = [c for c in cols_to_include if c in df.columns]
                sub_df = df[existing_cols].copy()
                rename_map = {
                    "run_name": "Run ID / File",
                    "run_type": "Type",
                    "total_dist_km": "Distance (km)",
                    "total_duration_min": "Duration (min)",
                    "avg_pace_min_km": "Avg Pace (min/km)",
                    "total_elevation_gain_m": "Elev Gain (m)",
                    "fatigue_index": "Fatigue Index",
                    "avg_pace_variability": "Pace Var"
                }
                sub_df = sub_df.rename(columns={k: v for k, v in rename_map.items() if k in sub_df.columns})
                for col in sub_df.select_dtypes(include=[np.number]).columns:
                    sub_df[col] = sub_df[col].round(2)
                try:
                    return sub_df.to_markdown(index=False)
                except Exception:
                    headers = list(sub_df.columns)
                    lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]
                    for _, row in sub_df.iterrows():
                        row_str = []
                        for val in row:
                            if isinstance(val, float):
                                row_str.append(f"{val:.2f}")
                            else:
                                row_str.append(str(val))
                        lines.append(" | ".join(row_str))
                    return "\n".join(lines)

            # Chat input
            prompt = st.chat_input("Ask about your runs...")
            if prompt:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Call API
                try:
                    client = genai.Client(api_key=final_api_key)
                    contents = []
                    for m in st.session_state.messages:
                        role = "user" if m["role"] == "user" else "model"
                        contents.append(
                            types.Content(
                                role=role,
                                parts=[types.Part.from_text(text=m["content"])]
                            )
                        )

                    runs_table = get_runs_context_markdown(feat_df)
                    
                    SYSTEM_INSTRUCTION = """You are "Antigravity Pace Coach", an elite AI running coach and GPX analytics expert.
You help runners analyze their GPX workouts, understand their running form (like pace distribution, elevation impacts, and fatigue trends), and guide them with data-backed coaching advice.

You are given a dataset containing summary features of the user's runs:
{runs_table}

Summary Stats:
- Total runs: {total_runs}
- Total distance: {total_km:.2f} km
- Avg pace: {avg_pace:.2f} min/km
- Total elevation gain: {total_elev:.0f} m
- Total training duration: {total_hrs:.1f} hours

When discussing specific runs, refer to their Run ID / File (e.g. run_01_easy.gpx).
Use your running coach expertise to explain concepts:
- **Run Types**: Easy (aerobic base), Tempo (lactate threshold), Interval (VO2 Max, high variability), Long (endurance).
- **Fatigue Index**: The difference between the average pace in the second half of the run vs the first half. A positive index means they slowed down in the second half (possible fatigue or uphill), whereas a negative/zero index means they maintained pace or executed a negative split.
- **Pace Variability**: High variability is expected in interval sessions, whereas steady runs should have low variability.

Always keep your advice encouraging, professional, structure-oriented, and backed by the specific data in the table above. If you notice trends (e.g. slowing down on long runs, or running tempo runs too fast), point them out!
"""
                    sys_inst = SYSTEM_INSTRUCTION.format(
                        runs_table=runs_table,
                        total_runs=total_runs,
                        total_km=total_km,
                        avg_pace=avg_pace,
                        total_elev=total_elev,
                        total_hrs=total_hrs
                    )

                    with st.spinner("Analyzing data and thinking..."):
                        response = client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=contents,
                            config=types.GenerateContentConfig(
                                system_instruction=sys_inst
                            )
                        )

                    assistant_response = response.text
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                    st.rerun()

                except Exception as e:
                    st.error(f"Error calling Gemini API: {e}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#445566;font-size:0.8rem;'>"
    "GPX Running Analytics · Built with Streamlit · scikit-learn · Plotly"
    "</div>",
    unsafe_allow_html=True,
)
