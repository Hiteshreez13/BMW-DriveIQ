"""
BMW DriveIQ — Streamlit Dashboard
====================================
Full interactive dashboard for driver behaviour analysis.
Visualises telemetry, classifications, risk scores, and AI coaching reports.

Run: streamlit run app.py
"""

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import json
from pathlib import Path

from utils.simulator import simulate_trip, PROFILES
from utils.feature_engineering import extract_window_features, WINDOW_SIZE, STEP_SIZE
from utils.coach import generate_report

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BMW DriveIQ",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Orbitron:wght@400;700&display=swap');

    .stApp { background-color: #080808; }
    html, body, [class*="css"] { font-family: 'Rajdhani', sans-serif; }

    h1, h2, h3 { font-family: 'Orbitron', sans-serif !important; }

    .metric-card {
        background: #0f0f0f;
        border: 1px solid #1a1a1a;
        border-left: 3px solid #e5000a;
        padding: 20px 24px;
        border-radius: 2px;
        margin-bottom: 12px;
    }
    .metric-val { font-size: 32px; font-weight: 700; color: #fff; font-family: 'Orbitron', sans-serif; }
    .metric-label { font-size: 11px; color: #888; letter-spacing: 3px; text-transform: uppercase; margin-top: 4px; }

    .risk-HIGH    { color: #e5000a !important; }
    .risk-MEDIUM-HIGH { color: #ff6600 !important; }
    .risk-MEDIUM  { color: #ffcc00 !important; }
    .risk-LOW     { color: #00cc66 !important; }

    .profile-badge {
        display: inline-block;
        padding: 6px 20px;
        background: #e5000a;
        color: white;
        font-family: 'Orbitron', sans-serif;
        font-size: 13px;
        letter-spacing: 2px;
        font-weight: 700;
        clip-path: polygon(8px 0%, 100% 0%, calc(100% - 8px) 100%, 0% 100%);
    }

    .stButton > button {
        background: #e5000a !important;
        color: white !important;
        font-family: 'Orbitron', sans-serif !important;
        letter-spacing: 2px !important;
        border: none !important;
        border-radius: 0 !important;
        font-size: 11px !important;
        padding: 10px 24px !important;
    }
    .stButton > button:hover { background: #ff1a24 !important; }

    .report-box {
        background: #0d0000;
        border: 1px solid #2a0005;
        border-left: 3px solid #e5000a;
        padding: 24px 28px;
        border-radius: 2px;
        font-size: 15px;
        line-height: 1.7;
        color: #ddd;
    }

    div[data-testid="stSidebar"] { background: #080808; border-right: 1px solid #1a1a1a; }
    .css-1d391kg { background: #080808; }

    .stSelectbox > div > div { background: #0f0f0f; border: 1px solid #2a2a2a; color: white; }
    .stSlider > div { color: white; }

    hr { border-color: #1a1a1a; }

    /* ── Tab bar ─────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: transparent;
        border-bottom: 1px solid #2a2a2a;
        padding-bottom: 0;
    }
    .stTabs [data-baseweb="tab"] {
        background: #111;
        border: 1px solid #2a2a2a;
        border-bottom: none;
        border-radius: 4px 4px 0 0;
        color: #aaa !important;
        font-family: 'Orbitron', sans-serif !important;
        font-size: 13px !important;
        font-weight: 600;
        letter-spacing: 1.5px;
        padding: 10px 22px !important;
        transition: all 0.15s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #fff !important;
        background: #1a1a1a;
        border-color: #e5000a;
    }
    .stTabs [aria-selected="true"] {
        background: #1a0002 !important;
        border-color: #e5000a !important;
        border-bottom: 2px solid #e5000a !important;
        color: #fff !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 20px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
LABEL_NAMES = {0: "SMOOTH", 1: "AGGRESSIVE", 2: "FATIGUED", 3: "SPORTY", 4: "HIGHWAY_CRUISE"}
RISK_LEVELS = {"SMOOTH": "LOW", "AGGRESSIVE": "HIGH", "FATIGUED": "MEDIUM-HIGH",
               "SPORTY": "MEDIUM", "HIGHWAY_CRUISE": "LOW"}
RISK_COLORS = {"LOW": "#00cc66", "MEDIUM": "#ffcc00", "MEDIUM-HIGH": "#ff6600", "HIGH": "#e5000a"}
PROFILE_COLORS = {"SMOOTH": "#00cc66", "AGGRESSIVE": "#e5000a", "FATIGUED": "#ff6600",
                  "SPORTY": "#ffcc00", "HIGHWAY_CRUISE": "#4488ff"}

BMW_RED = "#e5000a"
DARK_BG = "#0a0a0a"
CARD_BG = "#0f0f0f"


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not Path("models/driveiq_model.joblib").exists():
        return None, None, None
    model = joblib.load("models/driveiq_model.joblib")
    feature_cols = joblib.load("models/feature_cols.joblib")
    with open("models/metadata.json") as f:
        meta = json.load(f)
    return model, feature_cols, meta


def make_dark_fig():
    return dict(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color="white", family="Rajdhani"),
        xaxis=dict(gridcolor="#1a1a1a", linecolor="#333", zerolinecolor="#333"),
        yaxis=dict(gridcolor="#1a1a1a", linecolor="#333", zerolinecolor="#333"),
        margin=dict(l=40, r=20, t=40, b=40)
    )


# ── Predict on a trip ─────────────────────────────────────────────────────────
def predict_trip_windows(trip_df, model, feature_cols):
    results = []
    for start in range(0, len(trip_df) - WINDOW_SIZE, STEP_SIZE):
        window = trip_df.iloc[start: start + WINDOW_SIZE]
        if len(window) < WINDOW_SIZE:
            continue
        feats = extract_window_features(window)
        X = np.array([[feats.get(col, 0) for col in feature_cols]])
        pred = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]
        results.append({
            "window_start": start,
            "profile": LABEL_NAMES[pred],
            "confidence": float(proba[pred]),
            "risk_score": feats.get("risk_score", 0),
            "stats": feats
        })
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    model, feature_cols, meta = load_model()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 20px 0 30px;'>
            <div style='font-family:Orbitron; font-size:22px; font-weight:900; color:white;'>
                BMW <span style='color:#e5000a;'>DriveIQ</span>
            </div>
            <div style='font-size:10px; letter-spacing:4px; color:#555; margin-top:4px;'>
                DRIVER INTELLIGENCE SYSTEM
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<p style='font-family:Orbitron;font-size:10px;letter-spacing:3px;color:#888;'>SIMULATION CONTROLS</p>", unsafe_allow_html=True)

        profile_select = st.selectbox(
            "Driver Profile",
            options=["🎲 Random"] + list(PROFILES.keys()),
            index=0
        )

        duration = st.slider("Trip Duration (seconds)", 60, 600, 180, step=30)

        run_btn = st.button("▶  ANALYSE DRIVER", use_container_width=True)

        st.markdown("---")

        if meta:
            st.markdown("<p style='font-family:Orbitron;font-size:10px;letter-spacing:3px;color:#888;'>MODEL STATS</p>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-card'><div class='metric-val'>{meta['accuracy']*100:.1f}%</div><div class='metric-label'>Test Accuracy</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-card'><div class='metric-val'>{meta['f1_weighted']:.3f}</div><div class='metric-label'>F1 Score (weighted)</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-card'><div class='metric-val'>{meta['cv_mean']*100:.1f}%</div><div class='metric-label'>CV Accuracy</div></div>", unsafe_allow_html=True)


    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='padding: 30px 0 10px;'>
        <h1 style='font-family:Orbitron; font-size:36px; font-weight:900; margin:0;
                   background: linear-gradient(90deg, #fff, #e5000a);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            BMW DriveIQ
        </h1>
        <p style='color:#666; font-size:13px; letter-spacing:4px; text-transform:uppercase; margin-top:6px;'>
            AI-Powered Driver Behaviour Intelligence System
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # ── No model warning ──────────────────────────────────────────────────────
    if model is None:
        st.error("⚠️ No trained model found. Run `python train.py` in your terminal first.")
        st.code("python train.py", language="bash")
        return

    # ── Run analysis ──────────────────────────────────────────────────────────
    if run_btn or "trip_df" not in st.session_state:
        with st.spinner("Simulating telemetry & running DriveIQ analysis..."):
            chosen = (None if profile_select == "🎲 Random"
                      else profile_select)
            if chosen is None:
                chosen = np.random.choice(list(PROFILES.keys()))

            trip_df = simulate_trip(chosen, duration_seconds=duration)
            results = predict_trip_windows(trip_df, model, feature_cols)

            st.session_state["trip_df"] = trip_df
            st.session_state["results"] = results
            st.session_state["true_profile"] = chosen

    trip_df = st.session_state["trip_df"]
    results = st.session_state["results"]
    true_profile = st.session_state["true_profile"]

    if not results:
        st.error("Not enough data. Increase trip duration.")
        return

    # ── Aggregate ─────────────────────────────────────────────────────────────
    from collections import Counter
    profile_counts = Counter(r["profile"] for r in results)
    dominant = profile_counts.most_common(1)[0][0]
    avg_conf = np.mean([r["confidence"] for r in results])
    avg_risk = np.mean([r["risk_score"] for r in results])
    avg_stats = {}
    for k in results[0]["stats"]:
        avg_stats[k] = round(np.mean([r["stats"][k] for r in results]), 4)

    risk_level = RISK_LEVELS.get(dominant, "UNKNOWN")
    risk_color = RISK_COLORS.get(risk_level, "#888")

    # ── KPI row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (dominant, "DRIVER PROFILE"),
        (f"{avg_conf:.0%}", "CONFIDENCE"),
        (f"{avg_stats.get('speed_mean', 0):.0f} km/h", "AVG SPEED"),
        (f"{avg_stats.get('hard_brake_count', 0):.1f}", "HARD BRAKES/WIN"),
        (risk_level, "RISK LEVEL"),
    ]
    for col, (val, label) in zip([c1, c2, c3, c4, c5], kpis):
        color = risk_color if label == "RISK LEVEL" else ("#e5000a" if label == "DRIVER PROFILE" else "white")
        col.markdown(f"""
        <div class='metric-card'>
            <div class='metric-val' style='font-size:20px; color:{color};'>{val}</div>
            <div class='metric-label'>{label}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📡 Telemetry", "🧠 Classification", "📊 Risk Analysis", "🤖 AI Report", "ℹ️ About"])

    # ── TAB 1: Telemetry ──────────────────────────────────────────────────────
    with tab1:
        st.subheader("Live Telemetry Feed")

        fig = make_subplots(rows=3, cols=2, vertical_spacing=0.08,
                            subplot_titles=["Speed (km/h)", "RPM",
                                            "Throttle & Brake", "Lateral G-Force",
                                            "Steering Angle", "Engine Load (%)"])
        t = trip_df["timestamp"].values

        panels = [
            (trip_df["speed_kmh"], BMW_RED, 1, 1),
            (trip_df["rpm"], "#4488ff", 1, 2),
        ]
        for series, color, row, col in panels:
            fig.add_trace(go.Scatter(x=t, y=series, mode="lines", line=dict(color=color, width=1.5),
                                     showlegend=False), row=row, col=col)

        fig.add_trace(go.Scatter(x=t, y=trip_df["throttle_pos"], mode="lines",
                                  name="Throttle", line=dict(color="#00cc66", width=1.2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=t, y=trip_df["brake_pressure"], mode="lines",
                                  name="Brake", line=dict(color=BMW_RED, width=1.2)), row=2, col=1)

        fig.add_trace(go.Scatter(x=t, y=trip_df["lateral_g"], mode="lines",
                                  line=dict(color="#ffcc00", width=1.2), showlegend=False), row=2, col=2)
        fig.add_trace(go.Scatter(x=t, y=trip_df["steering_angle_deg"], mode="lines",
                                  line=dict(color="#ff6600", width=1.2), showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=t, y=trip_df["engine_load_pct"], mode="lines",
                                  line=dict(color="#aa44ff", width=1.2), showlegend=False), row=3, col=2)

        fig.update_layout(height=600, **make_dark_fig())
        for ann in fig["layout"]["annotations"]:
            ann["font"] = dict(color="#888", size=11, family="Rajdhani")
        st.plotly_chart(fig, use_container_width=True)

    # ── TAB 2: Classification ─────────────────────────────────────────────────
    with tab2:
        col_a, col_b = st.columns([2, 1])

        with col_a:
            st.subheader("Window-Level Classifications")
            window_starts = [r["window_start"] for r in results]
            window_profiles = [r["profile"] for r in results]
            window_confs = [r["confidence"] for r in results]

            colors = [PROFILE_COLORS.get(p, "#888") for p in window_profiles]
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=window_starts, y=window_confs,
                marker_color=colors,
                hovertemplate="<b>t=%{x}s</b><br>Profile: %{customdata}<br>Confidence: %{y:.1%}<extra></extra>",
                customdata=window_profiles
            ))
            fig2.update_layout(
                height=320,
                yaxis_title="Confidence", xaxis_title="Window Start (s)",
                yaxis_range=[0, 1],
                **make_dark_fig()
            )
            st.plotly_chart(fig2, use_container_width=True)

        with col_b:
            st.subheader("Profile Distribution")
            fig3 = go.Figure(go.Pie(
                labels=list(profile_counts.keys()),
                values=list(profile_counts.values()),
                hole=0.55,
                marker_colors=[PROFILE_COLORS.get(p, "#888") for p in profile_counts.keys()],
                textfont=dict(color="white", family="Rajdhani", size=11)
            ))
            fig3.update_layout(
                height=320,
                showlegend=True,
                legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0)"),
                **make_dark_fig()
            )
            st.plotly_chart(fig3, use_container_width=True)

        # Confusion note
        if true_profile:
            match = dominant == true_profile
            if match:
                st.success(f"✅ Correctly classified as **{dominant}** (true profile: {true_profile})")
            else:
                st.warning(f"⚠️ Predicted **{dominant}** | True profile was **{true_profile}** — check distribution above")

    # ── TAB 3: Risk Analysis ──────────────────────────────────────────────────
    with tab3:
        st.subheader("Risk Profile & Safety Signals")

        r_col1, r_col2 = st.columns(2)

        with r_col1:
            # Risk score over time
            risk_scores = [r["risk_score"] for r in results]
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(
                x=list(range(len(risk_scores))), y=risk_scores,
                mode="lines+markers",
                line=dict(color=BMW_RED, width=2),
                fill="tozeroy",
                fillcolor="rgba(229,0,10,0.15)",
                marker=dict(color=[BMW_RED if s > np.percentile(risk_scores, 75) else "#333"
                                   for s in risk_scores], size=5)
            ))
            fig4.add_hline(y=np.percentile(risk_scores, 75), line_dash="dot",
                           line_color="#ff6600", annotation_text="75th percentile",
                           annotation_font_color="#ff6600")
            fig4.update_layout(height=280, yaxis_title="Risk Score",
                               xaxis_title="Window", **make_dark_fig())
            st.plotly_chart(fig4, use_container_width=True)

        with r_col2:
            # Radar chart of key risk dimensions
            dims = ["Speed", "Braking", "Cornering", "Throttle", "Steering"]
            vals = [
                min(avg_stats.get("speed_mean", 0) / 150, 1),
                min(avg_stats.get("hard_brake_count", 0) / 3, 1),
                min(avg_stats.get("lateral_g_max", 0) / 0.8, 1),
                min(avg_stats.get("throttle_high_pct", 0) / 0.5, 1),
                min(avg_stats.get("steering_std", 0) / 20, 1),
            ]
            fig5 = go.Figure(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=dims + [dims[0]],
                fill="toself",
                fillcolor="rgba(229,0,10,0.2)",
                line=dict(color=BMW_RED, width=2),
                marker=dict(color=BMW_RED)
            ))
            fig5.update_layout(
                height=280,
                polar=dict(
                    bgcolor=DARK_BG,
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor="#1a1a1a",
                                    tickfont=dict(color="#555")),
                    angularaxis=dict(gridcolor="#1a1a1a", tickfont=dict(color="white"))
                ),
                **make_dark_fig()
            )
            st.plotly_chart(fig5, use_container_width=True)

        # Key safety metrics
        st.subheader("Safety Metrics Breakdown")
        sm_cols = st.columns(4)
        safety_metrics = [
            ("Hard Brakes / window", f"{avg_stats.get('hard_brake_count', 0):.2f}",
             "#e5000a" if avg_stats.get('hard_brake_count', 0) > 0.5 else "#00cc66"),
            ("Peak Lateral G", f"{avg_stats.get('lateral_g_max', 0):.3f}g",
             "#e5000a" if avg_stats.get('lateral_g_max', 0) > 0.4 else "#00cc66"),
            ("TTC Danger %", f"{avg_stats.get('ttc_low_pct', 0):.1%}",
             "#e5000a" if avg_stats.get('ttc_low_pct', 0) > 0.1 else "#00cc66"),
            ("Peak Jerk", f"{avg_stats.get('jerk_max', 0):.3f} m/s³",
             "#e5000a" if avg_stats.get('jerk_max', 0) > 3 else "#00cc66"),
        ]
        for col, (label, val, color) in zip(sm_cols, safety_metrics):
            col.markdown(f"""
            <div class='metric-card'>
                <div class='metric-val' style='color:{color};font-size:22px;'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>""", unsafe_allow_html=True)

    # ── TAB 4: AI Coaching Report ──────────────────────────────────────────────
    with tab4:
        st.subheader("AI Coaching Report — Powered by Groq")

        dominant_label = [k for k, v in LABEL_NAMES.items() if v == dominant][0]

        if st.button("🤖 Generate Coaching Report", use_container_width=False):
            with st.spinner("Claude is analysing your driving patterns..."):
                report = generate_report(
                    stats=avg_stats,
                    predicted_label=dominant_label,
                    confidence=avg_conf,
                    verbose=False
                )
                st.session_state["report"] = report

        if "report" in st.session_state:
            st.markdown(f"""
            <div class='report-box'>
                {st.session_state['report'].replace(chr(10), '<br>')}
            </div>""", unsafe_allow_html=True)

            st.download_button(
                "📄 Download Report",
                data=st.session_state["report"],
                file_name="bmw_driveiq_report.md",
                mime="text/markdown"
            )
        else:
            # Show stats summary even without API key
            st.markdown(f"""
            <div class='report-box'>
                <strong style='color:#e5000a;font-family:Orbitron;font-size:14px;'>SESSION SUMMARY</strong><br><br>
                <strong>Dominant Profile:</strong> {dominant}<br>
                <strong>Risk Level:</strong> <span class='risk-{risk_level}'>{risk_level}</span><br>
                <strong>Avg Speed:</strong> {avg_stats.get('speed_mean', 0):.1f} km/h<br>
                <strong>Peak Lateral G:</strong> {avg_stats.get('lateral_g_max', 0):.3f}g<br>
                <strong>Hard Brake Events:</strong> {avg_stats.get('hard_brake_count', 0):.2f}/window<br>
                <strong>Risk Score:</strong> {avg_stats.get('risk_score', 0):.2f}<br><br>
                <em style='color:#555;'>Add your Groq API key in the sidebar for the full AI coaching report.</em>
            </div>
            """, unsafe_allow_html=True)

    # ── TAB 5: About ──────────────────────────────────────────────────────────
    with tab5:

        st.markdown("""
        <div style='max-width:900px; margin:0 auto;'>

        <h2 style='font-family:Orbitron; color:#fff; font-size:22px; letter-spacing:2px; margin-bottom:4px;'>
            BMW <span style='color:#e5000a;'>DriveIQ</span>
        </h2>
        <p style='color:#555; font-size:11px; letter-spacing:4px; text-transform:uppercase; margin-bottom:32px;'>
            AI-Powered Driver Behaviour Intelligence System
        </p>

        </div>
        """, unsafe_allow_html=True)

        a1, a2 = st.columns([3, 2], gap="large")

        with a1:
            st.markdown("""
            <div class='report-box'>
                <strong style='color:#e5000a; font-family:Orbitron; font-size:13px; letter-spacing:2px;'>
                    PROJECT OVERVIEW
                </strong><br><br>
                BMW DriveIQ is a real-time driver behaviour intelligence system that classifies driving style
                from raw vehicle telemetry using machine learning, then generates personalised coaching
                feedback powered by a large language model.<br><br>
                The system replicates the kind of AI layer you would find inside a BMW ConnectedDrive or
                iDrive 9 platform — turning streams of sensor data into actionable, human-readable intelligence
                that helps drivers understand their habits, manage risk, and improve their technique behind the wheel.<br><br>
                The project demonstrates a full ML engineering pipeline: from synthetic data generation
                and feature engineering, through model training and evaluation, to a production-grade
                interactive dashboard — all built from scratch.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("""
            <div class='report-box'>
                <strong style='color:#e5000a; font-family:Orbitron; font-size:13px; letter-spacing:2px;'>
                    HOW IT WORKS
                </strong><br><br>
                <strong style='color:#ccc;'>1 — Telemetry Simulation</strong><br>
                A physics-informed simulator generates realistic multi-channel vehicle telemetry — speed,
                RPM, throttle position, brake pressure, lateral G-force, steering angle, engine load,
                gear, and more — for five distinct driver archetypes. Each archetype is parameterised
                with profile-specific distributions to capture the real behavioural differences between,
                say, a fatigued motorway driver and an aggressive urban commuter.<br><br>
                <strong style='color:#ccc;'>2 — Feature Engineering</strong><br>
                Raw telemetry is sliced into 10-second sliding windows (5-second step) and 52 statistical
                features are extracted per window: means, standard deviations, percentiles, event counts
                (hard brakes, high-throttle bursts), jerk metrics, time-to-collision estimates, and a
                composite risk score. This transforms a time-series problem into a tabular classification task.<br><br>
                <strong style='color:#ccc;'>3 — XGBoost Classification</strong><br>
                An XGBoost gradient-boosted classifier is trained on 11,600 labelled feature windows across
                five classes. The model achieves <strong style='color:#e5000a;'>99.5% test accuracy</strong>
                and <strong style='color:#e5000a;'>99.5% 5-fold CV accuracy</strong>, confirming that the
                engineered features are highly discriminative between driving styles.<br><br>
                <strong style='color:#ccc;'>4 — LLM Coaching Layer</strong><br>
                The classification result and aggregated telemetry stats are passed to a Groq-hosted
                LLaMA 3.3 70B model via a structured prompt. The model acts as a BMW Driving Experience
                instructor — generating a personalised coaching report with performance snapshots, risk
                analysis, specific recommendations, and a DriveIQ score out of 100.
            </div>
            """, unsafe_allow_html=True)

        with a2:
            st.markdown("""
            <div class='report-box'>
                <strong style='color:#e5000a; font-family:Orbitron; font-size:13px; letter-spacing:2px;'>
                    DRIVER PROFILES
                </strong><br><br>
                <span style='color:#00cc66; font-weight:700;'>● SMOOTH</span><br>
                <span style='color:#888; font-size:13px;'>Consistent, controlled inputs. Low jerk, steady throttle,
                minimal hard brakes. Risk: LOW.</span><br><br>
                <span style='color:#e5000a; font-weight:700;'>● AGGRESSIVE</span><br>
                <span style='color:#888; font-size:13px;'>High throttle bursts, frequent hard braking, sharp
                steering corrections, elevated lateral G. Risk: HIGH.</span><br><br>
                <span style='color:#ff6600; font-weight:700;'>● FATIGUED</span><br>
                <span style='color:#888; font-size:13px;'>Erratic, inconsistent behaviour — late reactions,
                micro-corrections, variable lane position. Risk: MEDIUM-HIGH.</span><br><br>
                <span style='color:#ffcc00; font-weight:700;'>● SPORTY</span><br>
                <span style='color:#888; font-size:13px;'>High-performance driving — elevated speeds and RPMs
                but controlled and deliberate. Risk: MEDIUM.</span><br><br>
                <span style='color:#4488ff; font-weight:700;'>● HIGHWAY CRUISE</span><br>
                <span style='color:#888; font-size:13px;'>Sustained high-speed, low-variance motorway driving.
                Smooth, predictable, minimal inputs. Risk: LOW.</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("""
            <div class='report-box'>
                <strong style='color:#e5000a; font-family:Orbitron; font-size:13px; letter-spacing:2px;'>
                    TECHNOLOGY STACK
                </strong><br><br>
                <strong style='color:#ccc;'>ML & Data</strong><br>
                <span style='color:#888;'>XGBoost · scikit-learn · pandas · NumPy · SciPy</span><br><br>
                <strong style='color:#ccc;'>AI Coaching</strong><br>
                <span style='color:#888;'>Groq API · LLaMA 3.3 70B Versatile</span><br><br>
                <strong style='color:#ccc;'>Visualisation</strong><br>
                <span style='color:#888;'>Streamlit · Plotly · Matplotlib</span><br><br>
                <strong style='color:#ccc;'>Telemetry Pipeline</strong><br>
                <span style='color:#888;'>Custom physics-informed simulator · 14 sensor channels ·
                52 engineered features · sliding-window extraction</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            if meta:
                st.markdown(f"""
                <div class='report-box'>
                    <strong style='color:#e5000a; font-family:Orbitron; font-size:13px; letter-spacing:2px;'>
                        MODEL PERFORMANCE
                    </strong><br><br>
                    <strong style='color:#ccc;'>Test Accuracy</strong><br>
                    <span style='color:#e5000a; font-size:22px; font-family:Orbitron; font-weight:700;'>
                        {meta['accuracy']*100:.2f}%
                    </span><br><br>
                    <strong style='color:#ccc;'>F1 Score (weighted)</strong><br>
                    <span style='color:#e5000a; font-size:22px; font-family:Orbitron; font-weight:700;'>
                        {meta['f1_weighted']:.4f}
                    </span><br><br>
                    <strong style='color:#ccc;'>5-Fold CV Accuracy</strong><br>
                    <span style='color:#e5000a; font-size:22px; font-family:Orbitron; font-weight:700;'>
                        {meta['cv_mean']*100:.2f}% <span style='font-size:13px; color:#555;'>± {meta['cv_std']*100:.2f}%</span>
                    </span><br><br>
                    <strong style='color:#ccc;'>Classes · Features · Windows</strong><br>
                    <span style='color:#888;'>{meta['n_classes']} profiles · {meta['n_features']} features · 11,600 windows</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div style='border: 1px solid #1a1a1a; border-left: 3px solid #e5000a;
                    background:#0f0f0f; padding: 24px 28px; border-radius:2px;'>
            <strong style='color:#e5000a; font-family:Orbitron; font-size:13px; letter-spacing:2px;'>
                ABOUT THE BUILDER
            </strong><br><br>
            <span style='color:#ccc; font-size:15px; line-height:1.8;'>
                Built by <strong style='color:#fff;'>Hiteshree Sharma</strong> as part of the
                AI/ML Engineering programme at <strong style='color:#fff;'>Masters' Union, Gurugram</strong>.<br><br>
                This project was designed to demonstrate end-to-end applied ML — bridging the gap between raw
                sensor data and human-intelligible driver intelligence, with a production-grade UI built to
                the visual standard of BMW's own digital ecosystem. The choice of BMW as the vehicle brand
                is intentional: BMW's ConnectedDrive and iDrive platforms represent the current frontier of
                in-car AI, and this project models what a next-generation driver coaching layer on that
                platform could look like.<br><br>
                The system is fully modular — the simulator, feature engineering pipeline, classifier, and
                LLM coaching layer are each independent components that can be swapped, retrained, or extended
                independently. The dashboard is built to handle real telemetry data with minimal modification.
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("""
    <div style='text-align:center; color:#333; font-size:11px; letter-spacing:3px; padding:10px 0;'>
        BMW DRIVEIQ · BUILT BY HITESHREE SHARMA · MASTERS' UNION GURUGRAM · AI/ML ENGINEERING
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
