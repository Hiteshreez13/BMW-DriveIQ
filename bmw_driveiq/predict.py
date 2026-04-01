"""
BMW DriveIQ — Inference Pipeline
==================================
Run real-time predictions on new telemetry data and generate coaching reports.

Usage:
  python predict.py                          # predict on a fresh simulated trip
  python predict.py --profile AGGRESSIVE     # simulate a specific driver type
  python predict.py --file data/my_trip.csv  # predict on your own telemetry CSV
"""

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import joblib
import json
import argparse
from pathlib import Path

from utils.simulator import simulate_trip, PROFILES
from utils.feature_engineering import extract_window_features, WINDOW_SIZE, STEP_SIZE
from utils.coach import generate_report, generate_session_summary

LABEL_NAMES = {0: "SMOOTH", 1: "AGGRESSIVE", 2: "FATIGUED", 3: "SPORTY", 4: "HIGHWAY_CRUISE"}
RISK_EMOJI  = {"LOW": "✅", "MEDIUM": "🟡", "MEDIUM-HIGH": "🟠", "HIGH": "🔴"}
RISK_LEVELS = {"SMOOTH": "LOW", "AGGRESSIVE": "HIGH", "FATIGUED": "MEDIUM-HIGH",
               "SPORTY": "MEDIUM", "HIGHWAY_CRUISE": "LOW"}


def load_model():
    if not Path("models/driveiq_model.joblib").exists():
        print("❌ No trained model found. Run `python train.py` first.")
        sys.exit(1)
    model = joblib.load("models/driveiq_model.joblib")
    feature_cols = joblib.load("models/feature_cols.joblib")
    return model, feature_cols


def predict_trip(trip_df: pd.DataFrame, model, feature_cols: list) -> list[dict]:
    """Run sliding-window prediction over a trip DataFrame."""
    results = []
    for start in range(0, len(trip_df) - WINDOW_SIZE, STEP_SIZE):
        window = trip_df.iloc[start: start + WINDOW_SIZE]
        if len(window) < WINDOW_SIZE:
            continue
        feats = extract_window_features(window)
        X = np.array([[feats.get(col, 0) for col in feature_cols]])
        pred = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]
        confidence = float(proba[pred])
        results.append({
            "window_start": start,
            "profile": LABEL_NAMES[pred],
            "label": pred,
            "confidence": confidence,
            "stats": feats
        })
    return results


def summarise_predictions(results: list[dict]) -> dict:
    """Aggregate window predictions into session-level summary."""
    from collections import Counter
    counts = Counter(r["profile"] for r in results)
    dominant = counts.most_common(1)[0][0]
    avg_conf = np.mean([r["confidence"] for r in results])

    avg_stats = {}
    keys = list(results[0]["stats"].keys())
    for k in keys:
        avg_stats[k] = round(np.mean([r["stats"][k] for r in results]), 4)

    return {
        "dominant_profile": dominant,
        "profile_distribution": dict(counts),
        "avg_confidence": avg_conf,
        "avg_stats": avg_stats,
        "total_windows": len(results),
        "risk_level": RISK_LEVELS.get(dominant, "UNKNOWN")
    }


def print_summary(summary: dict):
    profile = summary["dominant_profile"]
    risk = summary["risk_level"]
    emoji = RISK_EMOJI.get(risk, "❓")
    conf = summary["avg_confidence"]
    dist = summary["profile_distribution"]

    print("\n" + "━"*60)
    print(f"  BMW DriveIQ — Session Analysis")
    print("━"*60)
    print(f"  Dominant Profile  : {profile}")
    print(f"  Risk Level        : {emoji} {risk}")
    print(f"  Avg Confidence    : {conf:.1%}")
    print(f"  Windows Analysed  : {summary['total_windows']}")
    print(f"\n  Profile Breakdown:")
    for p, count in sorted(dist.items(), key=lambda x: -x[1]):
        bar = "█" * int(count / summary["total_windows"] * 30)
        pct = count / summary["total_windows"]
        print(f"    {p:<20} {bar:<30} {pct:.0%}")

    s = summary["avg_stats"]
    print(f"\n  Key Metrics (session average):")
    print(f"    Speed          : {s.get('speed_mean', 0):.1f} km/h avg | {s.get('speed_max', 0):.1f} km/h peak")
    print(f"    Hard Brakes    : {s.get('hard_brake_count', 0):.2f} / window")
    print(f"    Lateral G      : {s.get('lateral_g_max', 0):.3f}g peak")
    print(f"    Risk Score     : {s.get('risk_score', 0):.2f}")
    print(f"    TTC Danger %   : {s.get('ttc_low_pct', 0):.1%}")
    print("━"*60 + "\n")


def run_predict(profile_name: str = None, csv_path: str = None, duration: int = 180):
    model, feature_cols = load_model()

    # ── Load or simulate trip ─────────────────────────────────────────────────
    if csv_path:
        print(f"📂 Loading telemetry from {csv_path}...")
        trip_df = pd.read_csv(csv_path)
    else:
        if profile_name is None:
            import random
            profile_name = random.choice(list(PROFILES.keys()))
        print(f"🎲 Simulating {duration}s trip — Profile hint: {profile_name}")
        trip_df = simulate_trip(profile_name, duration_seconds=duration)

    # ── Run predictions ───────────────────────────────────────────────────────
    print(f"🔍 Running DriveIQ analysis ({len(trip_df)} data points)...")
    results = predict_trip(trip_df, model, feature_cols)

    if not results:
        print("❌ Not enough data for analysis (need at least 10 seconds).")
        return

    # ── Summarise ─────────────────────────────────────────────────────────────
    summary = summarise_predictions(results)
    print_summary(summary)

    # ── Generate coaching report ──────────────────────────────────────────────
    dominant_label = [k for k, v in LABEL_NAMES.items()
                      if v == summary["dominant_profile"]][0]
    print("🤖 Generating AI coaching report via Claude...\n")
    report = generate_report(
        stats=summary["avg_stats"],
        predicted_label=dominant_label,
        confidence=summary["avg_confidence"]
    )

    # ── Save report ───────────────────────────────────────────────────────────
    Path("outputs").mkdir(exist_ok=True)
    report_path = "outputs/coaching_report.md"
    with open(report_path, "w") as f:
        f.write(f"# BMW DriveIQ — Coaching Report\n\n")
        f.write(f"**Profile**: {summary['dominant_profile']}  \n")
        f.write(f"**Risk**: {summary['risk_level']}  \n")
        f.write(f"**Confidence**: {summary['avg_confidence']:.1%}  \n\n")
        f.write("---\n\n")
        f.write(report)

    print(f"📄 Report saved → {report_path}")
    return summary, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BMW DriveIQ Prediction Engine")
    parser.add_argument("--profile", type=str, choices=list(PROFILES.keys()),
                        help="Simulate a specific driver profile")
    parser.add_argument("--file", type=str, help="Path to telemetry CSV")
    parser.add_argument("--duration", type=int, default=180, help="Trip duration in seconds")
    args = parser.parse_args()
    run_predict(args.profile, args.file, args.duration)
