"""
BMW DriveIQ — Flask app for Railway deployment
===============================================
Serves the static dashboard and exposes /api/simulate and /api/report.
"""

import json
import os
import sys

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "bmw_driveiq"))

import joblib
from utils.coach import generate_report
from utils.feature_engineering import STEP_SIZE, WINDOW_SIZE, extract_window_features
from utils.simulator import PROFILES, simulate_trip

LABEL_NAMES = {0: "SMOOTH", 1: "AGGRESSIVE", 2: "FATIGUED", 3: "SPORTY", 4: "HIGHWAY_CRUISE"}

app = Flask(__name__, static_folder="public", static_url_path="")


# ── Helpers ───────────────────────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


def load_model():
    models_dir = os.path.join(BASE_DIR, "bmw_driveiq", "models")
    model       = joblib.load(os.path.join(models_dir, "driveiq_model.joblib"))
    feat_cols   = joblib.load(os.path.join(models_dir, "feature_cols.joblib"))
    with open(os.path.join(models_dir, "metadata.json")) as f:
        meta = json.load(f)
    return model, feat_cols, meta


def predict_windows(trip_df, model, feat_cols):
    results = []
    for start in range(0, len(trip_df) - WINDOW_SIZE, STEP_SIZE):
        window = trip_df.iloc[start: start + WINDOW_SIZE]
        if len(window) < WINDOW_SIZE:
            continue
        feats = extract_window_features(window)
        X     = np.array([[feats.get(c, 0) for c in feat_cols]])
        pred  = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]
        results.append({
            "window_start": start,
            "profile":      LABEL_NAMES[pred],
            "confidence":   float(proba[pred]),
            "risk_score":   float(feats.get("risk_score", 0)),
            "stats":        {k: float(v) for k, v in feats.items()},
        })
    return results


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("public", "index.html")


@app.route("/api/simulate", methods=["POST"])
def simulate():
    data     = request.get_json(force=True) or {}
    profile  = data.get("profile") or str(np.random.choice(list(PROFILES.keys())))
    duration = max(60, min(600, int(data.get("duration", 180))))

    trip_df            = simulate_trip(profile, duration_seconds=duration)
    model, feat_cols, meta = load_model()
    windows            = predict_windows(trip_df, model, feat_cols)

    telemetry = {
        "timestamp":          trip_df["timestamp"].tolist(),
        "speed_kmh":          trip_df["speed_kmh"].tolist(),
        "rpm":                trip_df["rpm"].tolist(),
        "throttle_pos":       trip_df["throttle_pos"].tolist(),
        "brake_pressure":     trip_df["brake_pressure"].tolist(),
        "lateral_g":          trip_df["lateral_g"].tolist(),
        "steering_angle_deg": trip_df["steering_angle_deg"].tolist(),
        "engine_load_pct":    trip_df["engine_load_pct"].tolist(),
    }

    payload = json.dumps(
        {"true_profile": profile, "telemetry": telemetry, "windows": windows, "model_meta": meta},
        cls=NumpyEncoder,
    )
    return app.response_class(payload, mimetype="application/json")


@app.route("/api/report", methods=["POST"])
def report():
    data            = request.get_json(force=True) or {}
    stats           = data.get("stats", {})
    predicted_label = int(data.get("predicted_label", 0))
    confidence      = float(data.get("confidence", 0.9))
    text            = generate_report(stats, predicted_label, confidence, verbose=False)
    return jsonify({"report": text})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
