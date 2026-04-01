"""
BMW DriveIQ — Simulate API Endpoint
Runs telemetry simulation + XGBoost classification and returns JSON.
"""

from http.server import BaseHTTPRequestHandler
import json
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "bmw_driveiq"))

import numpy as np
from utils.simulator import simulate_trip, PROFILES
from utils.feature_engineering import extract_window_features, WINDOW_SIZE, STEP_SIZE
import joblib

LABEL_NAMES = {0: "SMOOTH", 1: "AGGRESSIVE", 2: "FATIGUED", 3: "SPORTY", 4: "HIGHWAY_CRUISE"}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_model():
    models_dir = os.path.join(BASE_DIR, "bmw_driveiq", "models")
    model = joblib.load(os.path.join(models_dir, "driveiq_model.joblib"))
    feature_cols = joblib.load(os.path.join(models_dir, "feature_cols.joblib"))
    with open(os.path.join(models_dir, "metadata.json")) as f:
        meta = json.load(f)
    return model, feature_cols, meta


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
            "risk_score": float(feats.get("risk_score", 0)),
            "stats": {k: float(v) for k, v in feats.items()},
        })
    return results


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length > 0 else {}

            profile = body.get("profile")
            duration = max(60, min(600, int(body.get("duration", 180))))

            if not profile:
                profile = str(np.random.choice(list(PROFILES.keys())))

            trip_df = simulate_trip(profile, duration_seconds=duration)
            model, feature_cols, meta = load_model()
            windows = predict_trip_windows(trip_df, model, feature_cols)

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

            payload = json.dumps({
                "true_profile": profile,
                "telemetry":    telemetry,
                "windows":      windows,
                "model_meta":   meta,
            }, cls=NumpyEncoder).encode()

            self._respond(200, payload)

        except Exception as e:
            self._respond(500, json.dumps({"error": str(e)}).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _respond(self, code, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass
