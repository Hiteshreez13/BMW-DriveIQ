"""
BMW DriveIQ — Feature Engineering Pipeline
===========================================
Converts raw 1Hz telemetry into 10-second window features for ML classification.

This mirrors real automotive signal processing:
- Rolling statistics capture behaviour trends
- Jerk (rate of change of acceleration) signals sudden manoeuvres
- Event counts per window quantify risky behaviour
- Percentile features capture tail behaviour (worst moments in window)
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from pathlib import Path


WINDOW_SIZE = 10        # seconds per classification window
STEP_SIZE = 5           # overlap step (50% overlap = more training samples)


def compute_jerk(accel: np.ndarray) -> np.ndarray:
    """Rate of change of acceleration — key signal for aggressive driving."""
    return np.diff(accel, prepend=accel[0])


def extract_window_features(window: pd.DataFrame) -> dict:
    """
    Extract all features from a single 10-second telemetry window.
    Returns a flat feature dict ready for ML training.
    """
    feats = {}

    # ── Speed features ────────────────────────────────────────────────────────
    sp = window["speed_kmh"].values
    feats["speed_mean"]   = np.mean(sp)
    feats["speed_std"]    = np.std(sp)
    feats["speed_max"]    = np.max(sp)
    feats["speed_min"]    = np.min(sp)
    feats["speed_range"]  = np.max(sp) - np.min(sp)
    feats["speed_p95"]    = np.percentile(sp, 95)

    # ── Acceleration features ─────────────────────────────────────────────────
    ac = window["acceleration_ms2"].values
    feats["accel_mean"]     = np.mean(ac)
    feats["accel_std"]      = np.std(ac)
    feats["accel_max"]      = np.max(ac)
    feats["accel_min"]      = np.min(ac)
    feats["accel_abs_mean"] = np.mean(np.abs(ac))
    feats["accel_p95"]      = np.percentile(np.abs(ac), 95)

    # ── Jerk (rate of acceleration change) ────────────────────────────────────
    jk = compute_jerk(ac)
    feats["jerk_mean"]    = np.mean(np.abs(jk))
    feats["jerk_max"]     = np.max(np.abs(jk))
    feats["jerk_std"]     = np.std(jk)

    # ── Throttle features ─────────────────────────────────────────────────────
    th = window["throttle_pos"].values
    feats["throttle_mean"]    = np.mean(th)
    feats["throttle_std"]     = np.std(th)
    feats["throttle_max"]     = np.max(th)
    feats["throttle_jerk"]    = np.mean(np.abs(np.diff(th, prepend=th[0])))
    feats["throttle_high_pct"]= np.mean(th > 0.7)   # % time at high throttle

    # ── Braking features ──────────────────────────────────────────────────────
    br = window["brake_pressure"].values
    feats["brake_mean"]       = np.mean(br)
    feats["brake_max"]        = np.max(br)
    feats["brake_std"]        = np.std(br)
    feats["brake_p95"]        = np.percentile(br, 95)
    feats["hard_brake_count"] = window["hard_brake_event"].sum()

    # ── Acceleration events ───────────────────────────────────────────────────
    feats["hard_accel_count"] = window["hard_accel_event"].sum()

    # ── RPM features ──────────────────────────────────────────────────────────
    rp = window["rpm"].values
    feats["rpm_mean"]     = np.mean(rp)
    feats["rpm_max"]      = np.max(rp)
    feats["rpm_std"]      = np.std(rp)
    feats["rpm_high_pct"] = np.mean(rp > 4000)  # % time above 4k RPM

    # ── Gear shift frequency ──────────────────────────────────────────────────
    gears = window["gear"].values
    feats["gear_changes"] = np.sum(np.abs(np.diff(gears)) > 0)
    feats["gear_mean"]    = np.mean(gears)

    # ── Steering & lateral dynamics ───────────────────────────────────────────
    st = window["steering_angle_deg"].values
    feats["steering_std"]       = np.std(st)
    feats["steering_abs_mean"]  = np.mean(np.abs(st))
    feats["steering_max"]       = np.max(np.abs(st))
    feats["steering_reversals"] = np.sum(np.diff(np.sign(st)) != 0)

    lg = window["lateral_g"].values
    feats["lateral_g_mean"] = np.mean(lg)
    feats["lateral_g_max"]  = np.max(lg)
    feats["lateral_g_std"]  = np.std(lg)
    feats["lateral_g_p95"]  = np.percentile(lg, 95)

    # ── Longitudinal G-force ──────────────────────────────────────────────────
    log_g = window["longitudinal_g"].values
    feats["long_g_mean"]    = np.mean(np.abs(log_g))
    feats["long_g_max"]     = np.max(np.abs(log_g))
    feats["long_g_neg_mean"]= np.mean(log_g[log_g < 0]) if any(log_g < 0) else 0  # braking g

    # ── Engine load ───────────────────────────────────────────────────────────
    el = window["engine_load_pct"].values
    feats["engine_load_mean"] = np.mean(el)
    feats["engine_load_max"]  = np.max(el)

    # ── Safety: TTC proxy ─────────────────────────────────────────────────────
    ttc = window["ttc_proxy"].values
    feats["ttc_min"]      = np.min(ttc)
    feats["ttc_mean"]     = np.mean(ttc)
    feats["ttc_low_pct"]  = np.mean(ttc < 1.5)   # % time in danger zone

    # ── Statistical shape features ────────────────────────────────────────────
    feats["speed_skew"]        = float(skew(sp))
    feats["accel_kurtosis"]    = float(kurtosis(ac))
    feats["steering_kurtosis"] = float(kurtosis(st))

    # ── Composite risk score (domain-engineered feature) ──────────────────────
    feats["risk_score"] = (
        feats["hard_brake_count"] * 2.0
        + feats["hard_accel_count"] * 1.5
        + feats["lateral_g_max"] * 3.0
        + feats["jerk_max"] * 1.0
        + feats["steering_std"] * 0.1
        + feats["ttc_low_pct"] * 4.0
    )

    return feats


def build_feature_matrix(
    raw_path: str = "data/telemetry_raw.csv",
    output_path: str = "data/features.csv"
) -> pd.DataFrame:
    """
    Slide a 10-second window over every trip and extract features.
    Returns a feature matrix ready for ML training.
    """
    print("⚙️  BMW DriveIQ — Feature Engineering Pipeline")
    print(f"   Loading raw telemetry from {raw_path}...")

    df = pd.read_csv(raw_path)
    trips = df["trip_id"].unique()

    print(f"   Processing {len(trips)} trips with {WINDOW_SIZE}s windows, {STEP_SIZE}s step...")

    all_features = []
    for i, trip_id in enumerate(trips):
        trip = df[df["trip_id"] == trip_id].reset_index(drop=True)
        label = trip["label"].iloc[0]
        profile = trip["profile"].iloc[0]

        for start in range(0, len(trip) - WINDOW_SIZE, STEP_SIZE):
            window = trip.iloc[start: start + WINDOW_SIZE]
            if len(window) < WINDOW_SIZE:
                continue
            feats = extract_window_features(window)
            feats["label"]   = label
            feats["profile"] = profile
            feats["trip_id"] = trip_id
            all_features.append(feats)

        if (i + 1) % 50 == 0:
            print(f"   [{i+1}/{len(trips)}] trips processed...")

    feat_df = pd.DataFrame(all_features)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(output_path, index=False)

    print(f"\n✅ Feature matrix saved → {output_path}")
    print(f"   Windows  : {len(feat_df):,}")
    print(f"   Features : {len(feat_df.columns) - 3}")
    print(f"   Class distribution:\n{feat_df['profile'].value_counts().to_string()}")
    return feat_df


if __name__ == "__main__":
    build_feature_matrix()
