"""
BMW DriveIQ — Vehicle Telemetry Simulator
==========================================
Generates realistic driving telemetry data for 5 driver behaviour profiles.
Simulates: speed, RPM, throttle, braking, steering, G-forces, gear position.

Each "trip" is a sequence of 1Hz sensor readings (one reading per second).
Windows of 10 seconds are used as ML classification units.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

# ── Driver profile physics definitions ───────────────────────────────────────
PROFILES = {
    "SMOOTH": {
        "speed_mean": 65, "speed_std": 12,
        "throttle_mean": 0.32, "throttle_std": 0.08,
        "brake_mean": 0.10, "brake_std": 0.05,
        "rpm_mean": 2200, "rpm_std": 350,
        "steering_std": 4.0,
        "lateral_g_std": 0.08,
        "hard_brake_prob": 0.01,
        "hard_accel_prob": 0.01,
        "label": 0
    },
    "AGGRESSIVE": {
        "speed_mean": 95, "speed_std": 28,
        "throttle_mean": 0.72, "throttle_std": 0.22,
        "brake_mean": 0.35, "brake_std": 0.18,
        "rpm_mean": 4800, "rpm_std": 1200,
        "steering_std": 14.0,
        "lateral_g_std": 0.38,
        "hard_brake_prob": 0.18,
        "hard_accel_prob": 0.22,
        "label": 1
    },
    "FATIGUED": {
        "speed_mean": 72, "speed_std": 18,
        "throttle_mean": 0.38, "throttle_std": 0.16,
        "brake_mean": 0.14, "brake_std": 0.12,
        "rpm_mean": 2600, "rpm_std": 600,
        "steering_std": 11.0,   # erratic micro-corrections
        "lateral_g_std": 0.14,
        "hard_brake_prob": 0.08,
        "hard_accel_prob": 0.04,
        "label": 2
    },
    "SPORTY": {
        "speed_mean": 88, "speed_std": 20,
        "throttle_mean": 0.62, "throttle_std": 0.18,
        "brake_mean": 0.22, "brake_std": 0.10,
        "rpm_mean": 4200, "rpm_std": 900,
        "steering_std": 9.0,
        "lateral_g_std": 0.24,
        "hard_brake_prob": 0.06,
        "hard_accel_prob": 0.14,
        "label": 3
    },
    "HIGHWAY_CRUISE": {
        "speed_mean": 110, "speed_std": 6,
        "throttle_mean": 0.42, "throttle_std": 0.05,
        "brake_mean": 0.04, "brake_std": 0.02,
        "rpm_mean": 2800, "rpm_std": 180,
        "steering_std": 2.0,
        "lateral_g_std": 0.03,
        "hard_brake_prob": 0.005,
        "hard_accel_prob": 0.005,
        "label": 4
    }
}

LABEL_NAMES = {0: "SMOOTH", 1: "AGGRESSIVE", 2: "FATIGUED", 3: "SPORTY", 4: "HIGHWAY_CRUISE"}


def simulate_trip(profile_name: str, duration_seconds: int = 300) -> pd.DataFrame:
    """Simulate a single driving trip for a given profile."""
    p = PROFILES[profile_name]
    n = duration_seconds
    t = np.arange(n)

    # ── Speed (km/h) with momentum (AR process for realism) ──────────────────
    speed = np.zeros(n)
    speed[0] = np.clip(np.random.normal(p["speed_mean"], p["speed_std"]), 5, 200)
    for i in range(1, n):
        target = np.random.normal(p["speed_mean"], p["speed_std"])
        speed[i] = 0.85 * speed[i-1] + 0.15 * target + np.random.normal(0, 1.5)
        speed[i] = np.clip(speed[i], 0, 220)

    # ── Acceleration (m/s²) derived from speed delta ──────────────────────────
    speed_ms = speed / 3.6
    accel = np.diff(speed_ms, prepend=speed_ms[0])

    # ── Throttle position (0–1) ───────────────────────────────────────────────
    throttle = np.clip(
        np.random.normal(p["throttle_mean"], p["throttle_std"], n)
        + (accel > 1.5).astype(float) * p["hard_accel_prob"] * 3,
        0, 1
    )

    # ── Brake pressure (0–1) ─────────────────────────────────────────────────
    brake = np.clip(
        np.random.normal(p["brake_mean"], p["brake_std"], n)
        + (accel < -2.0).astype(float) * 0.4,
        0, 1
    )

    # ── Hard events (boolean spikes) ─────────────────────────────────────────
    hard_brake = (np.random.rand(n) < p["hard_brake_prob"]) & (brake > 0.5)
    hard_accel = (np.random.rand(n) < p["hard_accel_prob"]) & (throttle > 0.7)

    # ── RPM with gear simulation ──────────────────────────────────────────────
    gear = np.clip(np.floor(speed / 30) + 1, 1, 8).astype(int)
    rpm_base = speed * 35 / gear
    rpm = np.clip(
        rpm_base + np.random.normal(0, p["rpm_std"], n),
        700, 8500
    )

    # ── Steering angle (degrees) ──────────────────────────────────────────────
    steering = np.random.normal(0, p["steering_std"], n)
    # Fatigued: add drift oscillation
    if profile_name == "FATIGUED":
        drift = 6 * np.sin(2 * np.pi * t / 45) + np.random.normal(0, 2, n)
        steering += drift

    # ── Lateral G-force (derived from steering + speed) ──────────────────────
    lateral_g = np.clip(
        (np.abs(steering) * speed / 1500)
        + np.random.normal(0, p["lateral_g_std"], n),
        0, 1.8
    )

    # ── Longitudinal G-force ─────────────────────────────────────────────────
    long_g = accel / 9.81

    # ── Engine load % ────────────────────────────────────────────────────────
    engine_load = np.clip(throttle * 100 + np.random.normal(0, 3, n), 0, 100)

    # ── Time to collision proxy (lower = more dangerous) ─────────────────────
    # Simulated: inverse of (speed * brake_delay)
    ttc_proxy = np.clip(
        50 / (speed + 1) + np.random.normal(0, 0.5, n),
        0.1, 10
    )
    ttc_proxy[hard_brake] *= 0.3  # danger moment

    df = pd.DataFrame({
        "timestamp": t,
        "speed_kmh": np.round(speed, 2),
        "acceleration_ms2": np.round(accel, 4),
        "throttle_pos": np.round(throttle, 4),
        "brake_pressure": np.round(brake, 4),
        "hard_brake_event": hard_brake.astype(int),
        "hard_accel_event": hard_accel.astype(int),
        "rpm": np.round(rpm, 0).astype(int),
        "gear": gear,
        "steering_angle_deg": np.round(steering, 2),
        "lateral_g": np.round(lateral_g, 4),
        "longitudinal_g": np.round(long_g, 4),
        "engine_load_pct": np.round(engine_load, 2),
        "ttc_proxy": np.round(ttc_proxy, 3),
        "label": p["label"],
        "profile": profile_name
    })

    return df


def generate_dataset(
    trips_per_class: int = 40,
    trip_duration: int = 300,
    output_path: str = "data/telemetry_raw.csv"
) -> pd.DataFrame:
    """Generate full training dataset across all driver profiles."""
    print(f"🚗 BMW DriveIQ — Generating telemetry dataset")
    print(f"   {trips_per_class} trips × {len(PROFILES)} profiles × {trip_duration}s = "
          f"{trips_per_class * len(PROFILES) * trip_duration:,} data points\n")

    all_trips = []
    for profile_name in PROFILES:
        print(f"   Simulating {profile_name} drivers...", end=" ")
        for trip_id in range(trips_per_class):
            trip_df = simulate_trip(profile_name, trip_duration)
            trip_df["trip_id"] = f"{profile_name}_{trip_id:03d}"
            all_trips.append(trip_df)
        print(f"✓ ({trips_per_class} trips)")

    df = pd.concat(all_trips, ignore_index=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n✅ Dataset saved → {output_path}")
    print(f"   Total rows : {len(df):,}")
    print(f"   Features   : {len(df.columns) - 3} sensor channels")
    print(f"   Classes    : {df['profile'].value_counts().to_dict()}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BMW DriveIQ Telemetry Simulator")
    parser.add_argument("--trips", type=int, default=40, help="Trips per class")
    parser.add_argument("--duration", type=int, default=300, help="Seconds per trip")
    parser.add_argument("--output", type=str, default="data/telemetry_raw.csv")
    args = parser.parse_args()
    generate_dataset(args.trips, args.duration, args.output)
