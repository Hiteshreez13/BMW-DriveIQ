"""
BMW DriveIQ — Model Training Pipeline
=======================================
Trains an XGBoost classifier on engineered telemetry features.
Outputs: trained model, feature importance plot, evaluation metrics.

Run: python train.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from xgboost import XGBClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Run data pipeline if needed ───────────────────────────────────────────────
from utils.simulator import generate_dataset
from utils.feature_engineering import build_feature_matrix

LABEL_NAMES = {0: "SMOOTH", 1: "AGGRESSIVE", 2: "FATIGUED", 3: "SPORTY", 4: "HIGHWAY_CRUISE"}
FEATURE_COLS_EXCLUDE = ["label", "profile", "trip_id"]


def load_or_generate_features(
    raw_path="data/telemetry_raw.csv",
    feat_path="data/features.csv",
    force_regen=False
) -> pd.DataFrame:
    if not Path(feat_path).exists() or force_regen:
        if not Path(raw_path).exists() or force_regen:
            print("📡 Generating telemetry data...")
            generate_dataset(trips_per_class=40, output_path=raw_path)
        print("⚙️  Building feature matrix...")
        build_feature_matrix(raw_path, feat_path)
    return pd.read_csv(feat_path)


def train(force_regen: bool = False):
    print("\n" + "="*60)
    print("  BMW DriveIQ — Training Pipeline")
    print("="*60 + "\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    df = load_or_generate_features(force_regen=force_regen)
    feature_cols = [c for c in df.columns if c not in FEATURE_COLS_EXCLUDE]
    X = df[feature_cols].values
    y = df["label"].values

    print(f"📊 Dataset: {len(df):,} windows | {len(feature_cols)} features | {len(np.unique(y))} classes")

    # ── Train / test split (stratified) ───────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")

    # ── XGBoost model ─────────────────────────────────────────────────────────
    print("\n🚀 Training XGBoost classifier...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        gamma=0.1,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # ── Evaluation ────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")

    print(f"\n📈 Results:")
    print(f"   Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"   F1 Score : {f1:.4f} (weighted)")

    target_names = [LABEL_NAMES[i] for i in range(5)]
    print(f"\n{classification_report(y_test, y_pred, target_names=target_names)}")

    # ── Cross-validation ──────────────────────────────────────────────────────
    print("🔄 5-Fold Cross-Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"   CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Save model & metadata ─────────────────────────────────────────────────
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/driveiq_model.joblib")
    joblib.dump(feature_cols, "models/feature_cols.joblib")

    metadata = {
        "accuracy": float(acc),
        "f1_weighted": float(f1),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "n_features": len(feature_cols),
        "n_classes": 5,
        "classes": LABEL_NAMES,
        "feature_cols": feature_cols
    }
    with open("models/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Model saved → models/driveiq_model.joblib")

    # ── Feature importance plot ───────────────────────────────────────────────
    _plot_feature_importance(model, feature_cols)
    _plot_confusion_matrix(y_test, y_pred, target_names)

    return model, feature_cols, metadata


def _plot_feature_importance(model, feature_cols, top_n=20):
    """Plot top N most important features."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0a0a0a")
    ax.set_facecolor("#0a0a0a")

    colors = ["#e5000a" if importances[i] > np.median(importances) else "#4a0005"
              for i in indices]

    ax.barh(range(top_n), importances[indices][::-1], color=colors[::-1], height=0.7)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_cols[i] for i in indices][::-1],
                       color="white", fontsize=9)
    ax.set_xlabel("Feature Importance (gain)", color="white")
    ax.set_title("BMW DriveIQ — Top Feature Importances", color="white",
                 fontsize=13, fontweight="bold", pad=15)
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#333")
    ax.spines["left"].set_color("#333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.label.set_color("white")

    plt.tight_layout()
    Path("outputs").mkdir(exist_ok=True)
    plt.savefig("outputs/feature_importance.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print("   📊 Feature importance plot → outputs/feature_importance.png")


def _plot_confusion_matrix(y_test, y_pred, target_names):
    """Plot styled confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#0a0a0a")
    ax.set_facecolor("#0a0a0a")

    im = ax.imshow(cm_norm, cmap="Reds", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color="white")

    ax.set_xticks(range(len(target_names)))
    ax.set_yticks(range(len(target_names)))
    ax.set_xticklabels(target_names, rotation=30, ha="right", color="white", fontsize=9)
    ax.set_yticklabels(target_names, color="white", fontsize=9)

    for i in range(len(target_names)):
        for j in range(len(target_names)):
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.0%})",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] < 0.6 else "black",
                    fontsize=8, fontweight="bold")

    ax.set_xlabel("Predicted", color="white")
    ax.set_ylabel("Actual", color="white")
    ax.set_title("BMW DriveIQ — Confusion Matrix", color="white",
                 fontsize=13, fontweight="bold", pad=15)

    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print("   📊 Confusion matrix → outputs/confusion_matrix.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--regen", action="store_true", help="Regenerate data from scratch")
    args = parser.parse_args()
    train(force_regen=args.regen)
