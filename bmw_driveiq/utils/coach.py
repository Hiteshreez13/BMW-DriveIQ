"""
BMW DriveIQ — AI Coaching Report Generator
===========================================
Uses Groq API to generate personalised, natural-language driver feedback
from classification results and telemetry statistics.

This mirrors BMW's HMI (Human-Machine Interface) layer —
turning raw ML predictions into actionable driver intelligence.
"""

import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

LABEL_NAMES = {
    0: "SMOOTH",
    1: "AGGRESSIVE",
    2: "FATIGUED",
    3: "SPORTY",
    4: "HIGHWAY_CRUISE"
}

RISK_LEVELS = {
    "SMOOTH": "LOW",
    "AGGRESSIVE": "HIGH",
    "FATIGUED": "MEDIUM-HIGH",
    "SPORTY": "MEDIUM",
    "HIGHWAY_CRUISE": "LOW"
}


def build_prompt(stats: dict, profile: str, confidence: float) -> str:
    """Build a structured prompt for the LLM with telemetry stats."""
    risk = RISK_LEVELS.get(profile, "UNKNOWN")

    return f"""You are BMW DriveIQ — an expert automotive AI coaching system built on BMW ConnectedDrive technology.

You have just analysed a driver's telemetry data and classified their behaviour.

## Classification Result
- **Driver Profile**: {profile}
- **Confidence**: {confidence:.1%}
- **Risk Level**: {risk}

## Telemetry Statistics
```json
{json.dumps(stats, indent=2)}
```

## Your Task
Generate a personalised BMW DriveIQ coaching report. Structure it exactly as follows:

### 🏁 DRIVER PROFILE: {profile}
One punchy sentence summarising this driver's style.

### 📊 PERFORMANCE SNAPSHOT
3–4 bullet points highlighting key metrics from the telemetry (use actual numbers from the stats).

### ⚠️ RISK ANALYSIS
Identify 2–3 specific risk behaviours detected, with the data evidence. Be direct and specific.

### 🎯 COACHING RECOMMENDATIONS
3 specific, actionable improvements this driver should make. Reference BMW driving principles.

### 🏆 STRENGTHS
1–2 things this driver does well (find something positive even for aggressive drivers).

### 📈 DRIVEIQ SCORE
Give an overall DriveIQ score out of 100. Brief explanation of the score.

---
Keep the tone professional but direct — like a BMW Driving Experience instructor, not a chatbot.
Use automotive terminology. Reference specific numbers from the telemetry stats.
Do NOT use generic advice. Every recommendation must be grounded in the actual data provided.
"""


def generate_report(
    stats: dict,
    predicted_label: int,
    confidence: float,
    verbose: bool = True
) -> str:
    """
    Generate a coaching report using Groq API.

    Args:
        stats: Dictionary of telemetry statistics for the session
        predicted_label: Integer class label (0–4)
        confidence: Model confidence score (0–1)
        verbose: Print the report to console

    Returns:
        Full coaching report as a string
    """
    profile = LABEL_NAMES.get(predicted_label, "UNKNOWN")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return _fallback_report(profile, stats, confidence)

    try:
        client = Groq(api_key=api_key)
        prompt = build_prompt(stats, profile, confidence)

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        report = response.choices[0].message.content

        if verbose:
            print("\n" + "="*60)
            print("BMW DriveIQ — Coaching Report")
            print("="*60)
            print(report)
            print("="*60 + "\n")

        return report

    except Exception as e:
        print(f"⚠️  Groq API error: {e}")
        return _fallback_report(profile, stats, confidence)


def _fallback_report(profile: str, stats: dict, confidence: float) -> str:
    """Fallback report if API key not set — useful for testing."""
    risk = RISK_LEVELS.get(profile, "UNKNOWN")
    return f"""
## BMW DriveIQ Report (Offline Mode)

**Driver Profile**: {profile}
**Risk Level**: {risk}
**Confidence**: {confidence:.1%}

### Key Stats
- Average Speed: {stats.get('speed_mean', 'N/A'):.1f} km/h
- Max Speed: {stats.get('speed_max', 'N/A'):.1f} km/h
- Hard Brake Events: {stats.get('hard_brake_count', 'N/A')}
- Risk Score: {stats.get('risk_score', 'N/A'):.2f}
- Lateral G (peak): {stats.get('lateral_g_max', 'N/A'):.3f}g

*Set GROQ_API_KEY in .env for full AI coaching reports.*
"""


def generate_session_summary(
    session_windows: list[dict],
    trip_duration_minutes: float
) -> str:
    """
    Generate a full session summary from multiple prediction windows.
    Aggregates stats across the session and gives overall coaching.
    """
    if not session_windows:
        return "No data to summarise."

    # Aggregate window-level predictions
    from collections import Counter
    profile_counts = Counter(w["profile"] for w in session_windows)
    dominant_profile = profile_counts.most_common(1)[0][0]
    dominant_label = [k for k, v in LABEL_NAMES.items() if v == dominant_profile][0]

    # Average confidence
    avg_confidence = sum(w["confidence"] for w in session_windows) / len(session_windows)

    # Aggregate telemetry stats
    agg_stats = {}
    numeric_keys = [k for k in session_windows[0]["stats"].keys()]
    for key in numeric_keys:
        vals = [w["stats"][key] for w in session_windows if key in w["stats"]]
        agg_stats[key] = round(sum(vals) / len(vals), 4)

    # Add session-level metadata
    agg_stats["session_duration_minutes"] = round(trip_duration_minutes, 1)
    agg_stats["profile_distribution"] = dict(profile_counts)
    agg_stats["total_windows_analysed"] = len(session_windows)

    return generate_report(agg_stats, dominant_label, avg_confidence)
