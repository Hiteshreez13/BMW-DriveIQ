# 🚗 BMW DriveIQ — AI-Powered Driver Behaviour Intelligence System

> Built by Hiteshree Sharma | Masters' Union, Gurugram

An end-to-end ML pipeline that simulates vehicle telemetry, classifies driver behaviour patterns,
and generates personalized AI coaching reports — modelled after BMW ConnectedDrive intelligence architecture.

---

## 🏗️ Architecture

```
Telemetry Simulator → Feature Engineering → XGBoost Classifier → Claude API → Coaching Report
                                    ↓
                           Streamlit Dashboard
```

## 📁 Project Structure

```
bmw_driveiq/
├── data/                      # Generated telemetry datasets
├── models/                    # Saved trained models
├── outputs/                   # Generated reports
├── utils/
│   ├── simulator.py           # Vehicle telemetry simulator
│   ├── feature_engineering.py # Signal processing & feature extraction
│   └── coach.py               # Claude API coaching report generator
├── train.py                   # Model training pipeline
├── predict.py                 # Run inference on new telemetry
├── app.py                     # Streamlit dashboard
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your Claude API key
```bash
# Create a .env file
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```
Get your key from: https://console.anthropic.com

### 3. Generate training data
```bash
python utils/simulator.py
```

### 4. Train the model
```bash
python train.py
```

### 5. Run the dashboard
```bash
streamlit run app.py
```

---

## 🧠 Driver Behaviour Classes

| Class | Description | Risk Level |
|-------|-------------|------------|
| `SMOOTH` | Consistent speed, gentle braking, smooth cornering | ✅ Low |
| `AGGRESSIVE` | Hard acceleration, sharp braking, high G-force | 🔴 High |
| `FATIGUED` | Micro-corrections, lane drift patterns, erratic throttle | 🟠 Medium-High |
| `SPORTY` | High RPM, controlled aggression, track-like patterns | 🟡 Medium |
| `HIGHWAY_CRUISE` | Steady speed, low variability, motorway profile | ✅ Low |

## 📊 Features Used

- **Speed dynamics**: mean, std, max speed per window
- **Braking intensity**: brake force, deceleration rate, panic brake events
- **Throttle behaviour**: throttle jerk, acceleration bursts
- **Lateral dynamics**: steering angle variance, lateral G-force peaks
- **Engine load**: RPM patterns, gear shift frequency
- **Safety signals**: time-to-collision proxy, hard event count

---

## 💡 Why This Matters for BMW

BMW's ConnectedDrive and Driver Experience systems use exactly this kind of pipeline:
- Personalised driver profiles → adaptive suspension & powertrain tuning
- Risk scoring → proactive safety interventions
- Coaching feedback → BMW Driving Experience programmes

This project demonstrates real understanding of the automotive AI stack.

---

*Built with Python, XGBoost, Claude API, Streamlit*
