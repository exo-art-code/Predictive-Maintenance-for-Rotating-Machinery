# Predictive Maintenance for Rotating Machinery

> **Machine learning-based bearing fault detection and Remaining Useful Life (RUL) prediction using the CWRU vibration dataset.**

---

## Problem Statement

Unplanned equipment failure in rotating machinery (motors, pumps, gearboxes) costs the manufacturing industry billions annually.  
This project builds an end-to-end ML pipeline that:

1. **Detects bearing faults** from raw vibration signals (4-class classification)
2. **Predicts Remaining Useful Life** so maintenance can be scheduled before failure

---

## Dataset

**Case Western Reserve University (CWRU) Bearing Fault Dataset** — the most widely cited open dataset in condition monitoring research.

| Property | Value |
|---|---|
| Sensor | Drive-end accelerometer |
| Sampling rate | 12,000 Hz |
| Fault types | Normal, Inner Race, Ball, Outer Race |
| Fault sizes | 0.007", 0.014", 0.021" (diameter) |
| Load conditions | 0–3 HP |

The dataset is **auto-downloaded** on first run — no manual steps required.

---

## Methodology

```
Raw vibration signal
        │
        ▼
 Sliding window (1024 samples, 50% overlap)
        │
        ▼
 Feature extraction (16 features)
  ├── Time domain:  RMS, Kurtosis, Crest Factor, Shape Factor, …
  ├── Frequency:    Mean Frequency, Band Energy Ratio, Peak Frequency, …
  └── Envelope:     Hilbert transform → amplitude demodulation
        │
        ▼
 Classification   →  Normal / Inner Race / Ball / Outer Race
 RUL Regression   →  Remaining Useful Life (%)
```

### Features Extracted

| Feature | Domain | Physical Meaning |
|---|---|---|
| RMS | Time | Signal energy — correlates with vibration severity |
| Kurtosis | Time | Impulsiveness — jumps sharply on fault onset |
| Crest Factor | Time | Peak-to-RMS ratio — flags transient impacts |
| Shape Factor | Time | Dimensionless, load-independent |
| Clearance Factor | Time | Sensitive to early ball defects |
| Mean Frequency | Frequency | Spectral centroid — shifts with fault |
| Band Energy Ratio (0–3 kHz) | Frequency | Energy concentration in fault bands |
| Peak Frequency | Frequency | Dominant excitation frequency |
| Spectral Kurtosis | Frequency | Non-stationarity in the spectrum |
| Envelope RMS | Envelope | Amplitude modulation by fault impulses |
| Envelope Kurtosis | Envelope | Best single early-fault indicator |

### Models Compared

| Model | Why used |
|---|---|
| Random Forest | Robust, interpretable, handles feature interactions |
| SVM (RBF kernel) | Strong with small, well-separated feature sets |
| XGBoost | Best accuracy, built-in feature importance |
| Gradient Boosted Regressor | RUL regression |

---

## Results

| Model | 5-Fold CV Accuracy |
|---|---|
| Random Forest | ~97% |
| SVM (RBF) | ~95% |
| **XGBoost** | **~98%** |

- XGBoost test set accuracy: **~98%**
- RUL prediction RMSE: **~5%**



## Project Structure

```
predictive_maintenance_ml/
├── main.py                          # Full pipeline — run this
├── requirements.txt
├── src/
│   ├── data_loader.py               # Downloads and segments CWRU .mat files
│   ├── feature_engineering.py       # 16 time/frequency/envelope features
│   ├── models.py                    # RF, SVM, XGBoost classifiers + RUL regressor
└── notebooks/
    └── exploration.ipynb            # Interactive EDA
```

---

## Setup & Run


clone this repo
cd predictive_maintenance_ml

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python main.py

On first run, the script downloads nearly 50 MB of .mat files from the CWRU server.

---

## Key Engineering Concepts Used

- **Vibration signal analysis** — standard technique in mechanical condition monitoring
- **Envelope analysis (Hilbert transform)** — separates fault modulation from carrier
- **Characteristic fault frequencies** — BPFI, BPFO, BSF from bearing geometry
- **Statistical signal features** — the same features used in ISO 13373 (machinery monitoring standard)
- **Remaining Useful Life** — core metric in ISO 13381 prognostics standard

---

## Author

**Naimisha** — B.Tech Mechanical Engineering, IIT Madras  
Contact: [me24b035@smail.iitm.ac.in]
