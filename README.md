# ChewML — Food Classification via AirPod IMU

**Semester project · Machine Learning for Smart and Connected Systems**  
Jonah Karstens · Solo project

---

## Members

<img src="reports/images/jonah_karstens.jpeg" alt="Jonah Karstens" width="180" />

**Jonah Karstens** — full project (solo)

---

## Research Question

> Can chewing patterns captured via in-ear IMU sensors (AirPods Pro) be used to classify different foods — and distinguish eating from not eating?

---

## Project Idea

Most food-tracking approaches rely on manual input or cameras. This project explores a more passive alternative: using the motion sensors already built into AirPods Pro. Chewing different foods creates distinct jaw-movement patterns that show up in accelerometer, gyroscope, and orientation data sampled at ~50 Hz via the **Sensor Logger** iOS app.

Four classes are being explored: **apple**, **chewing gum**, **skyr/yogurt**, and **still** (not eating).

---

## Status

This project is in early exploration. The current pipeline — preprocessing, feature extraction, and a first classifier — is a rough proof of concept to understand the data and test whether the idea is even feasible. Everything from feature selection to the final model is still open.

| Phase | Status |
|---|---|
| Data collection (12 sessions, 4 classes) | ✅ First round done |
| Preprocessing & signal exploration | ✅ Rough pipeline in place |
| Feature extraction (37 features) | ⚠️ Preliminary — which features matter is still open |
| First classifier (Random Forest, LOO-CV) | ⚠️ Proof of concept — 92% on 12 sessions, not generalizable yet |
| More recordings & proper dataset | ⏳ Planned |
| Feature selection | ⏳ Planned |
| Final model & evaluation | ⏳ Planned |

---

## First Results

A Random Forest trained on the current 12-session dataset achieves **92% Leave-One-Out accuracy**. This is an encouraging early result but should be treated with caution — the dataset is very small and the feature set has not been optimized yet.

![Confusion Matrix and Feature Importance](reports/images/ml_final.png)

---

## Project Structure

```
data/raw/          Raw recordings (ZIP archives, one per session)
notebooks/         Exploratory analysis
reports/           Weekly progress reports
results/           Plots and outputs from experiments
src/               Preprocessing, training, and evaluation scripts
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Weekly Reports

- [Week 5](reports/week05.md)
