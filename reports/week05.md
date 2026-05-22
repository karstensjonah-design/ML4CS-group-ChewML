# Week 05 Report — Machine Learning for Smart and Connected Systems (ML4SCS)

## Weekly Goal

Define the project from scratch, collect the first dataset, and build a complete baseline pipeline — from raw sensor data to a working classifier — within the first week. This is the first report for this project, as the course was joined in Week 5 after leaving a previous group.

---

## Work Done This Week

### 0. Project Setup

After joining the course in Week 5 as a solo student, the project was defined and set up from scratch this week. The project investigates whether AirPod IMU sensor data can be used to classify foods based on chewing patterns. The central research question is:

> *Can the chewing motion captured via in-ear IMU sensors (AirPods Pro) distinguish between different foods — and between eating and not eating?*

Four food classes were defined: **apple**, **chewing gum**, **skyr/yogurt**, and **still** (not eating). Data was recorded using the **Sensor Logger** iOS app with AirPods Pro, capturing accelerometer, gyroscope, and Euler angle data (pitch, roll, yaw) at approximately **50 Hz**.

The GitHub repository was set up this week and organised into `data/raw/` for raw recordings, `src/` for processing scripts, `notebooks/` for exploration, and `results/` for plots and outputs.

---

### 1. Data Work

**Data collection:** A total of **12 sessions** were self-recorded — 3 per class, each approximately 60 seconds long. All sessions were exported as CSV files from Sensor Logger and stored as ZIP archives in `data/raw/`.

**Sensor channels recorded:**
- Linear acceleration: `accelerationX/Y/Z`
- Gravity vector: `gravityX/Y/Z`
- Gyroscope: `rotationRateX/Y/Z`
- Euler angles: `pitch`, `roll`, `yaw`

**Preprocessing steps applied to each session:**
1. The first and last 5 seconds of each recording were trimmed to remove startup artefacts and transition noise.
2. Linear acceleration was computed by subtracting the gravity vector from raw acceleration: `lin_x = accelerationX − gravityX` (and analogously for Y, Z).
3. A scalar magnitude signal was computed: `magnitude = √(lin_x² + lin_y² + lin_z²)`

The magnitude signal proved to be the most informative single channel, as it captures overall head/jaw movement intensity independent of sensor orientation.

---

### 2. Analysis / Modelling Work

**Feature extraction:** A total of **37 features** were extracted per session from three domains:

*Time domain (acceleration & rotation):*
- Mean, standard deviation, and maximum of linear acceleration (X, Y, Z) and magnitude
- Stillness ratio (fraction of samples with magnitude < 0.02)
- Movement events (samples above 75th percentile of magnitude)
- Mean, standard deviation, and maximum of gyroscope (X, Y, Z)

*Euler angle domain:*
- Mean, standard deviation, and range of pitch, roll, and yaw

*Frequency domain (FFT-based):*
- Dominant chewing frequency (peak frequency in 0.5–4 Hz band via Welch PSD)
- Chewing band power (summed PSD in 0.5–4 Hz)
- Total signal power
- Rhythmicity (chewing band power / total power)

**Model:** A **Random Forest classifier** was trained on the 12-session feature matrix. Leave-One-Out cross-validation (LOO-CV) was used to evaluate generalisation given the small dataset.

---

### 3. Repository / Documentation Work

- Repository initialised with `.gitignore`, `README.md`, and `requirements.txt`
- Folder structure created: `data/raw/`, `src/`, `notebooks/`, `results/`, `reports/`
- Scripts written: `preprocessing.py`, `train.py`, `evaluate.py`, `airpods_pipeline_v2.py`, `build_feature_matrix.py`
- Feature matrix exported to `feature_matrix_final.csv` (12 rows × 37 features)
- All result plots saved under `results/`

---

## Experiments Conducted

| Experiment | Change Made | Result | Interpretation |
|---|---|---|---|
| Exp 1 | Initial signal exploration: still vs. restless (T-1, T-2) | Visual distinction clearly visible in magnitude | Confirmed that IMU signal captures motion differences |
| Exp 2 | Expanded to all 4 food classes | Distinct signal patterns per class visible | Different foods produce characteristic chewing rhythms |
| Exp 3 | Added frequency-domain features (FFT/Welch) | Feature separability improved significantly | Chewing frequency (0.5–4 Hz band) is highly class-specific |
| Exp 4 | Random Forest + LOO-CV on full feature matrix | **92% LOO-CV accuracy (11/12 correct)** | Strong result for a 12-session dataset |

---

## Results

### Signal Comparison — All 4 Classes

The magnitude signal shows clearly distinct patterns per class. Still sitting produces a near-flat signal with very low variance, while all three food classes show rhythmic oscillations at different amplitudes and frequencies.

![Signal comparison of all 4 classes](images/alle_klassen_signal.png)

*Figure 1: Magnitude signal over time for one representative session per class. Still (blue) is flat, apple (red) shows irregular chewing bursts, skyr (green) shows steady rhythmic movement, and chewing gum (purple) produces rapid, lower-amplitude oscillations.*

---

### Feature Importance & Confusion Matrix (LOO-CV)

![ML results: LOO Confusion Matrix and Feature Importance](images/ml_final.png)

*Figure 2 (left): Leave-One-Out confusion matrix. Apfel, Kaugummi, and Skyr are classified perfectly (3/3). One Still session was misclassified as Kaugummi, yielding **92% overall LOO accuracy**.*

*Figure 2 (right): Top 12 feature importances from the Random Forest. `rot_y_mean`, `chew_band_power`, and `rhythmicity` are the most discriminative features — confirming that the frequency-domain approach adds significant value beyond simple time-domain statistics.*

---

### Feature Comparison Across Classes

![Feature comparison across all 4 classes](images/alle_klassen_features.png)

*Figure 3: Normalised feature values for the 8 most important features across all classes. Apple shows the highest chewing band power and rot_x_std, while Still has the highest rhythmicity due to its constant low-level noise floor. Chewing gum and skyr are separated primarily by dominant chewing frequency and yaw range.*

---

## Challenges

The main challenge was the **very small dataset** (12 sessions total, 3 per class), which makes LOO-CV the only viable evaluation strategy and limits confidence in the result. The one misclassified still session highlights that resting head motion can occasionally resemble slow chewing, particularly for chewing gum. A larger dataset would improve robustness considerably.

A second challenge was the **frequency-domain feature extraction**: choosing the right frequency band (0.5–4 Hz) required manual inspection of power spectral densities per class to confirm it captured actual chewing and not artefact noise.

---

## Key Insights

- Chewing creates measurable and class-specific rhythmic patterns in AirPod IMU data even at 50 Hz.
- Frequency-domain features (chewing band power, rhythmicity) are the most discriminative — more so than raw time-domain statistics.
- The `rot_y_mean` gyroscope feature being the top predictor suggests that jaw-opening direction (left-right rotation) differs systematically between food types.
- 92% LOO accuracy on 12 sessions is a strong baseline that justifies extending the dataset.

---

## Plan for Next Week

- Record additional sessions to expand the dataset 
- Apply proper train/test split evaluation once dataset is large enough
- Investigate the one misclassified Still session (possible recording artefact)
- Explore feature selection to reduce dimensionality
- Working on the presentation

---

## Contributions

- Jonah Karstens: full project (solo) — project definition, data collection, preprocessing, feature engineering, modelling, evaluation, repository setup

---

## Extended Work — Dataset Expansion & Pipeline Refinement

### Additional Data Collection

The dataset was expanded from **12 to 24 sessions** (6 per class) by recording additional sessions with all four food classes (Apfel 4–6, Kaugummi 4–6, Skyr 4–7, Still 4–9). With the expanded dataset the baseline LOO-CV accuracy dropped from 92 % to **87.5 %** (21/24 correct), reflecting the greater within-class variability introduced by the new sessions — particularly in *Still* and *Skyr*, where occasional head movements contaminate the signal.

### Signal Processing Approaches Investigated

Three approaches were tested to improve classification accuracy by cleaning the signal before feature extraction.

**Approach 1 — Low-Pass Filter (Butterworth, 4th order, zero-phase)**

A Butterworth low-pass filter was applied to the magnitude signal at cutoffs of 2, 5, 10, and 20 Hz and compared against the unfiltered baseline. The chewing band (0.5–4 Hz) lies well below any tested cutoff, so all filters above 4 Hz preserve the relevant signal. Accuracy was essentially unchanged across all settings. The conclusion is that high-frequency noise is not the limiting factor here — the problem lies in *sustained* low-frequency contamination from head movement, which a low-pass filter cannot address.

![Low-pass filter comparison — grid view](images/lp_filter_grid.png)

*Figure 4: Magnitude signal for one representative session per class (columns), filtered at each cutoff (rows). Within each column the y-axis is shared. Filters above 5 Hz leave the chewing rhythm intact; the 2 Hz filter attenuates the upper chewing band.*

**Approach 2 — Hampel Spike Filter (5 σ)**

The Hampel filter detects isolated single-sample outliers by comparing each sample to a rolling median and flagging deviations beyond `5 × 1.4826 × MAD`. At 3 σ the filter incorrectly removed genuine chewing peaks (106 samples in Apfel, 110 in Kaugummi), reducing accuracy. At **5 σ** only true artefacts are caught (≤ 5 per session in Still and Skyr), leaving the chewing signal intact. LOO-CV accuracy after Hampel filtering remained at **87.5 %** — confirming that isolated outliers are not the root cause of misclassifications.

![Hampel spike filter — spike detection per class](images/hampel_filter.png)

*Figure 5: Hampel filter output (window = 0.5 s, 5 σ) for one session per class. Grey = original signal, blue = cleaned signal, red dots = detected spikes. True isolated artefacts are flagged in Still and Skyr; chewing peaks in Apfel and Kaugummi are correctly left untouched.*

**Approach 3 — Movement-Segment Exclusion (Adaptive Threshold)**

The key insight is that misclassified sessions contain *sustained head-movement phases* — several seconds of high-amplitude signal that are physically unrelated to eating but corrupt the feature statistics. An adaptive threshold was computed per session:

> `threshold = max(0.02 g, 5 × session_median_magnitude)`

This scales automatically to each session's own baseline without requiring class labels, and produces thresholds that match the domain intuition (Still: 0.020 g, Skyr: 0.060 g, Apfel: 0.095 g). A rolling maximum over 1 s (50 samples) then excludes entire contaminated windows rather than just the peak sample. After removing movement segments, LOO-CV accuracy rose to **100 %** (24/24 correct).

![Movement-segment exclusion — adaptive threshold per class](images/movement_exclusion.png)

*Figure 6: Adaptive movement-segment exclusion for one session per class. Red-shaded regions are excluded; the coloured line shows the retained signal; the dashed red line is the adaptive threshold. Still excludes the head-movement burst at ~13–15 s; Skyr excludes intermittent high-amplitude bursts.*

### Results Summary

| Pipeline | LOO-CV Accuracy | Correct / Total |
|---|---|---|
| Baseline (24 sessions, no filter) | 87.5 % | 21 / 24 |
| + Hampel spike removal (5 σ) | 87.5 % | 21 / 24 |
| **+ Movement-segment exclusion** | **100 %** | **24 / 24** |

![Confusion matrix comparison — all three pipelines](images/confusion_comparison.png)

*Figure 7: LOO-CV confusion matrices for the three pipeline variants. Left: baseline — three Skyr sessions misclassified as Still. Centre: Hampel filter — no change. Right: movement exclusion — perfect classification across all four classes.*

### Reproducible Notebook

The full pipeline is implemented in `notebooks/02_analysis.ipynb` (44 cells). The notebook dynamically scans `data/raw/` for ZIP archives at startup, so adding new sessions requires only dropping a new file into that folder and re-running. A final cell regenerates and saves all four report figures to `reports/images/` for documentation.
