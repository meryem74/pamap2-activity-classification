# PAMAP2 Activity Classification — Standing vs. Walking

End-to-end biomedical signal-processing and machine-learning pipeline on the
**PAMAP2 Physical Activity Monitoring** dataset, using the **ankle IMU
accelerometer**. The project loads and explores the raw signals, performs
band-pass filtering and windowed feature extraction, tests feature significance,
and trains four classifiers to distinguish **standing** from **walking** — first
with an 80/20 split, then with **leave-one-subject-out cross-validation
(LOSO-CV)**.

The methodological framework (Butterworth band-pass filtering → windowing →
feature extraction → 80/20 + LOSO-CV evaluation) follows Tokmak & Semiz's PAMAP2
study, adapted here from chest SCG / MET estimation to ankle-acceleration
standing-vs-walking classification.

---

## Dataset

[PAMAP2 Physical Activity Monitoring](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)
— 9 subjects, 3 Colibri IMUs (hand / chest / ankle) sampled at ~100 Hz, plus
heart rate. This project uses the ankle IMU's ±16g 3-axis acceleration channels.
The notebook downloads and unpacks the dataset automatically.

---

## Pipeline

**Part 1 — single subject (subject101):**
1. Load the data; inspect structure, sampling rate (100 Hz) and signal length.
2. Average heart rate for standing, running and rope-jumping sessions.
3. Band-pass filter (0.5–20 Hz) the ankle acceleration X/Y/Z; plot the spectra and locate dominant frequencies.
4. Compute the acceleration magnitude of the walking session: |a| = √(x² + y² + z²).
5. Segment the magnitude signal into non-overlapping 4-second windows.
6. Extract 6 features per window.
7. Write the feature matrix to CSV (with headers).

**Part 2 — all subjects (standing vs. walking):**
1. Filter the ankle X/Y/Z signals (0.5–20 Hz) for every subject.
2. Take the acceleration magnitude.
3. Split into standing and walking regions.
4. Segment both into 4-second windows with 50% overlap.
5. Extract features → common CSV with columns: 6 features + `subject_id` + `activity_id` (0 = standing, 1 = walking).
6. Test feature significance between the two classes (Mann-Whitney U).
7. Binary classification with **SVM, Random Forest, Logistic Regression, XGBoost** — 80/20 split, then LOSO-CV. `subject_id` is used only to form LOSO folds, never as a feature or label.
8. Report accuracy, recall (sensitivity), precision and F1-score.

---

## Features extracted (per 4-second window)

| Feature | Description |
|---|---|
| `energy` | Σ x[n]² |
| `shannon_entropy` | Shannon entropy from a 32-bin amplitude histogram, −Σ pᵢ log₂ pᵢ |
| `max_amplitude` | Maximum value in the window |
| `max_location` | Index of the maximum value |
| `min_amplitude` | Minimum value in the window |
| `min_location` | Index of the minimum value |

---

## Results

**Statistical significance (Mann-Whitney U, α = 0.05).** The four amplitude/energy
features differ significantly between classes; the two location features do not
(peak/trough position within a window is essentially random for both activities).

| Feature | p-value | Significant |
|---|---|---|
| energy | < 0.001 | Yes |
| shannon_entropy | 8.5 × 10⁻⁴⁷ | Yes |
| max_amplitude | < 0.001 | Yes |
| max_location | 0.939 | No |
| min_amplitude | < 0.001 | Yes |
| min_location | 0.264 | No |

**Classification — 80/20 split**

| Model | Accuracy | Recall | Precision | F1 |
|---|---|---|---|---|
| SVM | 0.991 | 0.983 | 1.000 | 0.991 |
| Random Forest | 0.991 | 0.983 | 1.000 | 0.991 |
| Logistic Regression | 0.991 | 0.983 | 1.000 | 0.991 |
| XGBoost | 0.988 | 0.983 | 0.996 | 0.989 |

**Classification — LOSO-CV (subject-averaged)**

| Model | Accuracy | Recall | Precision | F1 |
|---|---|---|---|---|
| SVM | 0.988 | 0.978 | 1.000 | 0.989 |
| Random Forest | 0.990 | 0.983 | 0.998 | 0.991 |
| Logistic Regression | 0.990 | 0.982 | 0.999 | 0.991 |
| XGBoost | 0.989 | 0.981 | 0.998 | 0.990 |

LOSO-CV results stay essentially identical to the 80/20 split, indicating that
the motion-based features generalize across subjects.

> Note: subject109 has no standing/walking regions in the protocol and is
> automatically skipped, so Part 2 runs on 8 subjects.

---

## How to run

**Google Colab (recommended):** open `PAMAP2_Colab.ipynb` in Colab and run all
cells. The notebook downloads the dataset, produces all plots, writes the CSV
files, and prints the statistics and classification results.

**Local:**
```bash
pip install numpy pandas scipy scikit-learn matplotlib xgboost
jupyter notebook PAMAP2_Colab.ipynb
```

---

## Repository structure

```
PAMAP2_Colab.ipynb     # full pipeline (Part 1 + Part 2), Colab-ready
PAMAP2_Rapor.docx      # technical report (Turkish)
README.md
```

---

## References

1. A. Reiss and D. Stricker, "Introducing a new benchmarked dataset for activity monitoring," *ISWC*, 2012.
2. A. Reiss and D. Stricker, "Creating and benchmarking a new dataset for physical activity monitoring," *PETRA*, 2012.
3. F. Tokmak and B. Semiz, "Unveiling the Relationships Between Seismocardiogram Signals, Physical Activity Types and Metabolic Equivalent of Task Scores," *IEEE*.
