<div align="center">

# 🏥 ReadmitRadar
### A Neural Network for Predicting 30-Day Hospital Readmission — Built from First Principles

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Sequential%20API-D00000?logo=keras&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-Preprocessing-F7931E?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-success)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

*Binary classification on real-shaped clinical data — with manual forward propagation, a loss-function bake-off, and five documented regularization trials to actually diagnose (not just patch) overfitting.*

</div>

---

## Why this project exists

Most "neural network from scratch" tutorials stop at `model.fit()` and a single accuracy
number. This project deliberately doesn't. It uses a healthcare readmission dataset to walk
through the full reasoning chain a practitioner actually needs:

- What is a forward pass actually computing, numerically, before any training happens?
- Why is Binary Cross-Entropy the right loss for this problem and not MSE — not as a rule
  someone said once, but demonstrated side-by-side on the same data?
- When a network overfits, is "add more dropout" actually working, or just suppressing
  training accuracy until the gap *looks* smaller? (Spoiler: the notebook catches this and
  changes the architecture instead.)

The result is a fully reproducible ANN pipeline, five regularization trials compared
head-to-head, and a final report written the way you'd actually hand it to a hospital
analytics team — including the honest limits of what the model can and can't do.

---

## Problem Statement

A hospital analytics team wants to flag, **before discharge**, which patients are likely to
be readmitted within 30 days — so follow-up calls, home-care referrals, and medication
reviews can be targeted at the patients who need them most, instead of spread evenly across
everyone.

**Task:** Binary classification — `readmitted_30_days` (1 = readmitted within 30 days, 0 = not).

---

## Dataset

`healthcare_readmission_dataset.csv` — **5,000 hospital visits, 14 features + target.**
Near-balanced classes (53.9% not readmitted / 46.1% readmitted), no missing values, no
duplicates.

| Column | Type | Description |
|---|---|---|
| `patient_id` | identifier | Dropped before modeling (no predictive value) |
| `age` | numeric | Patient age in years |
| `bmi` | numeric | Body Mass Index |
| `systolic_bp` / `diastolic_bp` | numeric | Blood pressure readings |
| `heart_rate` | numeric | Resting heart rate (bpm) |
| `glucose_level` | numeric | Blood glucose level |
| `num_previous_admissions` | numeric | Count of prior hospital admissions |
| `length_of_stay` | numeric | Days spent in hospital for this visit |
| `medication_count` | numeric | Number of medications prescribed |
| `chronic_condition` | categorical | cardiac / diabetes / hypertension / respiratory / none |
| `smoker` | categorical (binary) | yes/no |
| `insurance_type` | categorical | private / public / self-pay |
| `followup_appointment` | categorical (binary) | Whether a follow-up was scheduled |
| `readmitted_30_days` | **target** | 0 = not readmitted, 1 = readmitted within 30 days |

---

## Methodology

### 1. Data Understanding & Preprocessing
EDA (shape, dtypes, duplicates, nulls, summary stats), univariate distributions, bivariate
crosstab heatmaps against the target, and a full correlation matrix. Categorical columns
encoded with binary mapping (2-category columns) or one-hot encoding (multi-category
columns); features scaled with `StandardScaler` fit on the training split only.

### 2. Forward Propagation & Loss Intuition
A one-hidden-layer forward pass is computed **by hand with NumPy** — random weight
initialization, ReLU activation, sigmoid output — to build real intuition for what a network
computes before any training occurs. Then **MSE vs. Binary Cross-Entropy** are compared
head-to-head on identical architectures to show empirically (not just by assertion) why BCE
is the correct loss for a sigmoid-output binary classifier.

### 3. Baseline ANN
A two-hidden-layer feed-forward network (64 → 32, ReLU, sigmoid output) trained with Adam
and BCE loss — establishing a baseline and immediately surfacing clear overfitting
(training accuracy climbing well past validation).

### 4. Hyperparameter Tuning & Regularization
Systematic comparison of **Adam, SGD, and RMSprop** across two batch-size/epoch
configurations each, with Dropout layers added as a first regularization pass — all results
consolidated into a single comparison table rather than eyeballed individually.

### 5. Advanced Training & Final Model
Five documented trials combining **Batch Normalization, Dropout, L1/L2 regularization,
EarlyStopping, and ReduceLROnPlateau**. Trials 1–4 only escalated dropout/L2 strength on the
same wide architecture — and the notebook explicitly diagnoses *why that stopped helping*
(it was shrinking training accuracy, not improving test accuracy). Trial 5 changes the
architecture itself (narrower 64 → 32 → 16 layers + L1/L2) and produces the best result of
the five trials.

---

## Final Model Architecture (Trial 5 — Recommended)

```
Input (19 features)
 → Dense(64, ReLU, L1+L2 regularized) → BatchNorm → Dropout(0.3)
 → Dense(32, ReLU, L2 regularized)    → BatchNorm → Dropout(0.3)
 → Dense(16, ReLU, L2 regularized)    → BatchNorm → Dropout(0.3)
 → Dense(1, Sigmoid)
```
Optimizer: **Adam** · Loss: **Binary Cross-Entropy** · Batch size: 32 · Up to 100 epochs with
**EarlyStopping** (patience 6) and **ReduceLROnPlateau** (patience 4).

---

## Results

| Stage | Architecture | Regularization | Train Acc | Test Acc | Train–Test Gap |
|---|---|---|---|---|---|
| Baseline | 64 → 32 | None | 0.8687 (last epoch) | 0.5860 | large (overfits) |
| Hyperparameter tuning (best) | 128 → 64 → 32 | Dropout 0.2 | 0.6332–0.7370 | 0.5670–0.6390 | 0.0270–0.1250 |
| Final — Trial 1 | 128 → 64 → 32 | Dropout 0.2 + L2 | 0.7210 | 0.6190 | 0.1020 |
| Final — Trial 2 | 128 → 64 → 32 | Dropout 0.3 + L2 | 0.7130 | 0.6120 | 0.1010 |
| Final — Trial 3 | 128 → 64 → 32 | Dropout 0.3–0.5 + L2 | 0.7080 | 0.6080 | 0.1000 |
| Final — Trial 4 | 128 → 64 → 32 | Dropout 0.5 + L2 | 0.6948 | 0.6150 | 0.0798 |
| **Final — Trial 5** ✅ | **64 → 32 → 16 (narrower)** | **Dropout 0.3 + L1/L2** | **0.7013** | **0.6260** | **0.0753** |

**Trial 5 is the recommended model** — the highest test accuracy *and* the smallest
train-test gap of the five trials, achieved by changing the network's shape rather than just
escalating regularization on the same wide architecture.

> **Note:** the single highest test score in the project (0.6390) came from an SGD run during
> hyperparameter tuning, with a 0.029 gap. As the notebook explains, SGD's small gap mostly
> reflects fitting the training data less in the same number of epochs, so Trial 5 is
> recommended for its controlled training (early stopping and learning-rate scheduling). The
> two are about one point apart.

### Key takeaway
Test accuracy across **every** configuration tried — three optimizers, multiple
dropout/L2 strengths, five architecture trials — stayed within a tight **0.567–0.639** band.
That consistency points to a **feature ceiling**, not a tuning problem: the 19 available
features place a practical limit on accuracy, and the next real gains will come from feature
engineering (interaction terms, clinically meaningful binning, additional data sources) or
trying tree-based model families — not further architecture search on the same inputs.

---

## Repository Structure

```
.
├── ReadmitRadar_Neural_Network_Hospital_Readmission.ipynb   # Full annotated notebook
├── requirements.txt                                          # Python dependencies
├── README.md                                                  # This file
└── healthcare_readmission_dataset.csv                         # Dataset (not included — see note below)
```

> **Note:** The dataset is not bundled in this repo. Provide your own
> `healthcare_readmission_dataset.csv` with the schema described above, or swap in a dataset
> of your own — the pipeline is schema-driven and adapts to any binary-target tabular health
> dataset with minor column-name edits.

---

## How to Run

```bash
git clone https://github.com/dshah-ds/readmitradar.git
cd readmitradar
pip install -r requirements.txt
jupyter notebook ReadmitRadar_Neural_Network_Hospital_Readmission.ipynb
```

The notebook was built in Google Colab, so the data-loading cell uses
`google.colab.files.upload()`. Running locally, just replace that one cell with:
```python
df = pd.read_csv("healthcare_readmission_dataset.csv")
```
Every other cell runs unchanged.

---

## Practical Recommendations (from the Final Report)

1. Use model risk scores to **prioritize**, not replace, clinical follow-up planning.
2. Watch patients with multiple previous admissions and longer stays — the most skewed,
   highest-signal features.
3. Treat medication count + chronic condition as secondary screening signals.
4. Use the model as a **triage aid**, not a standalone discharge decision — at 62.6% test
   accuracy, it's meaningfully better than chance but not precise enough to act on alone.
5. Prioritize feature engineering over further architecture tuning for the next iteration.
6. Retrain periodically — readmission patterns shift as protocols and populations change.

---

## Tech Stack

`Python` · `Pandas` · `NumPy` · `scikit-learn` · `TensorFlow / Keras` · `Matplotlib` · `Seaborn`

---

## Author

**Devam Shah** · Data Scientist & Analytics Engineer

- Website: [devam-shah.com](https://devam-shah.com)
- LinkedIn: [linkedin.com/in/devam-a-shah](https://www.linkedin.com/in/devam-a-shah)
- GitHub: [github.com/dshah-ds](https://github.com/dshah-ds)
- Medium: [@devamshah26](https://medium.com/@devamshah26)
- Blog: [One Notebook at a Time](https://onenotebookatatime.blogspot.com)

If this was useful, a ⭐ on the repo is always appreciated — and I'm always happy to talk
through the modeling choices, overfitting diagnosis, or what I'd try next.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
