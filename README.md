# PathSnap: Map-Matching & Route Reconstruction

UE24CS352A Machine Learning hackathon. We match noisy GPS observations to road-network arcs with a
candidate-scoring classifier, then pick a connected route with HMM/Viterbi decoding.
The model runs behind a FastAPI backend (Laptop 1), and the evaluator uses a Streamlit UI (Laptop 2).

## Pipeline

```
.track (GPS) ─┐
.nodes/.arcs ─┼─> project to metres ─> candidate arcs (≤50 m, top-8) ─> features ─> classifier P(arc|point)
              │                                                                          │
.route ───────┘ (training labels only, via monotonic alignment)          HMM/Viterbi over candidates
                                                                                         │
                                                                    one road arc per GPS point
```

| Stage | File |
|---|---|
| Load / clean raw files | `src/pathsnap/io.py` |
| Local metric projection, geometry | `src/pathsnap/geo.py` |
| Candidate generation, shortest paths | `src/pathsnap/network.py` |
| Per-point reference labels from `.route` | `src/pathsnap/labels.py` |
| Feature engineering | `src/pathsnap/features.py` |
| HMM + Viterbi | `src/pathsnap/hmm.py` |
| Metrics | `src/pathsnap/metrics.py` |
| End-to-end inference | `src/pathsnap/pipeline.py` |
| Folium maps | `src/pathsnap/viz.py`, `scripts/make_map.py` |
| API | `api/main.py` |
| UI | `ui/app.py` |

## Setup

Requires **Python 3.11–3.13** (the team uses 3.13).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements-lock.txt
```

**Everyone installs `requirements-lock.txt`**, which pins exact versions. Saved models (`.joblib`)
only load reliably with the same scikit-learn / LightGBM versions they were trained with, so
all team members, and both demo laptops, must use identical versions.
`requirements.txt` only lists the minimum versions and is not used for installing.
Check your version with `python -c "import sklearn; print(sklearn.__version__)"`. It should print `1.7.2`.

The raw data is **not** in git, because it is too large. Unzip `Student folder.zip` so that you get:

```
data/raw/train/<trip_id>/...
data/raw/student_test/<trip_id>/...
```

## Running

Full pipeline, in order:

```bash
python scripts/eda.py                         # EDA tables, plots, Folium maps
python scripts/build_features.py              # -> data/processed/*.parquet
python scripts/train_models.py                # all 5 models: CV + student_test -> results/experiments.csv
python scripts/plot_mlp_loss.py               # MLP train vs validation loss curves
python scripts/bayes_opt.py 30                # Optuna on LightGBM -> models/final_model.joblib
python scripts/tune_hmm.py final_model        # HMM params on train OOF -> models/hmm_params.json
python scripts/evaluate_hmm.py final_model    # classifier-only vs HMM per trip (repeat for each model)
python scripts/plot_comparison.py             # results/model_comparison.csv + comparison plots
python scripts/plot_confusion.py final_model  # confusion matrix
python scripts/make_map.py 00000077           # -> results/maps/00000077.html
```

## Deployment (two laptops)

Full step-by-step guide and troubleshooting: **[docs/DEMO_CHECKLIST.md](docs/DEMO_CHECKLIST.md)**.
Use a phone hotspot for both laptops, because campus Wi-Fi often blocks laptop-to-laptop traffic.

**Laptop 1 (API):**
1. Once, as Administrator: `setup_firewall.bat` (opens port 8000)
2. `pip install -r requirements-lock.txt`
3. `run_api.bat C:\path\to\evaluator_trips`. It prints this laptop's IP, and every request is logged in this window.

**Laptop 2 (UI):**
1. `pip install -r requirements-lock.txt`
2. `run_ui.bat <laptop1-ip>`, then click **Test connection** in the sidebar.

Manual equivalent (PowerShell uses `$env:VAR="..."`; CMD uses `set VAR=...`):
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
```bash
python -m streamlit run ui/app.py
```

The ground-truth `.route` is read only inside the API, after prediction, to compute accuracy. It is never given to the model and never sent to the UI.

## Results (student_test, 10 trips)

**Final system:** LightGBM tuned with Bayesian optimisation (`models/final_model.joblib`) +
HMM/Viterbi with β = 1.5, unreachable = −120 (`models/hmm_params.json`). The API loads both automatically.

### Model comparison

All models use the same features, the same 5 trip-grouped CV folds on the 80 train trips, and the same
HMM parameters. Source: `results/model_comparison.csv`, plots `results/plots/model_comparison*.png`.

| Model | Family | CV macro-F1 (± std) | Test Top-1 | Test macro-F1 | Log loss | Top-1 + HMM¹ | Continuity + HMM¹ |
|---|---|---|---|---|---|---|---|
| Logistic Regression | Linear | 0.8967 ± 0.021 | 83.08% | 0.9013 | 0.1698 | 91.35% | 0.904 |
| Random Forest | Ensemble | 0.9045 ± 0.021 | 85.05% | 0.9132 | 0.1183 | 94.16% | 0.905 |
| LightGBM | Boosting | 0.9063 ± 0.021 | 85.12% | 0.9136 | 0.1161 | 94.22% | 0.904 |
| KNN (100k-row subsample) | Instance-based | 0.8978 ± 0.022 | 83.81% | 0.9057 | 0.1877 | 93.14% | 0.901 |
| MLP (64-32) | Neural network | 0.9034 ± 0.022 | 85.04% | 0.9132 | 0.1190 | 94.07% | 0.905 |
| **LightGBM tuned (final)** | Boosting | **0.9070 ± 0.022** | **85.72%** | **0.9173** | **0.1140** | 93.86% | **0.906** |

¹ Mean over the 10 trips (the other test columns pool all points).

- **Selection** was made on CV macro-F1, which never uses student-test data. The top three models
  (LightGBM, Random Forest, MLP) are within 0.003 of each other, far less than the ±0.02 fold-to-fold spread.
- **After HMM decoding**, the tuned model is 0.3 pts behind untuned LightGBM / Random Forest on these 10 trips
  (about 50 of ~19k points). The choice was not changed based on test results.
- **Inference times** in `experiments.csv` were logged during each training run under different machine load.
  Back-to-back, the tuned model (600 trees) is about 2× slower than untuned (160 trees): 0.69 s vs 0.36 s
  (`results/bayes_opt_comparison.csv`).

### Bayesian optimisation (LightGBM)

Optuna TPE, 30 trials (+1 interrupted, marked FAIL), MedianPruner, objective = 5-fold CV macro-F1 on the
train trips. Record: `results/bayes_opt_trials.csv`, `results/optuna_lightgbm.db`, `results/bayes_opt_log.txt`.

| | CV macro-F1 | Test Top-1 | Test macro-F1 | Log loss | Inference |
|---|---|---|---|---|---|
| Untuned (160 trees, lr 0.07, 31 leaves) | 0.9063 | 85.12% | 0.9136 | 0.1161 | 0.36 s |
| Tuned (600 trees, lr 0.038, 82 leaves) | 0.9070 | 85.72% | 0.9173 | 0.1140 | 0.69 s |

### HMM / Viterbi

| Final model | Top-1 | Route continuity | Route F1 |
|---|---|---|---|
| Classifier only | 88.05% | 0.782 | 0.915 |
| + HMM (default β=5, unreachable=−30) | 93.78% | 0.906 | 0.975 |
| + HMM (tuned β=1.5, unreachable=−120) | 93.86% | 0.906 | 0.976 |

HMM parameters were tuned on the **train** trips with out-of-fold classifier scores
(`python scripts/tune_hmm.py final_model`). Grid, log and heatmap: `results/hmm_tuning_*`,
`results/plots/hmm_tuning_heatmap.png`. The earlier tuning with logistic regression is kept as `*_logreg`.

> The per-model sections below were written during earlier experiments, with earlier HMM settings.
> The tables above are the current, comparable numbers.

## Reproduce the Random Forest experiment

After building the feature tables, run these from the project root in the activated environment:

```powershell
python scripts/train_models.py random_forest
python scripts/evaluate_hmm.py random_forest
python scripts/plot_confusion.py random_forest
```

The forest uses 48 trees, Gini impurity, a maximum depth of 14, and at least five samples per leaf. Each tree trains on a bootstrap sample of 25% of the training rows and considers a square-root subset of features at each split. These initial settings limit fitting time on the million-row candidate table. The random seed is 42. The model uses the same candidate features and five trip-grouped folds as logistic regression.

Training saves `models/random_forest.joblib` and appends the measured results to `results/experiments.csv`. Decoder evaluation saves per-trip scores to `results/random_forest_hmm_vs_classifier.csv`. The confusion matrix in `results/plots/random_forest_confusion.png` counts candidate rows on labelled student-test observations, with one predicted positive per observation.

| Model | CV macro F1 | Student-test macro F1 | Pooled Top-1 | Log loss |
|---|---:|---:|---:|---:|
| logreg | 0.8967 | 0.9013 | 83.08% | 0.1698 |
| random_forest | 0.9045 | 0.9132 | 85.05% | 0.1183 |
| lightgbm | 0.9063 | 0.9136 | 85.12% | 0.1161 |

On the ten student-test trips, giving each trip equal weight, Random Forest Top-1 accuracy was 87.45% before decoding and 93.74% after HMM decoding. The table above pools labelled observations across trips.

To try this model in the API, run the following in PowerShell on Laptop 1, with the environment activated. The API uses the student-test folder by default.

```powershell
$env:PATHSNAP_MODEL = 'random_forest.joblib'
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Reproduce the LightGBM experiment

LightGBM sequentially fits trees to reduce binary log loss on candidate roads. The initial configuration uses 160 trees, a learning rate of 0.07, up to 31 leaves per tree, and a minimum of 80 training rows per leaf. The classifier uses the same features and trip-grouped folds as the other models. The split membership and scores for each fold are in `results/lightgbm_cv_folds.csv`.

Run from the project root in the pinned environment:

```powershell
python scripts/train_models.py lightgbm
python scripts/evaluate_hmm.py lightgbm
python scripts/plot_confusion.py lightgbm
$env:PATHSNAP_MODEL = 'lightgbm.joblib'
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

With the API running on the student-test folder, use a second terminal to run `python scripts/verify_student_api.py lightgbm`. It checks the three application endpoints and compares every returned metric with the saved per-trip evaluation.

For this run the decoder used the shared parameters in `models/hmm_params.json`, previously tuned with logistic regression. Across the ten student-test trips, giving each trip equal weight, LightGBM Top-1 accuracy was 87.51% with the classifier and 93.75% after decoding. Per-trip metrics are in `results/lightgbm_hmm_vs_classifier.csv`, and the candidate confusion matrix is in `results/plots/lightgbm_confusion.png`.

## Team task split

| Member | Owns | Deliverables |
|---|---|---|
| A | EDA & data quality | `notebooks/01_eda.ipynb`: inventory, gaps, network connectivity, anomalies (e.g. trip 00000054 GPS is ~130 m off its route), Folium maps |
| B | Models 2–3 | Random Forest, LightGBM in `scripts/train_models.py`; confusion matrices |
| C | Models 4–5 + Bayesian optimisation | KNN, MLP (with loss curves); `scripts/bayes_opt.py` (Optuna, GroupKFold on train), trial log CSV, tuned vs untuned |
| D | HMM tuning + API/UI + deployment | tune `beta` / emission weight on train + student_test; two-laptop demo; screenshots |
| E | Report | integrate results/plots into the PDF; everyone reviews their sections |

Rules to remember: never train or tune on evaluator data, keep all experiment logs, and every member must be able to explain every part of the work.

## Git workflow

```bash
git checkout -b <yourname>/<task>     # e.g. dikshith/lightgbm
git add <files> && git commit -m "Add LightGBM model"
git push -u origin <yourname>/<task>  # then open a Pull Request on GitHub
```
Before you commit, set `git config user.email` to the email on **your own** GitHub account. Otherwise your commits won't count as yours.
