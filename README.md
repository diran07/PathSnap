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

```bash
python scripts/build_features.py        # -> data/processed/*.parquet
python scripts/train_models.py          # CV + student_test, appends results/experiments.csv
python scripts/evaluate_hmm.py logreg   # classifier-only vs HMM -> results/hmm_vs_classifier.csv
python scripts/make_map.py 00000077     # -> results/maps/00000077.html
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

## Current baseline results (student_test, 10 trips)

| Method | Top-1 acc | Route continuity | Route F1 |
|---|---|---|---|
| Logistic Regression only | 0.854 | 0.742 | 0.898 |
| Logistic Regression + HMM (default β=5, unreachable=−30) | 0.910 | 0.905 | 0.971 |
| Logistic Regression + HMM (tuned β=1, unreachable=−200) | 0.927 | 0.904 | 0.976 |

HMM parameters were tuned on the **train** trips using out-of-fold classifier scores
(`python scripts/tune_hmm.py logreg`). The search grid, log and heatmap are in
`results/hmm_tuning_*` and `results/plots/hmm_tuning_heatmap.png`. The tuned values are saved to
`models/hmm_params.json`, which the pipeline and API load automatically. Re-run the tuning
whenever the final classifier changes.

(Per-trip numbers are in `results/hmm_vs_classifier.csv` and `results/hmm_tuning_summary.csv`.)

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

On the ten student-test trips, giving each trip equal weight, Random Forest Top-1 accuracy was 87.45% before decoding and 93.74% after HMM decoding. The table above pools labelled observations across trips.

To try this model in the API, run the following in PowerShell on Laptop 1, with the environment activated. The API uses the student-test folder by default.

```powershell
$env:PATHSNAP_MODEL = 'random_forest.joblib'
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

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
