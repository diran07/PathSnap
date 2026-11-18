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

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

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
2. `pip install -r requirements-lock.txt` (exact versions the model was trained with)
3. `run_api.bat C:\path\to\evaluator_trips`. It prints this laptop's IP, and every request is logged in this window.

**Laptop 2 (UI):**
1. `pip install -r requirements.txt`
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
| Logistic Regression + HMM | 0.910 | 0.905 | 0.971 |

(Per-trip numbers are in `results/hmm_vs_classifier.csv`.)

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
