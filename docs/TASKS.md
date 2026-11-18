# Team Tasks

Each person works on **their own branch**, commits **from their own laptop**, and merges into `main` with a Pull Request.
Every person must be able to explain their own part, and the overall pipeline, in the viva.

## One-time setup (everyone)

```bash
git clone https://github.com/diran07/PathSnap.git
cd PathSnap
git config user.name  "Your Name"
git config user.email "your-github-email@gmail.com"     # must match your GitHub account
pip install -r requirements.txt
```
Unzip `Student folder.zip` so you have `data/raw/train/...` and `data/raw/student_test/...`, then:
```bash
python scripts/build_features.py      # builds data/processed/*.parquet (~2 min)
```

## Daily workflow

```bash
git checkout main && git pull
git checkout -b <name>/<task>          # e.g. dikshith/rf-lightgbm
# ...work...
git add <your files>
git commit -m "Short description of what you did"
git push -u origin <name>/<task>
```
Then open a Pull Request on GitHub → someone reviews → **Merge**.
Commit small and often, e.g. "Add random forest", then "Add confusion matrix". Don't put everything in one giant commit at the end.

**Never commit:** `data/raw`, `data/processed`, passwords or keys. **Never** train or tune on evaluator data.

---

## diran07: EDA, HMM tuning, report lead

Branch: `diran/eda`, then `diran/hmm-tuning`

- [ ] `scripts/eda.py`: dataset inventory, missing/invalid values, time gaps, trip extents, network size and connectivity, GPS-to-route offset, anomalies; plots in `results/plots/`, Folium maps in `results/maps/`
- [ ] Write the EDA section of the report (say which partition each analysis uses)
- [ ] `scripts/tune_hmm.py`: grid or Optuna search over `beta` and the emission weight using **train** trips only, then report on student_test
- [ ] Analyse why HMM made trip 00000007 worse (look at its map)
- [ ] Assemble the final PDF report from everyone's sections

## DIkshith1311: Models 2 and 3 (tree ensembles and boosting)

Branch: `dikshith/rf-lightgbm`

- [ ] Add `random_forest` (`RandomForestClassifier`) to `MODELS` in `scripts/train_models.py`
- [ ] Add `lightgbm` (`LGBMClassifier`) to `MODELS`
- [ ] Run `python scripts/train_models.py random_forest lightgbm` (results append to `results/experiments.csv`)
- [ ] `scripts/plot_confusion.py`: confusion matrix per model on student_test → `results/plots/confusion_<model>.png`
- [ ] Feature importance plot for both models
- [ ] Report: one subsection per model answering the compulsory questions in Deliverables.pdf section 3.2

## Dishanth2: Models 4 and 5 (KNN and MLP) and comparison plots

Branch: `dishanth/knn-mlp`

- [ ] Add `knn` (`StandardScaler` + `KNeighborsClassifier`) to `MODELS`. KNN is slow on 1.1M rows, so explain any subsampling you do
- [ ] Add `mlp` (`StandardScaler` + `MLPClassifier` with `early_stopping=True`)
- [ ] MLP training/validation loss plot from `loss_curve_` / `validation_scores_` → `results/plots/mlp_loss.png`
- [ ] `scripts/plot_comparison.py`: reads `results/experiments.csv` and plots CV macro-F1 (with std error bars), student-test Top-1 and macro-F1, log loss, and inference time for **all** models
- [ ] Report: subsections for KNN and MLP, plus the overall comparison table (Deliverables.pdf section 3.1)

## theDEMIGOD600: Bayesian optimisation, final model, deployment

Branch: `demigod/bayes-opt`, then `demigod/deployment`

- [ ] Once the comparison shows the best model (likely LightGBM), write `scripts/bayes_opt.py` using Optuna (TPE sampler):
  - search space justified in comments
  - objective = mean macro-F1 from `GroupKFold(5)` on **train** trips
  - at least 30 trials; save every trial to `results/bayes_opt_trials.csv`
  - refit the best params on all train trips → save `models/final_model.joblib`
  - tuned vs untuned table: CV macro-F1, student-test Top-1, macro-F1, log loss, inference time
- [ ] Optuna plots: optimisation history and parameter importance → `results/plots/`
- [ ] Run `python scripts/evaluate_hmm.py final_model` with the final model
- [ ] Deployment rehearsal using `docs/DEMO_CHECKLIST.md`, with two laptops on a phone hotspot
- [ ] Screenshots: the UI showing results, and the API terminal logs → `report/screenshots/`
- [ ] Report: Bayesian optimisation section (Deliverables.pdf section 4) and the API/UI section (section 7)

---

## Order / dependencies

```
now  ─┬─ diran: EDA ───────────────────────────────┐
      ├─ dikshith: RF + LightGBM ──┐               │
      └─ dishanth: KNN + MLP ──────┴─> comparison ─┴─> demigod: Bayes opt ─> final model ─> HMM re-eval ─> report
demigod meanwhile: deployment rehearsal with the current logreg model
```

## Before submission

- [ ] All branches merged into `main`
- [ ] README results table updated with final numbers
- [ ] Repo made **public**
- [ ] Report states the AI assistance used, as required by the rules
