# PathSnap: Map-Matching and Route Reconstruction
**UE24CS352A Machine Learning · Team 15** · Repository: https://github.com/diran07/PathSnap

> DRAFT. Each section owner must rewrite their part in their own words before submission.
> All numbers below come from files in `results/` and are on the **student-test** partition unless stated.
> Evaluator-test results are not used anywhere in this report.

---

## 1. Problem and approach
Noisy GPS points must be matched to the road segments (arcs) the vehicle actually used.
Pipeline: GPS → metric projection → candidate arcs → ML classifier scores each candidate →
HMM/Viterbi selects a connected route → FastAPI backend + Streamlit UI (two laptops).

---

## 2. Dataset understanding and EDA
*Partitions used: train (80 trips) and student-test (10 trips). Evaluator-test was not explored.*

**Data.** Each trip has `.track` (lon, lat, t at ~1 Hz), `.nodes` (lon, lat), `.arcs` (directed node pairs) and
`.route` (reference arc sequence). Train: 208,417 GPS points; student-test: 19,160.

**Preprocessing.**
- Dropped invalid coordinates and duplicate timestamps, sorted by time (none were found in practice).
- Projected lon/lat to metres around each trip's centre (equirectangular; < 1 % error at trip scale).
- Candidates: arcs within **50 m** of each point, nearest **8** kept (nearest 3 if none within 50 m).
  The reference arc is missing from the candidates for **0.8 %** of student-test points (2.1 % train).
- Labels: `.route` has no per-point label, so each point was aligned to the route in time order
  (monotonic dynamic programming); points > 100 m from the route are left unlabelled.

**Missing values and anomalies** (`results/eda/`)
- No missing values, invalid coordinates or out-of-order timestamps in any trip.
- 98 % (student-test) / 94 % (train) of steps are exactly 1 s; **22 train trips** have gaps > 60 s (max ≈ 42 h).
- Median GPS-to-route distance ≈ **4.8 m**, which justifies the 50 m search radius.
- Trips 54 and 33 lie ~130 m from their reference routes (GPS path 169 km vs route 31 km for trip 54),
  so their reference route likely covers only part of the trip.
- Speed spikes > 200 km/h (GPS glitches); 14 trips have fragmented road networks; 95 % of arcs are two-way.

**Features (14 per candidate) and why**
| Feature group | Why |
|---|---|
| Distance, distance rank, gap to nearest, ratio to nearest | Closer arcs are more likely |
| Heading difference to arc direction (and its cosine) | Separates a road from its opposite direction |
| Speed, time gap | Heading is unreliable when stationary; large gaps add uncertainty |
| Arc length, position along arc | Context of the projection |
| Mean / max distance of the arc to the 2 neighbouring points on each side | Temporal consistency |

**Figures:** `eda_trip_sizes.png`, `eda_sampling_speed.png`, `eda_gps_to_route.png`, `eda_network.png`,
`eda_trip_locations.png`, Folium map `results/maps/eda_00000001.html` *(insert screenshot)*.

---

## 3. Model experimentation and comparison
Setup for every model: same 14 features, **5-fold GroupKFold on the 80 train trips** (no trip in two folds),
evaluation on the 10 student-test trips, seed 42. Results: `results/model_comparison.csv`.

### 3.1 Overall comparison
| Model | Family | Key configuration | CV macro-F1 | Test Top-1 | Route continuity (+HMM) | Test macro-F1 | Log loss | Inference |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | Linear | C=1, standardised | 0.8967 ± 0.021 | 83.08 % | 0.904 | 0.9013 | 0.1698 | 0.11 s |
| Random Forest | Ensemble | 48 trees, depth 14, leaf ≥ 5 | 0.9044 ± 0.021 | 85.03 % | 0.905 | 0.9131 | 0.1188 | 0.25 s |
| LightGBM | Boosting | 160 trees, lr 0.07, 31 leaves | 0.9063 ± 0.021 | 85.12 % | 0.904 | 0.9136 | 0.1161 | 1.24 s* |
| KNN | Instance-based | k=25, distance-weighted, 100k-row subsample | 0.8978 ± 0.022 | 83.81 % | 0.901 | 0.9057 | 0.1877 | 25.6 s |
| MLP | Neural network | 64-32 ReLU, Adam, early stopping | 0.9034 ± 0.022 | 85.04 % | 0.905 | 0.9132 | 0.1190 | 0.16 s |
| **LightGBM tuned (final)** | Boosting | 600 trees, lr 0.038, 82 leaves | **0.9070 ± 0.022** | **85.72 %** | **0.906** | **0.9173** | **0.1140** | 0.69 s |

Test Top-1 and macro-F1 pool all labelled points. Macro-F1 is computed on candidates, with the top-scored
candidate of each point as the positive prediction. *Inference times were logged under different machine
load; back-to-back, untuned LightGBM takes 0.36 s.

**Primary metric: CV macro-F1.** It does not use student-test data and treats correct and incorrect candidates
equally despite the ~1:5 imbalance (≈ 18.9k correct vs 94.8k other candidates on student-test). **Final model: LightGBM** (best CV macro-F1, best log loss, which matters
because the HMM uses the probabilities, and 2× faster to train than Random Forest).
The top three models are within 0.003 CV macro-F1, much less than the ±0.02 spread between folds.

### 3.2 Per-model answers
*(Owners: expand each in your own words, linking to the numbers.)*

**Logistic Regression.** *Why:* simple, fast and gives calibrated probabilities, so it is a good baseline.
*Strengths/limits:* fastest (1.8 s train), but linear, so it cannot model feature interactions (e.g. a heading
difference matters only when moving). *Hyperparameters:* C=1.0, L2, standardised inputs. *Loss:* L2-regularised
log loss. *Results:* Top-1 83.08 %, macro-F1 0.9013, log loss 0.1698 (weakest). *Fit:* CV 0.8967 vs test 0.9013,
no overfitting; the lowest scores suggest **underfitting**.

**Random Forest.** *Why:* nonlinear, robust to feature scale. *Strengths/limits:* strong accuracy; model file is
large (21 MB). *Hyperparameters:* 48 trees, max depth 14, min leaf 5, 25 % bootstrap rows, √features. *Loss:* Gini
impurity. *Results:* Top-1 85.03 %, macro-F1 0.9131, log loss 0.1188. *Fit:* fold F1 0.875–0.932, test above CV
mean, depth and leaf limits controlled overfitting.

**LightGBM.** *Why:* gradient boosting handles interactions and imbalance well, and is fast. *Strengths/limits:*
best untuned scores; more hyperparameters to tune. *Hyperparameters:* 160 trees, lr 0.07, 31 leaves, min 80 rows
per leaf. *Loss:* binary log loss. *Results:* Top-1 85.12 %, macro-F1 0.9136, log loss 0.1161. *Fit:* fold F1
0.877–0.935, test ≥ CV mean, so no overfitting.

**KNN.** *Why:* tests whether similar candidates share labels. *Strengths/limits:* no training, but prediction
compares against stored rows: **25.6 s** inference, so it was trained on a seeded 100k-row subsample
(100k and 200k gave the same fold-1 F1, 0.866). *Hyperparameters:* k=25, distance weights, standardised. *Loss:* none
(distance-weighted vote). *Results:* Top-1 83.81 %, worst log loss 0.1877 (probabilities from neighbour votes are
poorly calibrated). *Fit:* CV 0.8978 vs test 0.9057, no overfitting.

**MLP.** *Why:* learns nonlinear feature combinations. *Strengths/limits:* good accuracy, fast inference; slowest
training (42 s). *Hyperparameters:* layers (64, 32), ReLU, Adam lr 1e-3, batch 1024, L2 α 1e-4, early stopping
(10 % validation), max 40 epochs. *Loss:* binary cross-entropy. *Results:* Top-1 85.04 %, macro-F1 0.9132,
log loss 0.1190. *Fit:* on held-out train trips (`mlp_loss.png`), validation loss is lowest at **epoch 26
(0.1544)** and then flattens or rises slightly while training loss keeps falling (0.1265 → 0.1245), so **mild
overfitting begins after epoch ~26**. The constant train/validation gap (~0.03) reflects unseen trips being harder.

### 3.3 Visualisations
`model_comparison.png` (CV macro-F1 with ±1 std, test Top-1 and macro-F1, log loss, inference time),
`model_comparison_hmm.png`, confusion matrices `*_confusion.png` (all 6 models), `mlp_loss.png`
(the only model with epoch-wise history).

---

## 4. Bayesian hyperparameter optimisation (LightGBM)
1. **Hyperparameters:** n_estimators, learning_rate, num_leaves, min_child_samples, subsample, colsample_bytree, reg_lambda.
2. **Ranges:** trees 100–800 (step 50); lr 0.01–0.2 (log); leaves 15–255 (log); min rows 20–1000 (log);
   subsample 0.5–1; colsample 0.5–1; L2 0.001–10 (log).
3. **Objective:** maximise mean 5-fold GroupKFold macro-F1 on the train trips (student-test not used).
4. **Trials:** 30 (19 completed, 11 pruned early by MedianPruner) plus 1 interrupted (marked FAIL); TPE sampler,
   seed 42; ~21 min of trial time. Record: `results/bayes_opt_trials.csv`, `optuna_lightgbm.db`.
5. **Best (trial 28):** 600 trees, lr 0.0384, 82 leaves, min 42 rows, subsample 0.80, colsample 0.83, L2 0.0105.
6. **Tuned vs untuned:**

| | CV macro-F1 | Test Top-1 | Test macro-F1 | Log loss | Inference (back-to-back) |
|---|---|---|---|---|---|
| Untuned | 0.9063 | 85.12 % | 0.9136 | 0.1161 | 0.36 s |
| Tuned | **0.9070** | **85.72 %** | **0.9173** | **0.1140** | 0.69 s |

The tuned model is better on every quality metric, but the CV gain is small (within fold spread), so the
initial settings were already near-optimal. Cost: ~2× slower inference (more trees). Most important
hyperparameters: colsample_bytree, n_estimators, subsample (`bayes_opt_importance.png`, `bayes_opt_history.png`).

---

## 5. HMM and Viterbi route reconstruction (`src/pathsnap/hmm.py`)
1. **Hidden states:** the candidate arcs of each GPS point.
2. **Emissions:** log of the classifier's probability, normalised over the point's candidates.
3. **Transitions:** `log p = −|driving distance − straight-line distance| / (β · Δt)`, where driving distance is
   the shortest path in the road network (Dijkstra on the sub-network near the trip).
4. **Missing candidates / GPS gaps:** if no arc is within 50 m, the 3 nearest are used. β is multiplied by the
   time gap Δt, so long gaps are penalised less.
5. **Disconnected segments:** unreachable transitions get a fixed log-probability (tuned: −120) instead of
   −∞, so decoding never fails and can "break" the route when the data demands it.
6. **Did it help?** Parameters tuned on train trips using out-of-fold scores (β = 1.5, unreachable = −120):

| Final model | Top-1 | Route continuity | Route F1 |
|---|---|---|---|
| Classifier only | 88.05 % | 0.782 | 0.915 |
| + HMM (default β=5, −30) | 93.78 % | 0.906 | 0.975 |
| + HMM (tuned) | **93.86 %** | **0.906** | **0.976** |

The HMM adds **+5.8 pts Top-1** and raises continuity from 0.78 to 0.91; it improves continuity for every
model (`model_comparison_hmm.png`). **Where it did not help:** with logistic regression and default parameters,
trip 07 fell from 90.1 % to 85.1 %. The vehicle was manoeuvring slowly ~30 m beside a two-way road, and the HMM
locked onto the opposite-direction arc. Tuning fixed this (99.7 %). A rule penalising backward movement along an
arc was tested and gave no gain on train, so it was not used. After decoding, the tuned model is 0.4–0.5 pts
behind Random Forest / untuned LightGBM on the 10 test trips; this is within noise, and the model choice was
not changed based on test results.

---

## 6. Geographical visualisation (Folium)
`results/maps/00000077.html` (`scripts/make_map.py`): GPS observations (blue), nearby road network (grey,
toggle), reference route (green), classifier-only route (orange), HMM route (red), start and end markers,
layer control, fitted to the trip. *(Insert screenshot.)* EDA maps: `results/maps/eda_*.html`.

---

## 7. API, UI and deployment
- **Laptop 1:** FastAPI (`api/main.py`), with endpoints `/health`, `/trips`, `/trips/summary`, `/predict`. Loads
  `final_model.joblib` and the HMM parameters. The reference route is read only on the server **after**
  prediction to compute metrics, and is never sent to the UI. Every request is logged.
- **Laptop 2:** Streamlit UI (`ui/app.py`) showing trip cards, route map, metrics and a classifier-vs-HMM comparison.
- Launch: `run_api.bat <eval folder>`, `run_ui.bat <laptop-1 IP>`; firewall: `setup_firewall.bat`.

*(Insert: UI screenshot, and Laptop-1 CMD screenshot showing `POST /predict from <laptop-2 IP> -> 200`.)*

---

## 8. Academic integrity and AI use
*(Adapt to what the team actually did.)* We used Claude (Anthropic) as a coding assistant for parts of the
implementation and for drafting this report's structure. All experiments were run by us, all numbers are
produced by the code in the repository and can be reproduced with the commands in the README, and every
team member can explain the code, results and design choices. No evaluator-test data was used for
training, tuning or model selection.

## Team contributions
| Member | Contribution |
|---|---|
| diran07 | *(fill in)* |
| DIkshith1311 | *(fill in)* |
| Dishanth2 | *(fill in)* |
| theDEMIGOD600 | *(fill in)* |
