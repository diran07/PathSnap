"""Build the technical report PDF: report/PathSnap_Technical_Report.pdf

All tables are read from results/ at build time, so the PDF always matches the repository.
Screenshots: put these in report/screenshots/ and re-run to embed them (otherwise a
placeholder box is drawn):
    api_logs.png   - Laptop 1 API window showing POST /predict requests from Laptop 2
    ui_laptop2.png - (optional) the UI on Laptop 2; otherwise the test-run screenshot is used

Usage: python scripts/build_report.py
"""
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

RES, PLOTS, FIG, SHOTS = ROOT / "results", ROOT / "results" / "plots", ROOT / "report" / "figures", ROOT / "report" / "screenshots"
OUT = ROOT / "report" / "PathSnap_Technical_Report.pdf"

TEAM = [("Dhruvanth", "PES1UG24AM362"), ("Dikshith Varma M N", "PES1UG24AM363"),
        ("Diran Somanna", "PES1UG24AM364"), ("Dishanth Gowda", "PES1UG24AM365")]

# ---------------------------------------------------------------- fonts / styles
FONT, BOLD = "Helvetica", "Helvetica-Bold"
for reg, bold in [("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
                  ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")]:
    if Path(reg).exists() and Path(bold).exists():
        pdfmetrics.registerFont(TTFont("Body", reg))
        pdfmetrics.registerFont(TTFont("Body-Bold", bold))
        pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body", boldItalic="Body-Bold")
        FONT, BOLD = "Body", "Body-Bold"
        break

ss = getSampleStyleSheet()
INK, MUTED, ACCENT, LINE = colors.HexColor("#111111"), colors.HexColor("#555555"), colors.HexColor("#1f1f1f"), colors.HexColor("#d9d9d9")
P = ParagraphStyle("P", parent=ss["Normal"], fontName=FONT, fontSize=9.6, leading=13.4, textColor=INK, spaceAfter=4)
SMALL = ParagraphStyle("S", parent=P, fontSize=8.2, leading=10.6, textColor=MUTED)
CAP = ParagraphStyle("C", parent=SMALL, alignment=TA_CENTER, spaceBefore=2, spaceAfter=10)
H1 = ParagraphStyle("H1", parent=P, fontName=BOLD, fontSize=14.5, leading=18, spaceBefore=10, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=P, fontName=BOLD, fontSize=11, leading=14, spaceBefore=8, spaceAfter=3)
BUL = ParagraphStyle("B", parent=P, leftIndent=12, bulletIndent=2, spaceAfter=2)
CELL = ParagraphStyle("cell", parent=P, fontSize=8.1, leading=10.2, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName=BOLD)

W = A4[0] - 3.6 * cm   # usable width


def para(t, s=P):
    return Paragraph(t, s)


def bullets(items):
    return [Paragraph(i, BUL, bulletText="•") for i in items]


def table(rows, widths=None, header=True, bold_last=False):
    data = [[c if not isinstance(c, str) else Paragraph(c, CELLB if (header and r == 0) else CELL)
             for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#efefef")))
    if bold_last:
        st.append(("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#f3fbe4")))
    t.setStyle(TableStyle(st))
    return t


def figure(path, caption, width=W, max_h=9.5 * cm):
    path = Path(path)
    if not path.exists():
        return placeholder(f"Missing figure: {path.name}", caption)
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(str(path)).getSize()
    ratio = ih / iw
    w = min(width - 6, max_h / ratio)
    img = Image(str(path), width=w, height=w * ratio)
    box = Table([[img], [para(caption, CAP)]], colWidths=[width])
    box.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                             ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 1),
                             ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
    return box


def placeholder(text, caption):
    box = Table([[para(f"<b>{text}</b>", ParagraphStyle("ph", parent=P, alignment=TA_CENTER, textColor=MUTED))]],
                colWidths=[W], rowHeights=[4 * cm])
    box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#b0b0b0")),
                             ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f6f6f6")),
                             ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    out = Table([[box], [para(caption, CAP)]], colWidths=[W])
    out.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    return out


def two(fig_a, fig_b):
    t = Table([[fig_a, fig_b]], colWidths=[W / 2, W / 2])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 2),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))
    return t


def f4(x):
    return f"{x:.4f}"


def pc(x):
    return f"{x * 100:.2f}%"


# ---------------------------------------------------------------- data from results/
comp = pd.read_csv(RES / "model_comparison.csv")
bo = pd.read_csv(RES / "bayes_opt_comparison.csv").set_index("model")
trials = pd.read_csv(RES / "bayes_opt_trials.csv")
hmm = pd.read_csv(RES / "hmm_tuning_summary.csv").groupby("method")[["top1_acc", "route_continuity", "route_f1"]].mean()
hp = json.loads((ROOT / "models" / "hmm_params.json").read_text())
cand = pd.read_csv(RES / "student_test_candidate_stats.csv")
cand_tr = pd.read_csv(RES / "train_candidate_stats.csv")
inv = pd.read_csv(RES / "eda" / "inventory.csv")
mlp = pd.read_csv(RES / "mlp_loss_history.csv")
best_ep = mlp.loc[mlp.val_log_loss.idxmin()]
best_trial = trials.loc[trials.value.idxmax()]


def row(name):
    return comp[comp.model == name].iloc[0]


def folds(m):
    f = RES / f"{m}_cv_folds.csv"
    if not f.exists():
        return ""
    v = pd.read_csv(f).macro_f1
    return f"fold macro-F1 {v.min():.3f}–{v.max():.3f}"


tr_i, te_i = inv[inv.partition == "train"], inv[inv.partition == "student_test"]

# ---------------------------------------------------------------- document
story = []

# title page
story += [Spacer(1, 3.2 * cm),
          para("PathSnap", ParagraphStyle("T", parent=H1, fontSize=30, leading=36, alignment=TA_CENTER)),
          para("Distributed Map-Matching &amp; Route Reconstruction",
               ParagraphStyle("T2", parent=H1, fontSize=15, alignment=TA_CENTER, textColor=MUTED)),
          Spacer(1, 0.4 * cm),
          para("Technical Report · UE24CS352A Machine Learning · PES University (RR Campus)",
               ParagraphStyle("T3", parent=P, alignment=TA_CENTER, textColor=MUTED)),
          Spacer(1, 1.6 * cm)]
tt = Table([[para("<b>Name</b>", CELLB), para("<b>SRN</b>", CELLB)]] +
           [[para(n, ParagraphStyle("n", parent=P, fontSize=11)), para(s, ParagraphStyle("n", parent=P, fontSize=11))]
            for n, s in TEAM], colWidths=[7.5 * cm, 5 * cm])
tt.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, LINE), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#efefef")),
                        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                        ("LEFTPADDING", (0, 0), (-1, -1), 8)]))
story += [tt, Spacer(1, 1.6 * cm),
          para("Repository: <b>https://github.com/diran07/PathSnap</b>", ParagraphStyle("r", parent=P, alignment=TA_CENTER)),
          para(f"{date.today():%d %B %Y}", ParagraphStyle("d", parent=P, alignment=TA_CENTER, textColor=MUTED)),
          Spacer(1, 2 * cm),
          para("All results in this report are on the <b>student-test</b> partition (10 trips) unless stated, and are "
               "produced by the code in the repository (tables are generated from <i>results/</i> by "
               "<i>scripts/build_report.py</i>). The evaluator-test partition was not used for training, tuning, "
               "model selection or any result in this report.", ParagraphStyle("note", parent=SMALL, alignment=TA_CENTER)),
          PageBreak()]

# 1 overview
fin = row("LightGBM (tuned)")
story += [para("1. Problem and approach", H1),
          para("GPS observations are noisy, so a vehicle can appear on a parallel street or jump between roads. "
               "PathSnap assigns each GPS observation to the directed road segment (arc) the vehicle was on and "
               "reconstructs a connected route."),
          para("<b>Pipeline:</b> GPS points → local metric projection → candidate arcs near each point → "
               "ML classifier scores each candidate → HMM/Viterbi decoding over the road network → "
               "FastAPI backend (Laptop 1) and Streamlit UI (Laptop 2)."),
          table([["Headline result (final system, student-test)", "Value"],
                 ["Observation-level Top-1, classifier only → + HMM", f"{pc(hmm.loc['classifier_only', 'top1_acc'])} → <b>{pc(hmm.loc['hmm_tuned', 'top1_acc'])}</b>"],
                 ["Route continuity, classifier only → + HMM", f"{hmm.loc['classifier_only', 'route_continuity']:.3f} → <b>{hmm.loc['hmm_tuned', 'route_continuity']:.3f}</b>"],
                 ["Route F1 (length-weighted), + HMM", f"<b>{hmm.loc['hmm_tuned', 'route_f1']:.3f}</b>"],
                 ["Final classifier", "LightGBM, Bayesian-optimised (Optuna TPE, 30 trials)"]],
                widths=[W * 0.6, W * 0.4])]

# 2 EDA
story += [para("2. Dataset understanding and exploratory analysis", H1),
          para("<i>Partitions used: train (80 trips) and student-test (10 trips). Evaluator-test was not explored.</i>", SMALL),
          para("2.1 Data and preprocessing", H2),
          para(f"Each trip has <b>.track</b> (longitude, latitude, time; nominally 1 Hz), <b>.nodes</b> (longitude, latitude), "
               f"<b>.arcs</b> (directed from-node, to-node) and <b>.route</b> (reference arc sequence). Train has "
               f"{int(tr_i.n_points.sum()):,} GPS points, student-test {int(te_i.n_points.sum()):,}."),
          *bullets([
              "<b>Cleaning:</b> drop invalid coordinates and duplicate timestamps, sort by time (none were present).",
              "<b>Projection:</b> longitude/latitude are angles, so each trip is projected to metres around its centroid "
              "(equirectangular; under 1 % distance error at trip scale).",
              f"<b>Candidates:</b> arcs within 50 m of a point (nearest 8 kept; nearest 3 if none within 50 m). The reference arc "
              f"is missing from the candidate set for {cand.candidate_miss_rate.mean() * 100:.1f} % of student-test points "
              f"({cand_tr.candidate_miss_rate.mean() * 100:.1f} % train).",
              "<b>Per-point labels:</b> .route has no per-point label, so each point is aligned to the route in time order "
              "(monotonic dynamic programming). Points more than 100 m from the route are left unlabelled.",
          ]),
          para("2.2 Observed patterns and data quality", H2),
          *bullets([
              "No missing values, invalid coordinates, duplicate or out-of-order timestamps in any of the 90 trips.",
              f"{te_i.dt_1s_frac.mean() * 100:.0f} % (student-test) / {tr_i.dt_1s_frac.mean() * 100:.0f} % (train) of time steps are "
              f"exactly 1 s; {int((tr_i.gaps_over_60s > 0).sum())} train trips have gaps over 60 s (largest "
              f"{tr_i.dt_max_s.max() / 3600:.0f} h).",
              f"Median GPS-to-reference-route distance is about {inv.gps_to_route_median_m.median():.1f} m, which motivated the 50 m radius.",
              "Trips 54 and 33 lie far from their reference routes (median 138 m and 122 m). This is a limitation of the "
              "reference labels, so these points are mostly unlabelled.",
              "Point-to-point speeds above 200 km/h occur (GPS jumps); 14 trips have fragmented road networks; "
              "about 95 % of arcs are two-way.",
          ]),
          figure(PLOTS / "eda_sampling_speed.png", "Fig. 1 - Sampling interval and point-to-point speed (train + student-test)",
                 max_h=6 * cm),
          two(figure(PLOTS / "eda_gps_to_route.png", "Fig. 2 - GPS distance to reference route (train + student-test)", W / 2 - 4),
              figure(PLOTS / "eda_trip_locations.png", "Fig. 2b - Trip locations by partition", W / 2 - 4)),
          two(figure(PLOTS / "eda_trip_sizes.png", "Fig. 3 - Trip sizes by partition", W / 2 - 4),
              figure(PLOTS / "eda_network.png", "Fig. 4 - Road-network size and connectivity", W / 2 - 4)),
          figure(FIG / "folium_eda_trip00000054.jpg",
                 "Fig. 5 - Folium map, train trip 00000054: GPS observations and reference route "
                 "(trip flagged in EDA: median GPS-to-route distance 138 m)", max_h=8 * cm),
          para("2.3 Features and why", H2),
          table([["Feature group (14 features per candidate)", "Reason"],
                 ["Distance, distance rank, gap and ratio to the nearest candidate", "Closer arcs are more likely"],
                 ["Heading difference to arc direction (and its cosine)", "Separates a road from its opposite direction"],
                 ["Speed, time gap to previous point", "Heading is unreliable when stationary; gaps add uncertainty"],
                 ["Arc length, position of the projection along the arc", "Geometric context of the match"],
                 ["Mean / max distance of the arc to the 2 neighbouring points each side", "Temporal consistency"]],
                widths=[W * 0.58, W * 0.42]),
          PageBreak()]

# 3 models
order = ["Logistic Reg.", "Random Forest", "LightGBM", "KNN", "MLP", "LightGBM (tuned)"]
cfg = {"Logistic Reg.": "C=1, L2, standardised", "Random Forest": "48 trees, depth 14, leaf ≥5, 25 % rows",
       "LightGBM": "160 trees, lr 0.07, 31 leaves", "KNN": "k=25, distance-weighted, 100k-row subsample",
       "MLP": "64-32 ReLU, Adam, early stop", "LightGBM (tuned)": "600 trees, lr 0.038, 82 leaves"}
rows = [["Model", "Family", "Key configuration", "CV macro-F1", "Top-1", "Contin. (+HMM)", "Macro-F1",
         "Log loss", "Infer."]]
for m in order:
    r = row(m)
    rows.append([("<b>%s</b>" % m) if m == "LightGBM (tuned)" else m, r.family, cfg[m],
                 f"{r.cv_macro_f1:.4f} ± {r.cv_macro_f1_std:.3f}", pc(r.test_top1_acc),
                 f"{r.route_continuity_hmm:.3f}", f4(r.test_macro_f1), f4(r.test_log_loss), f"{r.test_inference_s:.2f} s"])
story += [para("3. Model experimentation and comparison", H1),
          para("Every model uses the same 14 features, the same <b>5-fold GroupKFold</b> on the 80 train trips "
               "(grouped by trip so no trip appears in both training and validation), random seed 42, and is evaluated on the "
               "10 student-test trips. Top-1 accuracy and macro-F1 pool all labelled observations; macro-F1 is computed on "
               "candidates, with the top-scored candidate of each observation as the positive prediction. "
               "Source: <i>results/model_comparison.csv</i>."),
          para("3.1 Overall model comparison", H2),
          table(rows, widths=[W * w for w in [0.13, 0.11, 0.21, 0.12, 0.08, 0.09, 0.09, 0.08, 0.09]], bold_last=True),
          para("Top-1, Macro-F1, Log loss: classifier only, student-test. Contin. = route continuity after HMM decoding. "
               "Infer. = time to score all student-test candidates, logged during each training run (machine load varied; "
               f"back-to-back, untuned LightGBM takes {bo.loc['lightgbm_untuned', 'test_inference_s']:.2f} s and the tuned model "
               f"{bo.loc['lightgbm_tuned', 'test_inference_s']:.2f} s).", SMALL),
          para("<b>Selection.</b> The primary metric is <b>CV macro-F1</b>: it never uses student-test data and weights correct "
               "and incorrect candidates equally despite the ~1:5 class imbalance. LightGBM was chosen: best CV macro-F1, best log "
               "loss (the HMM consumes these probabilities directly) and about twice as fast to train as Random Forest. The top three "
               "models are within 0.003 CV macro-F1, much less than the ±0.02 spread between folds, so the margin is small."),
          figure(PLOTS / "model_comparison.png", "Fig. 6 - CV macro-F1 (±1 std), student-test Top-1 / macro-F1, log loss, inference time",
                 max_h=10.5 * cm),
          figure(PLOTS / "model_comparison_hmm.png", "Fig. 7 - Classifier only vs classifier + HMM, all models (student-test)",
                 max_h=6.5 * cm),
          para("3.2 Per-model analysis", H2)]

lr, rf, lg, kn, ml = (row(m) for m in ["Logistic Reg.", "Random Forest", "LightGBM", "KNN", "MLP"])
models_txt = [
    ("Model 1 - Logistic Regression (linear)",
     "<b>Why:</b> simple, fast and gives calibrated probabilities, so it is a baseline for the HMM. "
     "<b>Strengths / limits:</b> fastest to train; linear, so it cannot model interactions such as heading mattering only when moving. "
     "<b>Hyperparameters:</b> C = 1.0 (L2), standardised inputs, max_iter 2000. <b>Objective:</b> L2-regularised log loss. "
     f"<b>Student-test:</b> Top-1 {pc(lr.test_top1_acc)}, macro-F1 {f4(lr.test_macro_f1)}, log loss {f4(lr.test_log_loss)} (weakest). "
     f"<b>Fit:</b> CV {f4(lr.cv_macro_f1)} vs test {f4(lr.test_macro_f1)}: no overfitting; the lowest scores of all models indicate underfitting."),
    ("Model 2 - Random Forest (ensemble)",
     "<b>Why:</b> nonlinear, robust to feature scale and outliers. <b>Strengths / limits:</b> strong accuracy; large model file (21 MB). "
     "<b>Hyperparameters:</b> 48 trees, max depth 14, min samples per leaf 5, 25 % bootstrap rows, √(features) per split. "
     f"<b>Objective:</b> Gini impurity. <b>Student-test:</b> Top-1 {pc(rf.test_top1_acc)}, macro-F1 {f4(rf.test_macro_f1)}, "
     f"log loss {f4(rf.test_log_loss)}. <b>Fit:</b> {folds('random_forest')}; test above the CV mean, depth and leaf limits keep it from overfitting."),
    ("Model 3 - LightGBM (boosting)",
     "<b>Why:</b> gradient boosting captures feature interactions and is fast on a million rows. "
     "<b>Strengths / limits:</b> best untuned scores; many hyperparameters (tuned in Section 4). "
     "<b>Hyperparameters:</b> 160 trees, learning rate 0.07, 31 leaves, min 80 rows per leaf. <b>Objective:</b> binary log loss. "
     f"<b>Student-test:</b> Top-1 {pc(lg.test_top1_acc)}, macro-F1 {f4(lg.test_macro_f1)}, log loss {f4(lg.test_log_loss)}. "
     f"<b>Fit:</b> {folds('lightgbm')}; test at or above the CV mean: no overfitting."),
    ("Model 4 - KNN (instance-based)",
     "<b>Why:</b> tests whether similar candidates share labels. <b>Strengths / limits:</b> no training, but prediction compares against "
     f"stored rows ({kn.test_inference_s:.1f} s inference), so it is fitted on a seeded 100k-row subsample (100k and 200k rows gave the same "
     "fold-1 macro-F1, 0.866). <b>Hyperparameters:</b> k = 25, distance weighting, standardised inputs. <b>Objective:</b> none "
     f"(distance-weighted vote). <b>Student-test:</b> Top-1 {pc(kn.test_top1_acc)}, macro-F1 {f4(kn.test_macro_f1)}, log loss "
     f"{f4(kn.test_log_loss)}, the worst calibration because probabilities come from neighbour votes. <b>Fit:</b> {folds('knn')}; no overfitting."),
    ("Model 5 - MLP (neural network)",
     "<b>Why:</b> learns nonlinear feature combinations. <b>Strengths / limits:</b> good accuracy and fast inference; slowest to train. "
     "<b>Hyperparameters:</b> hidden layers (64, 32), ReLU, Adam (lr 0.001), batch 1024, L2 α = 0.0001, early stopping on 10 % "
     f"validation, max 40 epochs. <b>Objective:</b> binary cross-entropy. <b>Student-test:</b> Top-1 {pc(ml.test_top1_acc)}, "
     f"macro-F1 {f4(ml.test_macro_f1)}, log loss {f4(ml.test_log_loss)}. <b>Fit:</b> on held-out train trips (Fig. 8) validation loss is "
     f"lowest at epoch {int(best_ep.epoch)} ({best_ep.val_log_loss:.4f}) and then flattens while training loss keeps falling "
     f"({best_ep.train_log_loss:.4f} → {mlp.train_log_loss.iloc[-1]:.4f}), so mild overfitting starts after about epoch {int(best_ep.epoch)}. "
     "The steady train/validation gap (~0.03) reflects unseen trips being harder."),
]
for title, txt in models_txt:
    story += [KeepTogether([para(title, ParagraphStyle("h3", parent=H2, fontSize=10)), para(txt)])]
story += [KeepTogether([para("3.3 Model visualisations", H2),
          two(figure(PLOTS / "mlp_loss.png", "Fig. 8 - MLP training vs validation loss (only model with epoch-wise history)", W / 2 - 4, 6 * cm),
              figure(PLOTS / "final_model_confusion.png", "Fig. 9 - Confusion matrix, final model (student-test candidates)", W / 2 - 4, 6 * cm))]),
          para("Confusion matrices for all six models are in <i>results/plots/*_confusion.png</i>.", SMALL),
          PageBreak()]

# 4 Bayesian optimisation
u, t = bo.loc["lightgbm_untuned"], bo.loc["lightgbm_tuned"]
n_c, n_p, n_f = (trials.state == "COMPLETE").sum(), (trials.state == "PRUNED").sum(), (trials.state == "FAIL").sum()
bp = {k.replace("params_", ""): v for k, v in best_trial.items() if k.startswith("params_")}
story += [para("4. Bayesian hyperparameter optimisation (LightGBM)", H1),
          table([["Question", "Answer"],
                 ["1. Hyperparameters searched", "n_estimators, learning_rate, num_leaves, min_child_samples, subsample, colsample_bytree, reg_lambda"],
                 ["2. Ranges / distributions", "trees 100–800 (step 50); learning rate 0.01–0.2 (log); leaves 15–255 (log); "
                                              "min rows 20–1000 (log); subsample 0.5–1; colsample 0.5–1; L2 0.001–10 (log)"],
                 ["3. Objective", "Maximise mean 5-fold GroupKFold macro-F1 on the 80 train trips (student-test not used)"],
                 ["4. Trials", f"30 = {n_c} completed + {n_p} pruned early (MedianPruner); plus {n_f} interrupted trial marked FAIL. "
                               f"TPE sampler, seed 42; {trials.duration_s.sum() / 60:.0f} min of trial time"],
                 ["5. Best values (trial %d)" % best_trial.number,
                  f"{int(bp['n_estimators'])} trees, lr {bp['learning_rate']:.4f}, {int(bp['num_leaves'])} leaves, "
                  f"min {int(bp['min_child_samples'])} rows, subsample {bp['subsample']:.2f}, colsample {bp['colsample_bytree']:.2f}, "
                  f"L2 {bp['reg_lambda']:.4f}"],
                 ["6. Better than untuned?", "Yes on every quality metric, but the CV gain is within the fold spread"]],
                widths=[W * 0.27, W * 0.73]),
          Spacer(1, 6),
          table([["LightGBM", "CV macro-F1", "Test Top-1", "Test macro-F1", "Log loss", "Inference time"],
                 ["Untuned", f4(u.cv_macro_f1_mean), pc(u.test_top1_acc), f4(u.test_macro_f1), f4(u.test_log_loss), f"{u.test_inference_s:.2f} s"],
                 ["Tuned (final)", f4(t.cv_macro_f1_mean), pc(t.test_top1_acc), f4(t.test_macro_f1), f4(t.test_log_loss), f"{t.test_inference_s:.2f} s"]],
                bold_last=True),
          para("Same folds, metrics and partitions for both. The initial settings were already close to optimal; the cost of tuning is about "
               "2× slower inference (600 vs 160 trees). Record: <i>results/bayes_opt_trials.csv</i>, <i>results/optuna_lightgbm.db</i>.", SMALL),
          two(figure(PLOTS / "bayes_opt_history.png", "Fig. 10 - Optimisation history (TPE)", W / 2 - 4, 5.5 * cm),
              figure(PLOTS / "bayes_opt_importance.png", "Fig. 11 - Hyperparameter importance (fANOVA)", W / 2 - 4, 5.5 * cm))]

# 5 HMM
story += [para("5. HMM and Viterbi route reconstruction", H1),
          table([["Question", "Our implementation (src/pathsnap/hmm.py)"],
                 ["1. Hidden states", "The candidate arcs of each GPS observation"],
                 ["2. Emission probabilities", "log of the classifier probability, normalised over the observation's candidates"],
                 ["3. Transition probabilities", "log p = -|driving distance - straight-line distance| / (β · Δt); driving distance is the "
                                                 "shortest path in the road network (Dijkstra on the sub-network near the trip)"],
                 ["4. Missing candidates / GPS gaps", "If no arc is within 50 m, the 3 nearest are used. β is multiplied by the time gap Δt, "
                                                       "so long gaps are penalised less"],
                 ["5. Disconnected segments", f"Unreachable transitions get a fixed log-probability ({hp['unreachable']:.0f}) instead of -∞, "
                                              "so decoding never fails and can break the route when the data demands it"],
                 ["6. Did HMM help?", "Yes - see table below and Fig. 7"]],
                widths=[W * 0.27, W * 0.73]),
          Spacer(1, 6),
          para(f"Parameters were tuned on the <b>train</b> trips using out-of-fold classifier scores (β = {hp['beta']}, "
               f"unreachable = {hp['unreachable']:.0f}); student-test is used only for reporting."),
          table([["Final model (student-test, mean of 10 trips)", "Top-1", "Route continuity", "Route F1"],
                 ["Classifier only", pc(hmm.loc["classifier_only", "top1_acc"]), f"{hmm.loc['classifier_only', 'route_continuity']:.3f}", f"{hmm.loc['classifier_only', 'route_f1']:.3f}"],
                 ["+ HMM (default β = 5, unreachable = -30)", pc(hmm.loc["hmm_default", "top1_acc"]), f"{hmm.loc['hmm_default', 'route_continuity']:.3f}", f"{hmm.loc['hmm_default', 'route_f1']:.3f}"],
                 ["+ HMM (tuned)", pc(hmm.loc["hmm_tuned", "top1_acc"]), f"{hmm.loc['hmm_tuned', 'route_continuity']:.3f}", f"{hmm.loc['hmm_tuned', 'route_f1']:.3f}"]],
                widths=[W * 0.46, W * 0.18, W * 0.18, W * 0.18], bold_last=True),
          Spacer(1, 4),
          *bullets([
              f"HMM decoding adds {(hmm.loc['hmm_tuned', 'top1_acc'] - hmm.loc['classifier_only', 'top1_acc']) * 100:.1f} points of Top-1 and raises route "
              "continuity from about 0.78 to 0.91; it improves continuity for every model (Fig. 7).",
              "<b>Where it did not help:</b> with logistic regression and default parameters, trip 00000007 dropped from 90.1 % to 85.1 %. "
              "The vehicle was manoeuvring slowly about 30 m beside a two-way road and the decoder locked onto the opposite-direction arc. "
              "Tuning fixed this (99.7 %). A rule penalising backward movement along an arc was tested and gave no gain on train, so it is not used.",
              "After decoding, the tuned model is 0.4–0.5 points behind Random Forest and untuned LightGBM on these 10 trips; this is within "
              "noise for 10 trips, and the model choice was not changed based on test results.",
          ]),
          figure(PLOTS / "hmm_tuning_heatmap.png", "Fig. 12 - HMM parameter search on train trips (out-of-fold scores); "
                 "the three panels are nearly identical, i.e. the backward-move rule had no effect", max_h=8 * cm),
          PageBreak()]

# 6 Folium
story += [para("6. Geographical visualisation (Folium)", H1),
          para("Generated by <i>scripts/make_map.py</i> (interactive HTML committed in <i>results/maps/</i>). Layers: GPS observations "
               "(blue), nearby road network (grey, toggle), reference route (green), classifier-only route (orange), HMM / Viterbi "
               "route (red), start and end markers, layer control; the extent is fitted to the trip."),
          figure(FIG / "folium_trip00000077_final_model.jpg",
                 "Fig. 13 - Student-test trip 00000077, final model: the HMM route (red) lies on the reference route (green)", max_h=11 * cm)]

# 7 API / UI
ui_shot = SHOTS / "ui_laptop2.png"
story += [para("7. API, UI and deployment", H1),
          *bullets([
              "<b>Laptop 1 - FastAPI</b> (<i>api/main.py</i>): <i>/health</i>, <i>/trips</i>, <i>/trips/summary</i>, <i>/predict</i>. "
              "Loads the final model and HMM parameters. The reference route is read only on the server, after prediction, to compute "
              "accuracy, and is never sent to the UI. Every request is logged.",
              "<b>Laptop 2 - Streamlit UI</b> (<i>ui/app.py</i>): trip list, route map, Top-1 / continuity / route F1 and a "
              "classifier-vs-HMM comparison.",
              "Launch: <i>run_api.bat &lt;evaluation folder&gt;</i> on Laptop 1, <i>run_ui.bat &lt;Laptop-1 IP&gt;</i> on Laptop 2; "
              "<i>setup_firewall.bat</i> opens port 8000.",
          ]),
          figure(ui_shot if ui_shot.exists() else FIG / "ui_trip00000000_student_test.jpg",
                 "Fig. 14 - Working UI" + ("" if ui_shot.exists() else " (student-test trip 00000000, captured during testing)"),
                 max_h=8.5 * cm),
          figure(SHOTS / "api_logs.png", "Fig. 15 - API responses in the CMD window of the receiving laptop (Laptop 1)", max_h=8 * cm)
          if (SHOTS / "api_logs.png").exists() else
          placeholder("Insert screenshot: Laptop 1 CMD showing  POST /predict from &lt;Laptop 2 IP&gt; -&gt; 200",
                      "Fig. 15 - API responses on the receiving laptop (save as report/screenshots/api_logs.png and rebuild)")]

# 8 integrity
story += [para("8. Academic integrity and AI use", H1),
          para("We used Claude (Anthropic) as an AI coding assistant during implementation and to help draft the structure of this "
               "report. All experiments were run by the team; every number in this report is produced by the code in the repository "
               "and can be reproduced with the commands in the README. Each team member can explain the code, results and design "
               "choices. No evaluator-test data was used for training, hyperparameter tuning, decoder tuning or model selection.")]


def on_page(c, doc):
    if doc.page == 1:
        return
    c.saveState()
    c.setFont(FONT, 7.5)
    c.setFillColor(MUTED)
    c.drawString(1.8 * cm, 1.1 * cm, "PathSnap · Technical Report · UE24CS352A")
    c.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"Page {doc.page}")
    c.restoreState()


doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm,
                        bottomMargin=1.7 * cm, title="PathSnap Technical Report",
                        author=", ".join(n for n, _ in TEAM))
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print("Saved", OUT)
