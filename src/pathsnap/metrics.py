"""Evaluation metrics used across the project."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score


def candidate_metrics(df: pd.DataFrame) -> dict:
    """Candidate-level binary metrics.

    df needs: pt, y (1 if candidate is the reference arc), p (model score).
    The predicted positive is the top-scored candidate of each point, so the
    macro-F1 reflects the actual decision rule rather than an arbitrary 0.5 cut.
    """
    pred = np.zeros(len(df), dtype=int)
    pred[df.index.get_indexer(df.groupby("pt")["p"].idxmax())] = 1  # one positive per point
    out = {
        "macro_f1": f1_score(df["y"], pred, average="macro"),
        "precision": precision_score(df["y"], pred, zero_division=0),
        "recall": recall_score(df["y"], pred, zero_division=0),
    }
    p = np.clip(df["p"].values, 1e-7, 1 - 1e-7)
    out["log_loss"] = log_loss(df["y"], p, labels=[0, 1])
    return out


def pointwise_accuracy(pred_arc, ref_arc) -> float:
    """Observation-level Top-1 accuracy over labelled points."""
    m = ref_arc >= 0
    return float((pred_arc[m] == ref_arc[m]).mean()) if m.any() else float("nan")


def candidate_miss_rate(cands: pd.DataFrame, ref_arc) -> float:
    lab = np.where(ref_arc >= 0)[0]
    have = set(zip(cands["pt"].values, cands["arc"].values))
    miss = sum((i, ref_arc[i]) not in have for i in lab)
    return miss / max(len(lab), 1)


def route_continuity(pred_arc, arcs) -> float:
    """Fraction of arc changes where the next arc starts where the previous ends."""
    p = pred_arc[pred_arc >= 0]
    ch = np.where(p[1:] != p[:-1])[0]
    if len(ch) == 0:
        return 1.0
    a, b = p[ch], p[ch + 1]
    return float((arcs[a, 1] == arcs[b, 0]).mean())


def route_set_scores(pred_arc, route, arc_len) -> dict:
    """Sequence-level overlap between predicted and reference arc sets (length weighted)."""
    P = set(int(x) for x in pred_arc if x >= 0)
    R = set(int(x) for x in route)
    inter = sum(arc_len[a] for a in P & R)
    lp = sum(arc_len[a] for a in P) or 1.0
    lr = sum(arc_len[a] for a in R) or 1.0
    prec, rec = inter / lp, inter / lr
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"route_precision": prec, "route_recall": rec, "route_f1": f1}
