"""Feature engineering for (GPS point, candidate arc) pairs.

The model scores every candidate of a point; the candidate with the highest
score is the classifier-only prediction. Features describe how well the
candidate agrees with the point itself and with its neighbouring points.
"""
import numpy as np
import pandas as pd

from .geo import angle_diff, bearing, point_segment
from .network import Prepared

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio", "n_cands",
    "heading_diff", "heading_cos", "speed", "dt_prev", "arc_len", "frac",
    "ctx_dist_mean", "ctx_dist_max", "is_nearest",
]


def point_kinematics(P: Prepared, min_move=2.0):
    """Heading (deg) and speed (m/s) per point from its neighbours.

    Heading is undefined when the vehicle is (almost) stationary; in that case
    the last valid heading is carried forward.
    """
    pts, t = P.pts, P.track["t"].values
    n = len(pts)
    prev = np.r_[0, np.arange(n - 1)]
    nxt = np.r_[np.arange(1, n), n - 1]
    d = pts[nxt] - pts[prev]
    dt = np.maximum(t[nxt] - t[prev], 1e-6)
    move = np.linalg.norm(d, axis=1)
    speed = move / dt
    head = bearing(d[:, 0], d[:, 1])
    head = pd.Series(np.where(move >= min_move, head, np.nan)).ffill().bfill().fillna(0).values
    dt_prev = np.r_[0.0, np.diff(t)]
    return head, speed, dt_prev, move >= min_move


def build_features(P: Prepared, cands: pd.DataFrame, ctx=2) -> pd.DataFrame:
    df = cands.copy()
    head, speed, dt_prev, moving = point_kinematics(P)
    g = df.groupby("pt")["dist"]
    df["dist_rank"] = g.rank(method="first") - 1
    dmin = g.transform("min")
    df["dist_minus_min"] = df["dist"] - dmin
    df["dist_ratio"] = df["dist"] / (dmin + 1.0)
    df["n_cands"] = g.transform("size")
    df["is_nearest"] = (df["dist_rank"] == 0).astype(int)

    pt = df["pt"].values
    arc = df["arc"].values
    hd = angle_diff(head[pt], P.arc_bearing[arc])
    # heading is uninformative while stopped
    hd = np.where(moving[pt], hd, 90.0)
    df["heading_diff"] = hd
    df["heading_cos"] = np.cos(np.radians(hd))
    df["speed"] = speed[pt]
    df["dt_prev"] = dt_prev[pt]
    df["arc_len"] = P.arc_len[arc]

    # distance from the same arc to neighbouring GPS points (temporal context)
    n = len(P.pts)
    ctx_d = []
    for off in [o for o in range(-ctx, ctx + 1) if o != 0]:
        q = np.clip(pt + off, 0, n - 1)
        d, _ = point_segment(P.pts[q], P.a_xy[arc], P.b_xy[arc])
        ctx_d.append(d)
    ctx_d = np.vstack(ctx_d)
    df["ctx_dist_mean"] = ctx_d.mean(0)
    df["ctx_dist_max"] = ctx_d.max(0)
    return df
