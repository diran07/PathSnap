"""Per-point reference labels derived from the reference route.

The .route file lists the arcs the vehicle travelled, in order, but there is
no per-GPS-point label. We derive one with a monotonic alignment: every GPS
point is assigned to one route position, positions never go backwards in time,
and the total point-to-arc distance is minimised (dynamic programming over
route arcs within a search radius of each point).

Points with no route arc within LABEL_MAX_DIST_M get label -1 (unlabelled);
they are excluded from accuracy metrics. This label is itself an estimate,
which should be acknowledged in the report.
"""
import numpy as np
from scipy.spatial import cKDTree

from . import config
from .geo import point_segment
from .network import Prepared


def reference_arcs(P: Prepared, route=None, max_dist=config.LABEL_MAX_DIST_M, densify=10.0):
    route = P.trip.route if route is None else route
    n = len(P.pts)
    if route is None or len(route) == 0:
        return np.full(n, -1, dtype=np.int64)
    A, B = P.a_xy[route], P.b_xy[route]
    M = len(route)

    # densify route arcs into sample points for a quick radius search
    samp_pos, samp_xy = [], []
    for j in range(M):
        m = max(1, int(np.ceil(P.arc_len[route[j]] / densify)))
        f = np.linspace(0, 1, m + 1)
        samp_xy.append(A[j] + f[:, None] * (B[j] - A[j]))
        samp_pos.append(np.full(m + 1, j))
    samp_xy = np.vstack(samp_xy)
    samp_pos = np.concatenate(samp_pos)
    tree = cKDTree(samp_xy)
    hits = tree.query_ball_point(P.pts, r=max_dist + densify)

    INF = 1e18
    prev_pos = None   # sorted route positions of previous step's states
    prev_cost = None
    back = []         # per point: (positions, index of chosen prev state)
    states = []
    for i in range(n):
        pos = np.unique(samp_pos[hits[i]]) if hits[i] else np.array([], dtype=np.int64)
        if len(pos):
            d, _ = point_segment(np.repeat(P.pts[i][None], len(pos), 0), A[pos], B[pos])
            keep = d <= max_dist
            pos, d = pos[keep], d[keep]
        if len(pos) == 0:
            states.append(pos); back.append(None)
            continue
        if prev_pos is None:
            cost = d.copy(); bp = np.full(len(pos), -1)
        else:
            # best previous state with position <= current (monotonic)
            cm = np.minimum.accumulate(prev_cost)
            am = np.zeros(len(prev_cost), dtype=np.int64)
            for t in range(1, len(prev_cost)):
                am[t] = am[t - 1] if prev_cost[t] >= cm[t - 1] else t
            idx = np.searchsorted(prev_pos, pos, side="right") - 1
            ok = idx >= 0
            cost = np.full(len(pos), INF); bp = np.full(len(pos), -1)
            cost[ok] = d[ok] + cm[idx[ok]]
            bp[ok] = am[idx[ok]]
            if not ok.any():  # cannot continue monotonically: restart
                cost = d.copy()
        states.append(pos); back.append(bp)
        prev_pos, prev_cost = pos, cost
        prev_cost_last_i = i

    labels = np.full(n, -1, dtype=np.int64)
    if prev_pos is None:
        return labels
    # Backtrack. back[i][j] indexes into the states of the previous *labelled*
    # point (points without any state are skipped). -1 means the chain was
    # restarted there, so we pick the nearest state of the previous point.
    j = int(np.argmin(prev_cost))
    i = prev_cost_last_i
    while i >= 0:
        labels[i] = route[states[i][j]]
        bj = back[i][j]
        i -= 1
        while i >= 0 and back[i] is None:
            i -= 1
        if i < 0:
            break
        if bj >= 0:
            j = int(bj)
        else:
            d, _ = point_segment(np.repeat(P.pts[i][None], len(states[i]), 0),
                                 A[states[i]], B[states[i]])
            j = int(np.argmin(d))
    return labels
