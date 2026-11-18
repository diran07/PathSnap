"""HMM + Viterbi route reconstruction.

Hidden states   : the candidate arcs of each GPS point.
Emissions       : log of the classifier's per-point normalised candidate probability.
Transitions     : Newson & Krumm style - a move from candidate a (at t) to b (at t+1)
                  is likely when the driving distance along the network is close to
                  the straight-line distance between the two GPS points:
                      log p = -|route_dist - gps_dist| / beta_t
                  beta grows with the time gap so large GPS gaps are tolerated.
Unreachable     : if b cannot be reached from a within the Dijkstra limit, the
                  transition gets a fixed low log-probability instead of -inf, so
                  the decoder can still "break" the route rather than fail.
"""
import numpy as np
import pandas as pd

from . import config
from .network import Prepared, ShortestPaths


def viterbi_decode(P: Prepared, scored: pd.DataFrame, sp: ShortestPaths = None,
                   beta=config.HMM_BETA, w_emit=config.HMM_EMISSION_WEIGHT,
                   unreachable=config.HMM_UNREACHABLE_LOGP, eps=1e-6):
    """scored: candidate rows with columns pt, arc, frac, p (normalised per point).

    Returns an array with one predicted arc per GPS point.
    """
    arcs = P.trip.arcs
    if sp is None:
        sp = ShortestPaths(P)
    scored = scored.sort_values(["pt", "p"], ascending=[True, False])
    groups = {k: g for k, g in scored.groupby("pt", sort=True)}
    pts_order = sorted(groups)
    sp.precompute(arcs[scored["arc"].unique(), 1])

    t = P.track["t"].values
    deltas, backs, states = [], [], []
    prev = None
    for i in pts_order:
        g = groups[i]
        arc = g["arc"].values
        pos = g["frac"].values * P.arc_len[arc]
        emit = w_emit * np.log(g["p"].values + eps)
        if prev is None:
            delta = emit
            bp = np.full(len(arc), -1)
        else:
            parc, ppos, pdelta, pi = prev
            gc = float(np.linalg.norm(P.pts[i] - P.pts[pi]))
            dt = max(t[i] - t[pi], 1.0)
            b = beta * dt
            T = np.empty((len(parc), len(arc)))
            for a_i, a in enumerate(parc):
                for b_i, c in enumerate(arc):
                    if a == c:
                        rd = abs(pos[b_i] - ppos[a_i])
                    else:
                        s = sp.dist(arcs[a, 1], arcs[c, 0])
                        rd = (P.arc_len[a] - ppos[a_i]) + s + pos[b_i]
                    T[a_i, b_i] = unreachable if not np.isfinite(rd) else -abs(rd - gc) / b
            M = pdelta[:, None] + T
            bp = M.argmax(0)
            delta = M.max(0) + emit
        deltas.append(delta); backs.append(bp); states.append(arc)
        prev = (arc, pos, delta, i)

    path = np.empty(len(pts_order), dtype=np.int64)
    j = int(np.argmax(deltas[-1]))
    for k in range(len(pts_order) - 1, -1, -1):
        path[k] = states[k][j]
        j = backs[k][j]
    out = np.full(len(P.pts), -1, dtype=np.int64)
    out[pts_order] = path
    return out
