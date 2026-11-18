"""HMM + Viterbi route reconstruction.

Hidden states   : the candidate arcs of each GPS point.
Emissions       : log of the classifier's per-point normalised candidate probability.
Transitions     : Newson & Krumm style - a move from candidate a (at t) to b (at t+1)
                  is likely when the driving distance along the network is close to
                  the straight-line distance between the two GPS points:
                      log p = -|route_dist - gps_dist| / (beta * dt)
                  beta is scaled by the time gap dt, so large GPS gaps are tolerated.
Unreachable     : if b cannot be reached from a within the Dijkstra limit, the
                  transition gets a fixed low log-probability instead of -inf, so
                  the decoder can still "break" the route rather than fail.

The work is split in two:
  build_lattice()  - parameter-independent geometry (route distances via Dijkstra);
                     this is the expensive part and is computed once per trip.
  decode()         - Viterbi for given (beta, w_emit, unreachable); cheap, so the
                     parameters can be tuned by re-running it many times.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .network import Prepared, ShortestPaths


@dataclass
class Lattice:
    n_points: int
    pts: np.ndarray        # GPS point index of each step
    arcs: list             # per step: candidate arc ids
    logp: list             # per step: log classifier probability
    rd: list               # per step (from 1): route-distance matrix prev x cur (inf = unreachable)
    back: list             # per step: metres moved BACKWARDS when staying on the same arc (else 0)
    rd_uturn: list         # per step: route distance if that backward move is a real U-turn loop
    gc: np.ndarray         # per step: straight-line distance to previous point
    dt: np.ndarray         # per step: time gap to previous point (>= 1 s)


def build_lattice(P: Prepared, scored: pd.DataFrame, sp: ShortestPaths = None, eps=1e-6) -> Lattice:
    """scored: candidate rows with columns pt, arc, frac, p (normalised per point)."""
    net = P.trip.arcs
    if sp is None:
        sp = ShortestPaths(P)
    scored = scored.sort_values(["pt", "p"], ascending=[True, False])
    sp.precompute(net[scored["arc"].unique(), 1])
    t = P.track["t"].values

    pts, arcs, logp, rd, back, rd_u, gc, dt = [], [], [], [], [], [], [], []
    prev = None
    for i, g in scored.groupby("pt", sort=True):
        arc = g["arc"].values
        pos = g["frac"].values * P.arc_len[arc]
        if prev is None:
            rd.append(None); back.append(None); rd_u.append(None); gc.append(0.0); dt.append(1.0)
        else:
            parc, ppos, pi = prev
            R = np.empty((len(parc), len(arc)))
            Bk = np.zeros_like(R)
            U = np.full_like(R, np.inf)
            for a_i, a in enumerate(parc):
                rem = P.arc_len[a] - ppos[a_i]
                for b_i, c in enumerate(arc):
                    if a == c:
                        R[a_i, b_i] = abs(pos[b_i] - ppos[a_i])
                        if pos[b_i] < ppos[a_i]:
                            # moving backwards on a directed arc: only possible by looping
                            # round (end of arc -> back to its start -> along it again)
                            Bk[a_i, b_i] = ppos[a_i] - pos[b_i]
                            U[a_i, b_i] = rem + sp.dist(net[a, 1], net[a, 0]) + pos[b_i]
                    else:
                        R[a_i, b_i] = rem + sp.dist(net[a, 1], net[c, 0]) + pos[b_i]
            rd.append(R); back.append(Bk); rd_u.append(U)
            gc.append(float(np.linalg.norm(P.pts[i] - P.pts[pi])))
            dt.append(max(float(t[i] - t[pi]), 1.0))
        pts.append(i); arcs.append(arc); logp.append(np.log(g["p"].values + eps))
        prev = (arc, pos, i)
    return Lattice(len(P.pts), np.array(pts), arcs, logp, rd, back, rd_u, np.array(gc), np.array(dt))


def decode(L: Lattice, beta=config.HMM_BETA, w_emit=config.HMM_EMISSION_WEIGHT,
           unreachable=config.HMM_UNREACHABLE_LOGP,
           backward_tol=config.HMM_BACKWARD_TOL_M) -> np.ndarray:
    """Viterbi over the lattice. Returns one predicted arc per GPS point (-1 if none).

    backward_tol: backward movement along the same arc up to this many metres is
    treated as GPS noise; anything more must be explained as a U-turn loop.
    Use np.inf to disable (the original behaviour).
    """
    backs = []
    delta = w_emit * L.logp[0]
    for k in range(1, len(L.arcs)):
        R = L.rd[k]
        if np.isfinite(backward_tol):
            R = np.where(L.back[k] > backward_tol, L.rd_uturn[k], R)
        T = -np.abs(R - L.gc[k]) / (beta * L.dt[k])
        T[~np.isfinite(R)] = unreachable
        M = delta[:, None] + T
        bp = M.argmax(0)
        backs.append(bp)
        delta = M[bp, np.arange(M.shape[1])] + w_emit * L.logp[k]
    path = np.empty(len(L.arcs), dtype=np.int64)
    j = int(np.argmax(delta))
    for k in range(len(L.arcs) - 1, -1, -1):
        path[k] = L.arcs[k][j]
        if k:
            j = backs[k - 1][j]
    out = np.full(L.n_points, -1, dtype=np.int64)
    out[L.pts] = path
    return out


def viterbi_decode(P: Prepared, scored: pd.DataFrame, sp: ShortestPaths = None, **params):
    """Convenience wrapper: build the lattice and decode in one call."""
    return decode(build_lattice(P, scored, sp), **params)
