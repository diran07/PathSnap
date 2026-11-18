"""Projected trip geometry, candidate generation and road-network shortest paths."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from . import config
from .geo import LocalProjection, bearing, point_segment
from .io import Trip, clean_track


@dataclass
class Prepared:
    trip: Trip
    proj: LocalProjection
    track: pd.DataFrame      # cleaned track
    pts: np.ndarray          # (n,2) metres
    node_xy: np.ndarray      # (N,2)
    a_xy: np.ndarray         # (A,2) arc start
    b_xy: np.ndarray         # (A,2) arc end
    arc_len: np.ndarray      # (A,)
    arc_bearing: np.ndarray  # (A,)


def prepare(trip: Trip) -> Prepared:
    track = clean_track(trip.track)
    proj = LocalProjection(track["lon"].mean(), track["lat"].mean())
    pts = proj.to_xy(track["lon"].values, track["lat"].values)
    node_xy = proj.to_xy(trip.nodes[:, 0], trip.nodes[:, 1])
    a = node_xy[trip.arcs[:, 0]]
    b = node_xy[trip.arcs[:, 1]]
    d = b - a
    return Prepared(trip, proj, track, pts, node_xy, a, b,
                    np.linalg.norm(d, axis=1), bearing(d[:, 0], d[:, 1]))


def _arc_samples(P: Prepared, step):
    """Points every <= step metres along every arc, with the arc id of each sample."""
    m = np.maximum(1, np.ceil(P.arc_len / step).astype(np.int64))
    arc_of = np.repeat(np.arange(len(m)), m + 1)
    start = np.repeat(np.cumsum(np.r_[0, m[:-1] + 1]), m + 1)
    f = (np.arange(len(arc_of)) - start) / m[arc_of]
    xy = P.a_xy[arc_of] + f[:, None] * (P.b_xy[arc_of] - P.a_xy[arc_of])
    return xy, arc_of


def generate_candidates(P: Prepared, radius=config.CAND_RADIUS_M, k=config.CAND_TOP_K,
                        fallback_k=config.CAND_FALLBACK_K, step=25.0) -> pd.DataFrame:
    """Candidate road segments for every GPS point.

    Arcs are densified into samples every <= `step` m. If an arc is within
    `radius` of a point, one of its samples is within radius + step/2, so a
    KD-tree radius query over samples finds every true candidate; exact
    point-to-segment distances are then computed for the hits.
    Returns columns: pt, arc, dist, frac.
    """
    samp_xy, samp_arc = _arc_samples(P, step)
    tree = cKDTree(samp_xy)
    rows = []
    for s in range(0, len(P.pts), 2000):          # chunked to bound memory
        chunk = P.pts[s:s + 2000]
        hits = tree.query_ball_point(chunk, r=radius + step / 2)
        pt_idx = np.repeat(np.arange(s, s + len(chunk)), [len(h) for h in hits])
        arc_idx = samp_arc[np.fromiter((j for h in hits for j in h), dtype=np.int64,
                                       count=len(pt_idx))]
        pa = pd.DataFrame({"pt": pt_idx, "arc": arc_idx}).drop_duplicates()
        d, f = point_segment(P.pts[pa.pt.values], P.a_xy[pa.arc.values], P.b_xy[pa.arc.values])
        pa["dist"], pa["frac"] = d, f
        rows.append(pa[pa["dist"] <= radius])
    df = pd.concat(rows, ignore_index=True)
    mid = (P.a_xy + P.b_xy) / 2
    tree = cKDTree(mid)

    # fallback for points with no candidate inside the radius (e.g. GPS outliers)
    missing = np.setdiff1d(np.arange(len(P.pts)), df["pt"].unique())
    if len(missing):
        _, nn = tree.query(P.pts[missing], k=min(fallback_k * 4, len(mid)))
        nn = np.atleast_2d(nn)
        mp = np.repeat(missing, nn.shape[1])
        ma = nn.reshape(-1)
        d2, f2 = point_segment(P.pts[mp], P.a_xy[ma], P.b_xy[ma])
        fb = pd.DataFrame({"pt": mp, "arc": ma, "dist": d2, "frac": f2})
        fb = fb.sort_values(["pt", "dist"]).groupby("pt").head(fallback_k)
        df = pd.concat([df, fb], ignore_index=True)

    df = df.sort_values(["pt", "dist"]).groupby("pt").head(k).reset_index(drop=True)
    return df


class ShortestPaths:
    """Lazy, cached directed shortest-path distances between nodes.

    Only the part of the network near the trajectory is used, which keeps
    Dijkstra fast even for networks with >100k arcs.
    """

    def __init__(self, P: Prepared, buffer_m=500.0, limit=config.HMM_SP_LIMIT_M):
        tree = cKDTree(P.pts)
        dn, _ = tree.query(P.node_xy, k=1, distance_upper_bound=buffer_m)
        keep_node = np.isfinite(dn)
        arcs = P.trip.arcs
        keep_arc = keep_node[arcs[:, 0]] & keep_node[arcs[:, 1]]
        self.global_to_local = -np.ones(len(P.node_xy), dtype=np.int64)
        local_nodes = np.where(keep_node)[0]
        self.global_to_local[local_nodes] = np.arange(len(local_nodes))
        u = self.global_to_local[arcs[keep_arc, 0]]
        v = self.global_to_local[arcs[keep_arc, 1]]
        w = np.maximum(P.arc_len[keep_arc], 1e-3)
        n = len(local_nodes)
        # keep the shortest of parallel arcs
        g = pd.DataFrame({"u": u, "v": v, "w": w}).groupby(["u", "v"], as_index=False)["w"].min()
        self.graph = csr_matrix((g["w"].values, (g["u"].values, g["v"].values)), shape=(n, n))
        self.limit = limit
        self._cache = {}

    def precompute(self, sources):
        src = [s for s in set(int(x) for x in sources) if s not in self._cache]
        loc = [self.global_to_local[s] for s in src]
        pairs = [(s, l) for s, l in zip(src, loc) if l >= 0]
        for s, l in zip(src, loc):
            if l < 0:
                self._cache[s] = None
        for i in range(0, len(pairs), 256):
            chunk = pairs[i:i + 256]
            D = dijkstra(self.graph, directed=True, indices=[l for _, l in chunk], limit=self.limit)
            for (s, _), row in zip(chunk, np.atleast_2d(D)):
                self._cache[s] = row.astype(np.float32)

    def dist(self, src_node: int, dst_node: int) -> float:
        if src_node == dst_node:
            return 0.0
        if src_node not in self._cache:
            self.precompute([src_node])
        row = self._cache[src_node]
        l = self.global_to_local[dst_node]
        if row is None or l < 0:
            return np.inf
        return float(row[l])
