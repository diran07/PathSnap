"""Reading raw trip folders.

Each trip folder <id>/ contains:
  <id>.track  lon lat t          (GPS observations, ~1 Hz)
  <id>.nodes  lon lat            (row index = node id)
  <id>.arcs   from_node to_node  (row index = arc id, directed)
  <id>.route  arc_id             (reference route, chronological) - may be absent
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Trip:
    trip_id: str
    track: pd.DataFrame          # columns lon, lat, t
    nodes: np.ndarray            # (N, 2) lon, lat
    arcs: np.ndarray             # (A, 2) from, to
    route: Optional[np.ndarray]  # (R,) arc ids or None


def list_trips(split_dir) -> list:
    split_dir = Path(split_dir)
    return sorted(p.name for p in split_dir.iterdir() if p.is_dir())


def load_trip(split_dir, trip_id: str, with_route: bool = True) -> Trip:
    d = Path(split_dir) / trip_id
    track = pd.read_csv(d / f"{trip_id}.track", sep=r"\s+", header=None,
                        names=["lon", "lat", "t"], dtype=float)
    nodes = np.loadtxt(d / f"{trip_id}.nodes", dtype=float).reshape(-1, 2)
    arcs = np.loadtxt(d / f"{trip_id}.arcs", dtype=np.int64).reshape(-1, 2)
    route = None
    rp = d / f"{trip_id}.route"
    if with_route and rp.exists():
        route = np.loadtxt(rp, dtype=np.int64).reshape(-1)
    return Trip(trip_id, track, nodes, arcs, route)


def clean_track(track: pd.DataFrame) -> pd.DataFrame:
    """Drop invalid rows, sort by time and remove duplicate timestamps."""
    t = track.copy()
    ok = t[["lon", "lat", "t"]].notna().all(axis=1)
    ok &= t["lon"].between(-180, 180) & t["lat"].between(-90, 90)
    t = t[ok].sort_values("t", kind="stable")
    t = t.drop_duplicates(subset="t", keep="first").reset_index(drop=True)
    return t
