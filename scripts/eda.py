"""Exploratory data analysis of the raw PathSnap dataset.

Partitions analysed: train (80 trips) and student_test (10 trips).
The evaluator partition is not available to us and is never touched.

Outputs
  results/eda/inventory.csv          one row per trip: files, sizes, row counts, extents, gaps, network stats
  results/eda/data_quality.csv       per-trip quality flags
  results/eda/summary_by_partition.csv
  results/plots/eda_*.png
  results/maps/eda_<trip>.html       Folium maps (GPS, nearby network, reference route)

Usage: python scripts/eda.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from pathsnap import config
from pathsnap.io import list_trips, load_trip
from pathsnap.network import _arc_samples, prepare
from pathsnap.viz import arcs_to_lines, route_map

OUT = config.RESULTS_DIR / "eda"
PLOTS = config.RESULTS_DIR / "plots"
MAPS = config.RESULTS_DIR / "maps"
SUFFIXES = ["track", "nodes", "arcs", "route"]


def raw_checks(split_dir, tid):
    """Checks on the raw track BEFORE cleaning."""
    raw = pd.read_csv(Path(split_dir) / tid / f"{tid}.track", sep=r"\s+", header=None,
                      names=["lon", "lat", "t"])
    dt = np.diff(raw["t"].values)
    return {
        "track_cols": raw.shape[1],
        "track_dtypes": ",".join(str(x) for x in raw.dtypes),
        "missing_values": int(raw.isna().sum().sum()),
        "invalid_coords": int((~raw["lon"].between(-180, 180) | ~raw["lat"].between(-90, 90)).sum()),
        "zero_coords": int(((raw["lon"] == 0) & (raw["lat"] == 0)).sum()),
        "non_increasing_t": int((dt <= 0).sum()),
        "duplicate_t": int(raw["t"].duplicated().sum()),
    }


def network_stats(trip):
    n, arcs = len(trip.nodes), trip.arcs
    g = csr_matrix((np.ones(len(arcs)), (arcs[:, 0], arcs[:, 1])), shape=(n, n))
    n_weak, lab_w = connected_components(g, directed=True, connection="weak")
    n_strong, lab_s = connected_components(g, directed=True, connection="strong")
    deg = np.bincount(arcs.ravel(), minlength=n)
    s = set(map(tuple, arcs))
    two_way = sum((b, a) in s for a, b in arcs) / max(len(arcs), 1)
    return {
        "n_nodes": n, "n_arcs": len(arcs),
        "isolated_nodes": int((deg == 0).sum()),
        "self_loops": int((arcs[:, 0] == arcs[:, 1]).sum()),
        "weak_components": int(n_weak),
        "largest_weak_frac": float(np.bincount(lab_w).max() / n),
        "strong_components": int(n_strong),
        "largest_strong_frac": float(np.bincount(lab_s).max() / n),
        "two_way_arc_frac": two_way,
    }


def route_stats(P):
    r = P.trip.route
    if r is None:
        return {"has_route": False}
    arcs = P.trip.arcs
    breaks = int((arcs[r[:-1], 1] != arcs[r[1:], 0]).sum())
    # distance from every GPS point to the reference route (route densified every 5 m)
    sub = type("S", (), {})()
    sub.a_xy, sub.b_xy, sub.arc_len = P.a_xy[r], P.b_xy[r], P.arc_len[r]
    xy, _ = _arc_samples(sub, 5.0)
    d, _ = cKDTree(xy).query(P.pts)
    return {
        "has_route": True, "route_arcs": len(r), "route_unique_arcs": len(set(r.tolist())),
        "route_length_km": float(P.arc_len[r].sum() / 1000),
        "route_breaks": breaks,
        "gps_to_route_median_m": float(np.median(d)),
        "gps_to_route_p95_m": float(np.percentile(d, 95)),
        "gps_far_from_route_frac": float((d > config.LABEL_MAX_DIST_M).mean()),
        "_dist": d,
    }


def analyse(split_name, split_dir, tid):
    d = Path(split_dir) / tid
    row = {"partition": split_name, "trip_id": tid}
    for s in SUFFIXES:
        f = d / f"{tid}.{s}"
        row[f"has_{s}"] = f.exists()
        row[f"{s}_kb"] = f.stat().st_size / 1024 if f.exists() else np.nan
    row.update(raw_checks(split_dir, tid))
    trip = load_trip(split_dir, tid)
    P = prepare(trip)
    t = P.track["t"].values
    dt = np.diff(t)
    step = np.linalg.norm(np.diff(P.pts, axis=0), axis=1)
    speed = step / np.maximum(dt, 1e-9)
    ext = P.pts.max(0) - P.pts.min(0)
    row.update({
        "n_points_raw": len(trip.track), "n_points": len(P.pts),
        "duration_min": (t[-1] - t[0]) / 60,
        "dt_median_s": float(np.median(dt)), "dt_1s_frac": float((dt == 1).mean()),
        "dt_max_s": float(dt.max()), "gaps_over_10s": int((dt > 10).sum()),
        "gaps_over_60s": int((dt > 60).sum()),
        "gps_path_km": float(step.sum() / 1000),
        "speed_median_kmh": float(np.median(speed) * 3.6),
        "speed_p99_kmh": float(np.percentile(speed, 99) * 3.6),
        "jumps_over_200kmh": int((speed * 3.6 > 200).sum()),
        "stationary_frac": float((step < 0.5).mean()),
        "lon_min": P.track.lon.min(), "lon_max": P.track.lon.max(),
        "lat_min": P.track.lat.min(), "lat_max": P.track.lat.max(),
        "extent_x_km": ext[0] / 1000, "extent_y_km": ext[1] / 1000,
        "nodes_cover_track": bool((trip.nodes[:, 0].min() <= P.track.lon.min()) &
                                  (trip.nodes[:, 0].max() >= P.track.lon.max()) &
                                  (trip.nodes[:, 1].min() <= P.track.lat.min()) &
                                  (trip.nodes[:, 1].max() >= P.track.lat.max())),
    })
    row.update(network_stats(trip))
    rs = route_stats(P)
    dist = rs.pop("_dist", None)
    row.update(rs)
    return row, dt, speed, dist


def flags(inv):
    f = pd.DataFrame({"partition": inv.partition, "trip_id": inv.trip_id})
    f["large_time_gap(>60s)"] = inv.gaps_over_60s > 0
    f["speed_jumps(>200km/h)"] = inv.jumps_over_200kmh > 0
    f["gps_off_route(median>30m)"] = inv.gps_to_route_median_m > 30
    f["route_has_breaks"] = inv.route_breaks > 0
    f["fragmented_network"] = inv.largest_weak_frac < 0.95
    f["mostly_stationary(>40%)"] = inv.stationary_frac > 0.4
    f["n_flags"] = f.iloc[:, 2:].sum(1)
    return f


def plots(inv, all_dt, all_speed, all_dist):
    c = {"train": "#1f77b4", "student_test": "#ff7f0e"}

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for p, g in inv.groupby("partition"):
        ax[0].hist(g.n_points, bins=30, alpha=0.6, label=p, color=c[p])
        ax[1].hist(g.duration_min, bins=30, alpha=0.6, label=p, color=c[p])
        ax[2].hist(g.route_length_km, bins=30, alpha=0.6, label=p, color=c[p])
    for a, t in zip(ax, ["GPS points per trip", "Trip duration (min)", "Reference route length (km)"]):
        a.set_title(t); a.legend()
    fig.tight_layout(); fig.savefig(PLOTS / "eda_trip_sizes.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    dt = np.concatenate(all_dt)
    vals, cnts = np.unique(np.clip(dt, 0, 11), return_counts=True)
    ax[0].bar([str(int(v)) if v < 11 else ">10" for v in vals], cnts / cnts.sum())
    ax[0].set_yscale("log"); ax[0].set_title("Sampling interval (s), train + student_test")
    ax[0].set_ylabel("fraction of steps")
    sp = np.concatenate(all_speed) * 3.6
    ax[1].hist(np.clip(sp, 0, 200), bins=100); ax[1].set_yscale("log")
    ax[1].set_title("Point-to-point speed (km/h, clipped at 200)")
    fig.tight_layout(); fig.savefig(PLOTS / "eda_sampling_speed.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    dd = np.concatenate([d for d in all_dist if d is not None])
    ax.hist(np.clip(dd, 0, 150), bins=150)
    ax.axvline(config.CAND_RADIUS_M, color="r", ls="--", label=f"candidate radius {config.CAND_RADIUS_M:.0f} m")
    ax.set_yscale("log"); ax.set_title("GPS point distance to reference route (m, clipped at 150)")
    ax.legend(); fig.tight_layout(); fig.savefig(PLOTS / "eda_gps_to_route.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for p, g in inv.groupby("partition"):
        ax[0].scatter(g.n_nodes, g.n_arcs, label=p, color=c[p], alpha=0.7)
        ax[1].scatter(g.n_points, g.largest_weak_frac, label=p, color=c[p], alpha=0.7)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("nodes"); ax[0].set_ylabel("arcs"); ax[0].set_title("Road-network size per trip")
    ax[1].set_xlabel("GPS points"); ax[1].set_ylabel("largest weakly-connected component")
    ax[1].set_title("Network connectivity"); ax[0].legend()
    fig.tight_layout(); fig.savefig(PLOTS / "eda_network.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for p, g in inv.groupby("partition"):
        ax.scatter((g.lon_min + g.lon_max) / 2, (g.lat_min + g.lat_max) / 2, label=p, color=c[p], s=30)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude"); ax.set_title("Trip centres (WGS84)")
    ax.legend(); fig.tight_layout(); fig.savefig(PLOTS / "eda_trip_locations.png", dpi=120); plt.close(fig)


def folium_map(split_dir, tid):
    trip = load_trip(split_dir, tid)
    P = prepare(trip)
    near = cKDTree(P.pts).query((P.a_xy + P.b_xy) / 2, distance_upper_bound=150)[0]
    nearby = np.where(np.isfinite(near))[0]
    m = route_map(P.track[["lat", "lon"]].values,
                  {"Reference route": (arcs_to_lines(trip.nodes, trip.arcs, trip.route), "#2ca02c")},
                  network_lines=arcs_to_lines(trip.nodes, trip.arcs, nearby))
    m.save(str(MAPS / f"eda_{tid}.html"))


def main():
    for p in (OUT, PLOTS, MAPS):
        p.mkdir(parents=True, exist_ok=True)
    rows, all_dt, all_speed, all_dist = [], [], [], []
    for name, d in [("train", config.TRAIN_DIR), ("student_test", config.STUDENT_TEST_DIR)]:
        for tid in list_trips(d):
            r, dt, sp, dist = analyse(name, d, tid)
            rows.append(r); all_dt.append(dt); all_speed.append(sp); all_dist.append(dist)
            print(f"{name:12s} {tid}  {r['n_points']:6d} pts  gap {r['dt_max_s']:9.0f}s  "
                  f"route-offset {r.get('gps_to_route_median_m', np.nan):6.1f} m")
    inv = pd.DataFrame(rows)
    inv.to_csv(OUT / "inventory.csv", index=False)
    fl = flags(inv)
    fl.to_csv(OUT / "data_quality.csv", index=False)

    summ = inv.groupby("partition").agg(
        trips=("trip_id", "count"), points_total=("n_points", "sum"),
        points_median=("n_points", "median"), points_min=("n_points", "min"),
        points_max=("n_points", "max"), dt_1s_frac_mean=("dt_1s_frac", "mean"),
        missing_values=("missing_values", "sum"), duplicate_t=("duplicate_t", "sum"),
        non_increasing_t=("non_increasing_t", "sum"), trips_with_gap_60s=("gaps_over_60s", lambda s: (s > 0).sum()),
        nodes_median=("n_nodes", "median"), arcs_median=("n_arcs", "median"),
        isolated_nodes_total=("isolated_nodes", "sum"), route_breaks_total=("route_breaks", "sum"),
        gps_to_route_median_m=("gps_to_route_median_m", "median"),
    ).T
    summ.to_csv(OUT / "summary_by_partition.csv")
    print("\n", summ.round(3).to_string())
    print("\nFlagged trips:\n", fl[fl.n_flags > 0].to_string(index=False))

    plots(inv, all_dt, all_speed, all_dist)
    # maps: one typical train trip, one test trip, and the most anomalous trip
    worst = inv.loc[inv.gps_to_route_median_m.idxmax()]
    picks = [("train", config.TRAIN_DIR, "00000001"), ("student_test", config.STUDENT_TEST_DIR, "00000000"),
             (worst.partition, config.TRAIN_DIR if worst.partition == "train" else config.STUDENT_TEST_DIR,
              worst.trip_id)]
    for _, d, tid in picks:
        folium_map(d, tid)
    print("\nSaved CSVs to", OUT, "| plots to", PLOTS, "| maps to", MAPS)


if __name__ == "__main__":
    main()
