"""Build candidate feature tables for the train and student_test partitions.

Output: data/processed/{train,student_test}_features.parquet
Usage:  python scripts/build_features.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from joblib import Parallel, delayed

from pathsnap import config
from pathsnap.io import list_trips, load_trip
from pathsnap.metrics import candidate_miss_rate
from pathsnap.pipeline import trip_table


def one(split_dir, tid):
    t0 = time.perf_counter()
    P, cands, feats, ref = trip_table(load_trip(split_dir, tid))
    stats = {"trip_id": tid, "n_points": len(P.pts), "n_candidates": len(cands),
             "labelled_frac": float((ref >= 0).mean()),
             "candidate_miss_rate": candidate_miss_rate(cands, ref),
             "seconds": time.perf_counter() - t0}
    return feats, stats


def main():
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, d in [("train", config.TRAIN_DIR), ("student_test", config.STUDENT_TEST_DIR)]:
        trips = list_trips(d)
        res = Parallel(n_jobs=4, verbose=5)(delayed(one)(d, t) for t in trips)
        feats = pd.concat([r[0] for r in res], ignore_index=True)
        stats = pd.DataFrame([r[1] for r in res])
        feats.to_parquet(config.PROCESSED_DIR / f"{name}_features.parquet", index=False)
        stats.to_csv(config.RESULTS_DIR / f"{name}_candidate_stats.csv", index=False)
        print(f"{name}: {len(trips)} trips, {len(feats):,} candidate rows, "
              f"mean miss rate {stats.candidate_miss_rate.mean():.3f}")


if __name__ == "__main__":
    main()
