"""Central configuration: paths and default parameters.

Change values here (or via environment variables) rather than hard-coding
them inside scripts, so every experiment is reproducible.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("PATHSNAP_DATA", ROOT / "data" / "raw"))
TRAIN_DIR = DATA_DIR / "train"
STUDENT_TEST_DIR = DATA_DIR / "student_test"
# Evaluator trips are NOT given to students. On evaluation day, point this to
# the folder the organisers provide (API laptop only).
EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", STUDENT_TEST_DIR))

PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

SEED = 42

# Candidate generation
CAND_RADIUS_M = 50.0      # search radius around each GPS point
CAND_TOP_K = 8            # keep at most K nearest arcs per point
CAND_FALLBACK_K = 3       # if nothing is within the radius, keep the K nearest anyway

# Reference labelling
LABEL_MAX_DIST_M = 100.0  # points further than this from the reference route are unlabelled

# HMM defaults (tune these on train / student_test only)
HMM_BETA = 5.0            # transition scale (metres)
HMM_EMISSION_WEIGHT = 1.0
HMM_UNREACHABLE_LOGP = -30.0
HMM_SP_LIMIT_M = 3000.0   # Dijkstra search limit
