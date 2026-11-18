# Shared paths and constants used by every script in the project
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Raw data partitions
TRAIN_DIR = ROOT / "train"
STUDENT_TEST_DIR = ROOT / "student_test"
PARTITIONS = {"train": TRAIN_DIR, "student_test": STUDENT_TEST_DIR}

# Output folders
RESULTS_DIR = ROOT / "results"
EDA_DIR = RESULTS_DIR / "eda"
EDA_PLOTS_DIR = EDA_DIR / "plots"
MAPS_DIR = RESULTS_DIR / "maps"
FEATURES_DIR = RESULTS_DIR / "features"
MODELS_DIR = RESULTS_DIR / "models"
MODEL_PLOTS_DIR = MODELS_DIR / "plots"
PREDICTIONS_DIR = MODELS_DIR / "predictions"

# Candidate generation settings
CANDIDATE_RADIUS_M = 50.0
MAX_CANDIDATES = 8
TRAIN_SAMPLE_RATE = 2  # keep every 2nd GPS point of training trips to limit dataset size
LARGE_GAP_S = 10.0     # time gap treated as a GPS dropout

# Reproducibility and cross-validation
SEED = 42
N_FOLDS = 5
