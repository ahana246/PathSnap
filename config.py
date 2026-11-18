# Shared paths and constants used by every script in the project
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Raw data partitions
TRAIN_DIR = ROOT / "train"
STUDENT_TEST_DIR = ROOT / "student_test"
PARTITIONS = {"train": TRAIN_DIR, "student_test": STUDENT_TEST_DIR}
EVALUATOR_TEST_DIR = ROOT / "evaluator_test"  # final evaluation only: never used for training or tuning

# Output folders
RESULTS_DIR = ROOT / "results"
EDA_DIR = RESULTS_DIR / "eda"
EDA_PLOTS_DIR = EDA_DIR / "plots"
MAPS_DIR = RESULTS_DIR / "maps"
FEATURES_DIR = RESULTS_DIR / "features"
MODELS_DIR = RESULTS_DIR / "models"
MODEL_PLOTS_DIR = MODELS_DIR / "plots"
PREDICTIONS_DIR = MODELS_DIR / "predictions"
TUNING_DIR = RESULTS_DIR / "tuning"
HMM_DIR = RESULTS_DIR / "hmm"

# Inference artifacts loaded by the HMM pipeline and the API
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "final_model.joblib"
MODEL_CONFIG_PATH = ARTIFACTS_DIR / "model_config.json"
HMM_CONFIG_PATH = ARTIFACTS_DIR / "hmm_config.json"

# Candidate generation settings
CANDIDATE_RADIUS_M = 50.0
MAX_CANDIDATES = 8
TRAIN_SAMPLE_RATE = 2  # keep every 2nd GPS point of training trips to limit dataset size
LARGE_GAP_S = 10.0     # time gap treated as a GPS dropout
LABEL_MAX_DIST_M = 50.0  # GPS points farther than this from the remaining route are left unlabelled

# HMM settings that are fixed rather than tuned
HMM_GAP_SPLIT_S = 60.0     # a time gap longer than this starts a new, unconnected route segment
HMM_SEARCH_SLACK_M = 200.0  # road search reaches GPS step distance + this; beyond it a move is treated as unreachable

# Reproducibility and cross-validation
SEED = 42
N_FOLDS = 5
