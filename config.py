"""Project configuration. Change paths and thresholds here, not inside the modules."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Elliptic dataset files (download from Kaggle: "elliptic-data-set")
RAW_DIR = ROOT / "data" / "raw"
FEATURES_FILE = "elliptic_txs_features.csv"   # no header: txId, time_step, 165 features
CLASSES_FILE = "elliptic_txs_classes.csv"     # header: txId, class  ('1' illicit, '2' licit, 'unknown')
EDGES_FILE = "elliptic_txs_edgelist.csv"      # header: txId1, txId2  (directed: txId1 -> txId2)

# Raw JSON pulled from the Blockstream Esplora API (semi-structured data step)
JSON_DIR = ROOT / "data" / "json"
ESPLORA_BASE_URL = "https://blockstream.info/api"

OUTPUT_DIR = ROOT / "outputs"
DB_PATH = OUTPUT_DIR / "elliptic.db"

# Expected structure, used by the data quality checks
EXPECTED_FEATURE_COLS = 167          # txId + time_step + 165 features
EXPECTED_TIME_STEPS = (1, 49)
VALID_CLASSES = {"1", "2", "unknown"}
CLASS_LABELS = {"1": "illicit", "2": "licit", "unknown": "unknown"}

# Anomaly detection: robust z-score = 0.6745 * (x - median) / MAD, computed per time step
MAD_THRESHOLD = 3.5

# Time-based split (as in Weber et al., 2019): features are chosen using time steps
# 1-34 only, and results are reported on time steps 35-49, which the selection never saw.
TRAIN_MAX_TIME_STEP = 34

# Feature selection on the training period. Every feature is tested in both directions
# (unusually high, unusually low); the most precise ones are kept, subject to:
N_SELECTED_FEATURES = 3          # how many features to watch
MIN_TRAIN_RECALL = 0.10          # a feature must catch at least 10% of known illicit
MIN_TRAIN_FLAGS = 50             # ignore features that flag almost nothing
MAX_CORRELATION = 0.90           # skip near-duplicates of an already chosen feature

# A transaction is flagged when at least this many selected features breach the threshold
MIN_FEATURES_BREACHED = 2
