from pathlib import Path

# =============================================
# PROJECT PATHS
# =============================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Input files
RAW_FILE = PROJECT_ROOT / "3_years_churn.csv"

# Intermediate files
CLEANED_FILE = PROJECT_ROOT / "cleaned_events.parquet"
ACTIVITY_FILE = PROJECT_ROOT / "activity_events.parquet"
ACTIVITY_DS_DIR = PROJECT_ROOT / "activity_events_ds"

# Output files
SCORES_PARQUET = PROJECT_ROOT / "customer_scores.parquet"
SCORES_CSV = PROJECT_ROOT / "customer_scores.csv"

# Artifacts directory
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_FILE = ARTIFACTS_DIR / "churn_model.joblib"
FEATURES_FILE = ARTIFACTS_DIR / "feature_list.json"
META_FILE = ARTIFACTS_DIR / "train_meta.json"
ROLLING_EVAL_FILE = ARTIFACTS_DIR / "rolling_eval.csv"
IMPORTANCE_FILE = ARTIFACTS_DIR / "feature_importance.csv"
BUCKET_THRESHOLDS_FILE = ARTIFACTS_DIR / "bucket_thresholds.json"


# =============================================
# DATA SCHEMA
# =============================================
ID_COL = "external_customerkey"
TIME_COL = "event_time"
EVENT_COL = "interaction_type"

# Required columns for cleaning
REQUIRED_COLS = [ID_COL, TIME_COL, EVENT_COL]

# Text columns to normalize
TEXT_COLS = [ID_COL, EVENT_COL, "channel", "shop"]


# =============================================
# ACTIVITY EVENTS
# =============================================
# Events that count as "activity" (engagement)
ACTIVITY_EVENTS = {
    "emarsys_open",
    "emarsys_click",
    "order",
}

# Prefix for session events
ACTIVITY_PREFIX = "emarsys_sessions_"

# Full list of activity event types (for feature engineering)
ACTIVITY_EVENT_TYPES = [
    "emarsys_open",
    "emarsys_click",
    "order",
    "emarsys_sessions_content_url",
    "emarsys_sessions_content_category",
    "emarsys_sessions_content_tag",
    "emarsys_sessions_content_title",
    "emarsys_sessions_view",
    "emarsys_sessions_category_view",
    "emarsys_sessions_cart_update",
    "emarsys_sessions_purchase",
]


# =============================================
# MODEL CONFIGURATION
# =============================================
# Churn definition
CHURN_WINDOW_DAYS = 30  # No activity in X days = churned
LOOKBACK_DAYS = 90      # Feature window

# Training configuration
ROLLING_STEP_DAYS = 30      # Step between training snapshots
MAX_POOL_SNAPSHOTS = 0      # 0 = use ALL available snapshots
N_EVAL_SNAPSHOTS = 12       # Number of evaluation snapshots

# Model hyperparameters
MODEL_PARAMS = {
    "loss": "log_loss",
    "penalty": "l2",
    "alpha": 1e-4,
    "max_iter": 5000,
    "tol": 1e-3,
    "class_weight": "balanced",
    "random_state": 42,
}

# Target column
TARGET = "churn_30d"


# =============================================
# SCORING CONFIGURATION
# =============================================
# Fixed probability thresholds for buckets
THRESH_VERY_HIGH = 0.8
THRESH_HIGH = 0.6
THRESH_MEDIUM = 0.4

# Bucket names
BUCKET_VERY_HIGH = "very_high"
BUCKET_HIGH = "high"
BUCKET_MEDIUM = "medium"
BUCKET_LOW = "low"
BUCKET_WEAK_SIGNAL = "weak_signal"


# =============================================
# FEATURE ENGINEERING
# =============================================
# Time windows for features (days)
FEATURE_WINDOWS = [30, 60, 90]

# Maximum recency clipping
MAX_RECENCY_DAYS = 365

# Default recency for missing values
DEFAULT_RECENCY_DAYS = 999
