from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = PROJECT_ROOT / "3_years_churn.csv"
CLEANED_FILE = PROJECT_ROOT / "cleaned_events.parquet"
ACTIVITY_FILE = PROJECT_ROOT / "activity_events.parquet"
ACTIVITY_DS_DIR = PROJECT_ROOT / "activity_events_ds"
SCORES_PARQUET = PROJECT_ROOT / "customer_scores.parquet"
SCORES_CSV = PROJECT_ROOT / "customer_scores.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_FILE = ARTIFACTS_DIR / "churn_model_lgbm.joblib"
FEATURES_FILE = ARTIFACTS_DIR / "feature_list_lgbm.json"
META_FILE = ARTIFACTS_DIR / "train_meta_lgbm.json"
ROLLING_EVAL_FILE = ARTIFACTS_DIR / "rolling_eval_lgbm.csv"
IMPORTANCE_FILE = ARTIFACTS_DIR / "feature_importance_lgbm.csv"
BUCKET_THRESHOLDS_FILE = ARTIFACTS_DIR / "bucket_thresholds.json"

ID_COL = "external_customerkey"
TIME_COL = "event_time"
EVENT_COL = "interaction_type"
REQUIRED_COLS = [ID_COL, TIME_COL, EVENT_COL]
TEXT_COLS = [ID_COL, EVENT_COL, "channel", "shop"]

ACTIVITY_EVENTS = {"emarsys_open", "emarsys_click", "order"}
ACTIVITY_PREFIX = "emarsys_sessions_"
ACTIVITY_EVENT_TYPES = [
    "emarsys_open", "emarsys_click", "order",
    "emarsys_sessions_content_url", "emarsys_sessions_content_category",
    "emarsys_sessions_content_tag", "emarsys_sessions_content_title",
    "emarsys_sessions_view", "emarsys_sessions_category_view",
    "emarsys_sessions_cart_update", "emarsys_sessions_purchase",
]

CHURN_WINDOW_DAYS = 30
LOOKBACK_DAYS = 90
ROLLING_STEP_DAYS = 30
MAX_POOL_SNAPSHOTS = 0
N_EVAL_SNAPSHOTS = 12
TARGET = "churn_30d"

THRESH_VERY_HIGH = 0.8
THRESH_HIGH = 0.6
THRESH_MEDIUM = 0.4
BUCKET_VERY_HIGH = "very_high"
BUCKET_HIGH = "high"
BUCKET_MEDIUM = "medium"
BUCKET_LOW = "low"
BUCKET_WEAK_SIGNAL = "weak_signal"

FEATURE_WINDOWS = [30, 60, 90]
MAX_RECENCY_DAYS = 365
DEFAULT_RECENCY_DAYS = 999
