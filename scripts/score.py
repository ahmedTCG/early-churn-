from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import joblib

from churn.features import build_customer_features

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*encountered in matmul.*")
np.seterr(over="ignore", divide="ignore", invalid="ignore")

ID_COL = "external_customerkey"
LOOKBACK_DAYS = 90

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIVITY_FILE = PROJECT_ROOT / "activity_events.parquet"

ART_DIR = PROJECT_ROOT / "artifacts"
MODEL_FILE = ART_DIR / "churn_model.joblib"
FEATURES_FILE = ART_DIR / "feature_list.json"

OUT_PARQUET = PROJECT_ROOT / "customer_scores.parquet"
OUT_CSV = PROJECT_ROOT / "customer_scores.csv"


def get_max_time() -> pd.Timestamp:
    s = pd.to_datetime(
        pd.read_parquet(ACTIVITY_FILE, columns=["event_time"])["event_time"],
        errors="coerce",
        utc=True,
    )
    return s.max()


def load_events(window_start: pd.Timestamp, window_end: pd.Timestamp) -> pd.DataFrame:
    cols = [ID_COL, "event_time", "interaction_type"]
    df = pd.read_parquet(ACTIVITY_FILE, columns=cols)

    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce", utc=True)
    df = df.dropna(subset=[ID_COL, "event_time", "interaction_type"]).copy()
    df = df[(df["event_time"] > window_start) & (df["event_time"] <= window_end)].copy()

    df[ID_COL] = df[ID_COL].astype("string").str.strip().astype("category")
    df["interaction_type"] = df["interaction_type"].astype("string").str.strip().str.lower().astype("category")
    return df


def make_X(feats: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    X = feats.drop(columns=[ID_COL], errors="ignore")
    X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0)

    if "recency_days" in X.columns:
        X["recency_days"] = X["recency_days"].clip(upper=365)

    for c in list(X.columns):
        if c.startswith(("n_events_last_", "active_days_last_", "cnt_", "n_session_events_", "n_email_engagement_", "n_orders_")):
            X[c] = np.log1p(X[c])

    for c in feature_list:
        if c not in X.columns:
            X[c] = 0

    return X[feature_list]


def main():
    if not ACTIVITY_FILE.exists():
        raise FileNotFoundError("Missing activity_events.parquet. Run: python scripts/activity.py")

    if not MODEL_FILE.exists() or not FEATURES_FILE.exists():
        raise FileNotFoundError("Missing model artifacts. Run: python scripts/train.py")

    model = joblib.load(MODEL_FILE)
    feature_list = json.loads(FEATURES_FILE.read_text(encoding="utf-8"))

    snapshot_time = get_max_time()
    window_start = snapshot_time - pd.Timedelta(days=LOOKBACK_DAYS)

    events = load_events(window_start, snapshot_time)

    feats = build_customer_features(events, snapshot_time=snapshot_time)
    X = make_X(feats, feature_list)

    proba = model.predict_proba(X)[:, 1]

    out = feats[[ID_COL]].copy()
    out["snapshot_time"] = snapshot_time
    out["churn_probability"] = proba

    out.to_parquet(OUT_PARQUET, index=False)
    out.to_csv(OUT_CSV, index=False)

    print("Snapshot time:", snapshot_time)
    print("Rows scored  :", len(out))
    print("Saved        :", OUT_PARQUET, "and", OUT_CSV)


if __name__ == "__main__":
    main()
