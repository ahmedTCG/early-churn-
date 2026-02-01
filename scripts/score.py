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
OUT_BUCKETED_CSV = PROJECT_ROOT / "customer_scores_bucketed.csv"
OUT_BUCKET_THRESHOLDS = ART_DIR / "bucket_thresholds.json"


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

    # --- Bucketization (percentile-based) ---
    # Buckets are relative to this scoring run's population:
    # very_high = top 10%, high = next 20%, medium = next 30%, low = bottom 40%
    p90 = out["churn_probability"].quantile(0.90)
    p70 = out["churn_probability"].quantile(0.70)
    p40 = out["churn_probability"].quantile(0.40)

    def bucketize(p: float) -> str:
        if p >= p90:
            return "very_high"
        elif p >= p70:
            return "high"
        elif p >= p40:
            return "medium"
        else:
            return "low"

    out["churn_bucket"] = out["churn_probability"].apply(bucketize)

    # --- Align decision threshold to buckets ---
    # Recommended default: act on top 30% risk (high + very_high)
    out["churn_action"] = out["churn_bucket"].isin(["high", "very_high"]).astype(int)

    # Persist the thresholds used for this scoring run (reproducible + auditable)
    ART_DIR.mkdir(exist_ok=True)
    OUT_BUCKET_THRESHOLDS.write_text(
        json.dumps({
            "snapshot_time": str(snapshot_time),
            "p40_medium_min": float(p40),
            "p70_high_min": float(p70),
            "p90_very_high_min": float(p90),
            "action_rule": "high_or_very_high",
        }, indent=2),
        encoding="utf-8"
    )

    out.to_parquet(OUT_PARQUET, index=False)
    out.to_csv(OUT_CSV, index=False)
    out.to_csv(OUT_BUCKETED_CSV, index=False)

    print("Bucket distribution:")
    print(out["churn_bucket"].value_counts(normalize=True).round(3))
    print("Bucket thresholds:")
    print(f"very_high >= {p90:.3f}")
    print(f"high      >= {p70:.3f}")
    print(f"medium    >= {p40:.3f}")

    print("Snapshot time:", snapshot_time)
    print("Rows scored  :", len(out))
    print("Saved parquet:", OUT_PARQUET)
    print("Saved csv    :", OUT_CSV)
    if "churn_bucket" in out.columns:
        print("Saved bucket :", OUT_BUCKETED_CSV)
    if "OUT_BUCKET_THRESHOLDS" in globals():
        print("Saved bucket thresholds:", OUT_BUCKET_THRESHOLDS)


if __name__ == "__main__":
    main()
