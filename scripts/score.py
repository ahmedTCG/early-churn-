import json
import warnings

import numpy as np
import pandas as pd
import joblib

from churn.config import (
    CLEANED_FILE,
    ACTIVITY_FILE,
    ARTIFACTS_DIR,
    MODEL_FILE,
    FEATURES_FILE,
    SCORES_PARQUET,
    SCORES_CSV,
    BUCKET_THRESHOLDS_FILE,
    ID_COL,
    TIME_COL,
    EVENT_COL,
    THRESH_VERY_HIGH,
    THRESH_HIGH,
    THRESH_MEDIUM,
    BUCKET_VERY_HIGH,
    BUCKET_HIGH,
    BUCKET_MEDIUM,
    BUCKET_LOW,
    BUCKET_WEAK_SIGNAL,
    MAX_RECENCY_DAYS,
)
from churn.features import build_customer_features

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*encountered in matmul.*")
np.seterr(over="ignore", divide="ignore", invalid="ignore")


def get_max_time() -> pd.Timestamp:
    s = pd.to_datetime(
        pd.read_parquet(ACTIVITY_FILE, columns=[TIME_COL])[TIME_COL],
        errors="coerce",
        utc=True,
    )
    return s.max()


def load_all_activity_events() -> pd.DataFrame:
    cols = [ID_COL, TIME_COL, EVENT_COL]
    df = pd.read_parquet(ACTIVITY_FILE, columns=cols)

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)
    df = df.dropna(subset=[ID_COL, TIME_COL, EVENT_COL]).copy()

    df[ID_COL] = df[ID_COL].astype("string").str.strip().astype("category")
    df[EVENT_COL] = df[EVENT_COL].astype("string").str.strip().str.lower().astype("category")
    return df


def get_weak_signal_customers(activity_customers: set, snapshot_time: pd.Timestamp) -> pd.DataFrame:
    if not CLEANED_FILE.exists():
        return pd.DataFrame(columns=[ID_COL])

    cleaned = pd.read_parquet(CLEANED_FILE, columns=[ID_COL])
    cleaned[ID_COL] = cleaned[ID_COL].astype("string").str.strip()
    all_customers = set(cleaned[ID_COL].unique())

    weak_signal_ids = all_customers - activity_customers

    if not weak_signal_ids:
        return pd.DataFrame(columns=[ID_COL])

    weak_df = pd.DataFrame({
        ID_COL: list(weak_signal_ids),
        "snapshot_time": snapshot_time,
        "churn_probability": 1.0,
        "churn_bucket": BUCKET_WEAK_SIGNAL,
        "churn_action": 1,
    })

    return weak_df


def make_X(feats: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    X = feats.drop(columns=[ID_COL], errors="ignore")
    X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0)

    if "recency_days" in X.columns:
        X["recency_days"] = X["recency_days"].clip(upper=MAX_RECENCY_DAYS)

    for c in list(X.columns):
        if c.startswith(("n_events_last_", "active_days_last_", "cnt_", "n_session_events_", "n_email_engagement_", "n_orders_")):
            X[c] = np.log1p(X[c])

    for c in feature_list:
        if c not in X.columns:
            X[c] = 0

    return X[feature_list]


def bucketize(p: float) -> str:
    if p >= THRESH_VERY_HIGH:
        return BUCKET_VERY_HIGH
    elif p >= THRESH_HIGH:
        return BUCKET_HIGH
    elif p >= THRESH_MEDIUM:
        return BUCKET_MEDIUM
    else:
        return BUCKET_LOW


def main():
    if not ACTIVITY_FILE.exists():
        raise FileNotFoundError("Missing activity_events.parquet. Run: python scripts/activity.py")

    if not MODEL_FILE.exists() or not FEATURES_FILE.exists():
        raise FileNotFoundError("Missing model artifacts. Run: python scripts/train.py")

    model = joblib.load(MODEL_FILE)
    feature_list = json.loads(FEATURES_FILE.read_text(encoding="utf-8"))

    snapshot_time = get_max_time()

    print("Loading all activity events (full history)...")
    events = load_all_activity_events()
    events = events[events[TIME_COL] <= snapshot_time].copy()

    activity_customers = set(events[ID_COL].unique())
    print(f"Customers with activity history: {len(activity_customers):,}")

    print("Building features...")
    feats = build_customer_features(events, snapshot_time=snapshot_time)
    print(f"Features built for: {len(feats):,} customers")

    recency = feats["recency_days"]
    print(f"\nRecency distribution:")
    print(f"  Min: {recency.min()} days")
    print(f"  Max: {recency.max()} days")
    print(f"  Median: {recency.median():.0f} days")
    print(f"  Customers with recency > 90 days: {(recency > 90).sum():,}")

    X = make_X(feats, feature_list)
    proba = model.predict_proba(X)[:, 1]

    scored = feats[[ID_COL]].copy()
    scored["snapshot_time"] = snapshot_time
    scored["churn_probability"] = proba
    scored["churn_bucket"] = scored["churn_probability"].apply(bucketize)
    scored["churn_action"] = scored["churn_bucket"].isin([BUCKET_HIGH, BUCKET_VERY_HIGH]).astype(int)

    weak_signal_df = get_weak_signal_customers(activity_customers, snapshot_time)
    n_weak_signal = len(weak_signal_df)

    if n_weak_signal > 0:
        out = pd.concat([scored, weak_signal_df], ignore_index=True)
    else:
        out = scored

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    BUCKET_THRESHOLDS_FILE.write_text(
        json.dumps({
            "snapshot_time": str(snapshot_time),
            "threshold_very_high": THRESH_VERY_HIGH,
            "threshold_high": THRESH_HIGH,
            "threshold_medium": THRESH_MEDIUM,
            "action_rule": "high_or_very_high_or_weak_signal",
            "weak_signal_note": "Customers who NEVER had activity events (only bounce/cancel/unsub)",
        }, indent=2),
        encoding="utf-8"
    )

    out.to_parquet(SCORES_PARQUET, index=False)
    out.to_csv(SCORES_CSV, index=False)

    print()
    print("=" * 60)
    print("SCORING RESULTS")
    print("=" * 60)
    print(f"Snapshot time: {snapshot_time}")
    print()

    print("Customers scored by model:")
    print(f"  Total: {len(scored):,}")
    print()

    recent_active = (feats["recency_days"] <= 90).sum()
    dormant = (feats["recency_days"] > 90).sum()
    print(f"  - Active in last 90 days: {recent_active:,}")
    print(f"  - Dormant (>90 days): {dormant:,}")
    print()

    print("Bucket distribution (model-scored):")
    bucket_counts = scored["churn_bucket"].value_counts()
    bucket_pct = scored["churn_bucket"].value_counts(normalize=True)
    for bucket in [BUCKET_VERY_HIGH, BUCKET_HIGH, BUCKET_MEDIUM, BUCKET_LOW]:
        if bucket in bucket_counts.index:
            print(f"  {bucket:10}: {bucket_counts[bucket]:>7,} ({bucket_pct[bucket]:.1%})")

    print()
    print("Bucket thresholds (fixed):")
    print(f"  {BUCKET_VERY_HIGH} >= {THRESH_VERY_HIGH}")
    print(f"  {BUCKET_HIGH}      >= {THRESH_HIGH}")
    print(f"  {BUCKET_MEDIUM}    >= {THRESH_MEDIUM}")
    print(f"  {BUCKET_LOW}       <  {THRESH_MEDIUM}")

    print()
    print("Weak signal customers (NEVER had activity events):")
    print(f"  Total: {n_weak_signal:,}")
    print(f"  These customers have only bounce/cancel/unsub events")
    print(f"  Assigned: churn_probability=1.0, churn_bucket='{BUCKET_WEAK_SIGNAL}'")

    print()
    print("Combined output:")
    print(f"  Total customers: {len(out):,}")
    print(f"  - Model scored:  {len(scored):,}")
    print(f"  - Weak signal:   {n_weak_signal:,}")

    print()
    print("Overall bucket distribution:")
    overall = out["churn_bucket"].value_counts()
    for bucket in [BUCKET_VERY_HIGH, BUCKET_HIGH, BUCKET_MEDIUM, BUCKET_LOW, BUCKET_WEAK_SIGNAL]:
        if bucket in overall.index:
            pct = overall[bucket] / len(out) * 100
            print(f"  {bucket:12}: {overall[bucket]:>7,} ({pct:.1f}%)")

    print()
    print(f"Saved parquet: {SCORES_PARQUET}")
    print(f"Saved csv: {SCORES_CSV}")
    print(f"Saved bucket thresholds: {BUCKET_THRESHOLDS_FILE}")


if __name__ == "__main__":
    main()