import sys
import pandas as pd

from churn.config import RAW_FILE, CLEAN_FILE, PROJECT_ROOT
from churn.cleaning import clean_events
from churn.features import build_customer_features

USAGE = "Usage: python scripts/run.py clean | activity | features | label"
CHURN_DAYS = 30


def step_clean():
    clean_events(RAW_FILE, CLEAN_FILE)


def step_activity():
    INPUT_FILE = PROJECT_ROOT / "cleaned_events.parquet"
    OUTPUT_FILE = PROJECT_ROOT / "activity_events.parquet"

    df = pd.read_parquet(INPUT_FILE)

    keep_mask = (
        df["interaction_type"].eq("emarsys_open")
        | df["interaction_type"].eq("emarsys_click")
        | df["interaction_type"].eq("order")
        | df["interaction_type"].str.startswith("emarsys_sessions_", na=False)
    )

    activity = df.loc[keep_mask].copy()

    print("Rows (cleaned):", len(df))
    print("Rows (activity):", len(activity))
    print("\nActivity interaction_type distribution:")
    print(activity["interaction_type"].value_counts())

    activity.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE}")


def step_features():
    INPUT_FILE = PROJECT_ROOT / "activity_events.parquet"
    OUTPUT_FILE = PROJECT_ROOT / "customer_features.parquet"

    df = pd.read_parquet(INPUT_FILE)
    feats = build_customer_features(df, snapshot_time=None)

    print("Customers:", feats["external_customerkey"].nunique())
    print("Feature columns:", feats.shape[1])

    feats.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE}")


def step_label():
    ACTIVITY_FILE = PROJECT_ROOT / "activity_events.parquet"
    OUTPUT_FILE = PROJECT_ROOT / "customer_features_labeled.parquet"

    events = pd.read_parquet(
        ACTIVITY_FILE,
        columns=["external_customerkey", "event_time", "interaction_type"],
    )
    events["event_time"] = pd.to_datetime(events["event_time"], errors="coerce", utc=True)
    events = events.dropna(subset=["external_customerkey", "event_time", "interaction_type"])

    max_time = events["event_time"].max()
    snapshot_time = max_time - pd.Timedelta(days=CHURN_DAYS)
    window_end = snapshot_time + pd.Timedelta(days=CHURN_DAYS)

    # Features at snapshot_time
    hist = events[events["event_time"] <= snapshot_time].copy()
    features = build_customer_features(hist, snapshot_time=snapshot_time)

    # Labels from future window
    future = events[
        (events["event_time"] > snapshot_time) &
        (events["event_time"] <= window_end)
    ]

    future_counts = (
        future.groupby("external_customerkey")
        .size()
        .rename("n_activity_events_next_30d")
        .reset_index()
    )

    labeled = features.merge(future_counts, on="external_customerkey", how="left")
    labeled["n_activity_events_next_30d"] = labeled["n_activity_events_next_30d"].fillna(0).astype(int)
    labeled["churn_30d"] = (labeled["n_activity_events_next_30d"] == 0).astype(int)

    labeled["snapshot_time"] = snapshot_time
    labeled["window_end"] = window_end

    print("Max event time :", max_time)
    print("Snapshot time  :", snapshot_time)
    print("Window end     :", window_end)
    print("\nCustomers:", labeled["external_customerkey"].nunique())
    print("Churn distribution:")
    print(labeled["churn_30d"].value_counts(normalize=True))

    labeled.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE}")


def main():
    if len(sys.argv) != 2:
        print(USAGE)
        sys.exit(1)

    step = sys.argv[1].lower()

    if step == "clean":
        step_clean()
        return
    if step == "activity":
        step_activity()
        return
    if step == "features":
        step_features()
        return
    if step == "label":
        step_label()
        return

    print(f"Unknown step: {step}")
    print(USAGE)
    sys.exit(1)


if __name__ == "__main__":
    main()
