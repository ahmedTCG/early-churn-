import pandas as pd
from churn.features import build_customer_features

ACTIVITY_FILE = "activity_events.parquet"
OUTPUT_FILE = "customer_features_labeled.parquet"
CHURN_DAYS = 30

def main():
    events = pd.read_parquet(ACTIVITY_FILE, columns=["external_customerkey", "event_time", "interaction_type"])
    events["event_time"] = pd.to_datetime(events["event_time"], errors="coerce", utc=True)
    events = events.dropna(subset=["external_customerkey", "event_time", "interaction_type"])

    max_time = events["event_time"].max()
    snapshot_time = max_time - pd.Timedelta(days=CHURN_DAYS)
    window_end = snapshot_time + pd.Timedelta(days=CHURN_DAYS)

    # 1) Features computed at the SAME snapshot_time
    hist = events[events["event_time"] <= snapshot_time].copy()
    features = build_customer_features(hist, snapshot_time=snapshot_time)

    # 2) Labels from the future window
    future = events[(events["event_time"] > snapshot_time) & (events["event_time"] <= window_end)]
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

if __name__ == "__main__":
    main()
