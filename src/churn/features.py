import pandas as pd
import numpy as np
from typing import Optional, List

# Activity event types (agreed list)
ACTIVITY_EVENT_TYPES: List[str] = [
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


def _ensure_datetime_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)


def _window_slice(df: pd.DataFrame, snapshot_time: pd.Timestamp, window_days: int) -> pd.DataFrame:
    start = snapshot_time - pd.Timedelta(days=window_days)
    return df[(df["event_time"] > start) & (df["event_time"] <= snapshot_time)].copy()


def _event_mix_counts(df_window: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """
    Return wide counts of ACTIVITY_EVENT_TYPES for a given window, ensuring
    ALL expected columns exist (filled with 0 if missing).
    """
    mix = df_window[df_window["interaction_type"].isin(ACTIVITY_EVENT_TYPES)]

    counts = (
        mix.groupby(["external_customerkey", "interaction_type"])
        .size()
        .unstack(fill_value=0)
    )

    # Ensure all expected event columns exist, even if zero in this window
    for e in ACTIVITY_EVENT_TYPES:
        if e not in counts.columns:
            counts[e] = 0

    # Keep stable column order
    counts = counts[ACTIVITY_EVENT_TYPES]

    # Rename to nice column names and restore key
    counts = counts.rename(columns={e: f"cnt_{e}_last_{window_days}d" for e in ACTIVITY_EVENT_TYPES})
    counts = counts.reset_index()

    return counts


def build_customer_features(
    df_events: pd.DataFrame,
    snapshot_time: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Build the full customer-level feature table at snapshot_time.

    Input requirements:
      - external_customerkey
      - event_time (datetime-like)
      - interaction_type

    Output:
      - One row per external_customerkey with all engineered features.
    """
    df = df_events.copy()

    if not {"external_customerkey", "event_time", "interaction_type"}.issubset(df.columns):
        raise ValueError("df_events must include external_customerkey, event_time, interaction_type")

    df["event_time"] = _ensure_datetime_utc(df["event_time"])
    df["interaction_type"] = df["interaction_type"].astype("string").str.strip().str.lower()
    df["external_customerkey"] = df["external_customerkey"].astype("string").str.strip()

    df = df.dropna(subset=["external_customerkey", "event_time", "interaction_type"])

    if snapshot_time is None:
        snapshot_time = df["event_time"].max()

    df_hist = df[df["event_time"] <= snapshot_time].copy()

    customers = df_hist[["external_customerkey"]].drop_duplicates()

    # Recency
    last_event = (
        df_hist.groupby("external_customerkey", as_index=False)["event_time"]
        .max()
        .rename(columns={"event_time": "last_event_time"})
    )
    last_event["recency_days"] = (snapshot_time - last_event["last_event_time"]).dt.days

    features = customers.merge(
        last_event[["external_customerkey", "recency_days"]],
        on="external_customerkey",
        how="left"
    )

    # Frequency + active days + mix counts (30/60/90)
    for w in (30, 60, 90):
        df_w = _window_slice(df_hist, snapshot_time, w)

        freq = (
            df_w.groupby("external_customerkey")
            .size()
            .rename(f"n_events_last_{w}d")
            .reset_index()
        )

        df_w["event_date"] = df_w["event_time"].dt.date
        active_days = (
            df_w.groupby("external_customerkey")["event_date"]
            .nunique()
            .rename(f"active_days_last_{w}d")
            .reset_index()
        )

        mix_counts = _event_mix_counts(df_w, w)

        features = (
            features
            .merge(freq, on="external_customerkey", how="left")
            .merge(active_days, on="external_customerkey", how="left")
            .merge(mix_counts, on="external_customerkey", how="left")
        )

    # Trend features
    features["freq_trend_30_60"] = features["n_events_last_30d"] / (features["n_events_last_60d"] + 1)
    features["freq_trend_60_90"] = features["n_events_last_60d"] / (features["n_events_last_90d"] + 1)

    # Activity ratio
    features["active_ratio_30d"] = features["active_days_last_30d"] / 30

    # Inactivity flags
    features["inactive_14d"] = (features["recency_days"] > 14).astype(int)
    features["inactive_30d"] = (features["recency_days"] > 30).astype(int)

    # History flag (will always be 1 if built from activity-only customers)
    features["has_any_activity_hist"] = (~features["recency_days"].isna()).astype(int)

    # Family counts + ratios (30d)
    session_cols_30 = [f"cnt_{e}_last_30d" for e in ACTIVITY_EVENT_TYPES if e.startswith("emarsys_sessions_")]
    features["n_session_events_last_30d"] = features[session_cols_30].sum(axis=1)

    features["n_email_engagement_last_30d"] = (
        features["cnt_emarsys_open_last_30d"] + features["cnt_emarsys_click_last_30d"]
    )

    features["n_orders_last_30d"] = features["cnt_order_last_30d"]

    denom_30 = features["n_events_last_30d"].fillna(0) + 1
    features["session_share_30d"] = features["n_session_events_last_30d"] / denom_30
    features["email_share_30d"] = features["n_email_engagement_last_30d"] / denom_30

    # Intensity proxy (30d)
    features["events_per_active_day_30d"] = features["n_events_last_30d"] / (features["active_days_last_30d"] + 1)

    # Click-to-open ratio (30d)
    opens_30 = features["cnt_emarsys_open_last_30d"]
    clicks_30 = features["cnt_emarsys_click_last_30d"]
    features["click_to_open_30d"] = clicks_30 / (opens_30 + 1)

    # Fill NAs
    count_prefixes = ("n_events_last_", "active_days_last_", "cnt_", "n_session_events_last_", "n_email_engagement_last_", "n_orders_last_")
    count_cols = [c for c in features.columns if c.startswith(count_prefixes)]
    features[count_cols] = features[count_cols].fillna(0)

    for c in ["freq_trend_30_60", "freq_trend_60_90", "active_ratio_30d", "session_share_30d", "email_share_30d",
              "events_per_active_day_30d", "click_to_open_30d"]:
        features[c] = features[c].replace([np.inf, -np.inf], np.nan).fillna(0)

    features["recency_days"] = features["recency_days"].fillna(999).astype(int)

    cols = ["external_customerkey"] + [c for c in features.columns if c != "external_customerkey"]
    return features[cols]
