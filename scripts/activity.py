from pathlib import Path
import pandas as pd

IN_FILE = Path("cleaned_events.parquet")
OUT_FILE = Path("activity_events.parquet")

# Your agreed activity events
KEEP_EVENTS = {
    "emarsys_open",
    "emarsys_click",
    "order",
}

KEEP_PREFIX = "emarsys_sessions_"


def build_activity(in_path: Path, out_path: Path) -> None:
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path}. Run: python scripts/clean.py")

    df = pd.read_parquet(in_path)

    rows_before = len(df)

    # Ensure consistent casing
    df["interaction_type"] = df["interaction_type"].astype("string").str.strip().str.lower()

    mask = df["interaction_type"].isin(KEEP_EVENTS) | df["interaction_type"].str.startswith(KEEP_PREFIX, na=False)
    activity = df.loc[mask].copy()

    rows_after = len(activity)

    activity.to_parquet(out_path, index=False)

    # One clean report
    print("Input      :", in_path)
    print("Saved      :", out_path)
    print("Rows before:", rows_before)
    print("Rows after :", rows_after)
    print("Dropped    :", rows_before - rows_after)
    print("Time range :", activity["event_time"].min(), "→", activity["event_time"].max())
    print("\nEvent counts:")
    print(activity["interaction_type"].value_counts())


def main():
    build_activity(IN_FILE, OUT_FILE)


if __name__ == "__main__":
    main()
