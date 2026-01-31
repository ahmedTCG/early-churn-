from pathlib import Path
import shutil

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "activity_events.parquet"
OUT_DIR = PROJECT_ROOT / "activity_events_ds"  # activity_events_ds/2024-01/*.parquet

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing {INPUT_FILE}. Run: python scripts/run.py activity")

    df = pd.read_parquet(INPUT_FILE, columns=["external_customerkey", "event_time", "interaction_type"])
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce", utc=True)
    df = df.dropna(subset=["external_customerkey", "event_time", "interaction_type"]).copy()

    df["event_month"] = df["event_time"].dt.strftime("%Y-%m")

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Rows:", len(df))
    print("Months:", df["event_month"].nunique())
    print("Range :", df["event_time"].min(), "→", df["event_time"].max())

    # Write one parquet file per month folder
    for m, g in df.groupby("event_month", sort=True):
        month_dir = OUT_DIR / m
        month_dir.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(g.drop(columns=["event_month"]), preserve_index=False)
        pq.write_table(table, month_dir / "part.parquet")

    print("\nSaved partitioned dataset to:", OUT_DIR)

if __name__ == "__main__":
    main()
