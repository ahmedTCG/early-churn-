from pathlib import Path
import shutil

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

IN_FILE = Path("activity_events.parquet")
OUT_DIR = Path("activity_events_ds")  # activity_events_ds/YYYY-MM/part.parquet


def partition_by_month(in_path: Path, out_dir: Path) -> None:
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path}. Run: python scripts/activity.py")

    df = pd.read_parquet(in_path, columns=["external_customerkey", "event_time", "interaction_type"])
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce", utc=True)
    df = df.dropna(subset=["external_customerkey", "event_time", "interaction_type"]).copy()

    df["event_month"] = df["event_time"].dt.strftime("%Y-%m")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    months = sorted(df["event_month"].unique().tolist())

    for m in months:
        g = df[df["event_month"] == m].drop(columns=["event_month"])
        month_dir = out_dir / m
        month_dir.mkdir(parents=True, exist_ok=True)

        table = pa.Table.from_pandas(g, preserve_index=False)
        pq.write_table(table, month_dir / "part.parquet")

    print("Input  :", in_path)
    print("Saved  :", out_dir)
    print("Rows   :", len(df))
    print("Months :", len(months))
    print("Range  :", df["event_time"].min(), "→", df["event_time"].max())


def main():
    partition_by_month(IN_FILE, OUT_DIR)


if __name__ == "__main__":
    main()
