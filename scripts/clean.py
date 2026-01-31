from pathlib import Path
import pandas as pd

RAW_FILE = Path("ubdated.csv")
OUT_FILE = Path("cleaned_events.parquet")

# Keep only what the whole project needs
USE_COLS = [
    "external_customerkey",
    "event_time",
    "interaction_type",
    "channel",
    "shop",
]

CRITICAL = ["external_customerkey", "event_time", "interaction_type", "shop"]


def clean_events(raw_path: Path, out_path: Path) -> None:
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw file: {raw_path}")

    # Load only required columns (faster + less memory)
    df = pd.read_csv(raw_path, usecols=USE_COLS)

    rows_before = len(df)

    # Basic cleanup
    for c in ["external_customerkey", "interaction_type", "channel", "shop"]:
        df[c] = df[c].astype("string").str.strip()

    df["interaction_type"] = df["interaction_type"].str.lower()
    df["channel"] = df["channel"].str.lower()

    # Parse time
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce", utc=True)

    # Drop invalid rows
    df = df.dropna(subset=CRITICAL).copy()

    # Enforce dtypes (good for parquet + groupbys later)
    df["external_customerkey"] = df["external_customerkey"].astype("string")
    df["interaction_type"] = df["interaction_type"].astype("string")
    df["channel"] = df["channel"].astype("string")
    df["shop"] = df["shop"].astype("string")

    rows_after = len(df)

    # Save
    df.to_parquet(out_path, index=False)

    # Report (one output)
    print("Raw file   :", raw_path)
    print("Saved      :", out_path)
    print("Rows before:", rows_before)
    print("Rows after :", rows_after)
    print("Dropped    :", rows_before - rows_after)
    print("Time range :", df["event_time"].min(), "→", df["event_time"].max())
    print("Columns    :", list(df.columns))


def main():
    clean_events(RAW_FILE, OUT_FILE)


if __name__ == "__main__":
    main()
