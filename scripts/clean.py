from pathlib import Path
import pandas as pd

RAW_FILE = Path("3_years_churn.csv")
OUT_FILE = Path("cleaned_events.parquet")

# Keep only what the whole project needs
USE_COLS = [
    "external_customerkey",
    "event_time",
    "interaction_type",
    "channel",
    "shop",
]

CRITICAL = ["external_customerkey", "event_time", "interaction_type"]


def clean_events(raw_path: Path, out_path: Path) -> None:
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw file: {raw_path}")

    # Load only required columns (faster + less memory)
        # Load only available required columns (supports files with or without channel/shop)
    header = pd.read_csv(raw_path, nrows=0)
    available = set(header.columns)
    required = {"external_customerkey", "event_time", "interaction_type"}
    missing = required - available
    if missing:
        raise ValueError(f"Raw file is missing required columns: {sorted(missing)}")

    optional = ["channel", "shop"]
    usecols = list(required) + [c for c in optional if c in available]
    df = pd.read_csv(raw_path, usecols=usecols)

    rows_before = len(df)

    # Basic cleanup (schema-safe: channel/shop may not exist)
    for c in ["external_customerkey", "interaction_type", "channel", "shop"]:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()

    if "interaction_type" in df.columns:
        df["interaction_type"] = df["interaction_type"].str.lower()
    if "channel" in df.columns:
        df["channel"] = df["channel"].str.lower()

    # Parse time
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce", utc=True)

    # Drop invalid rows
    df = df.dropna(subset=CRITICAL).copy()

    # Enforce dtypes (schema-safe)
    for c in ["external_customerkey", "interaction_type", "channel", "shop"]:
        if c in df.columns:
            df[c] = df[c].astype("string")

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
