import pandas as pd
from .config import CRITICAL_COLS, TEXT_COLS, DROP_COLS

def clean_text(s: pd.Series) -> pd.Series:
    s = s.astype("string")
    s = s.str.strip().str.lower()
    s = s.replace({"": pd.NA, "nan": pd.NA, "none": pd.NA, "null": pd.NA})
    return s

def clean_events(raw_path, output_path) -> None:
    df = pd.read_csv(raw_path)
    rows_before = len(df)

    # standardize column names
    df.columns = [c.lower() for c in df.columns]

    # parse timestamps
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce", utc=True)

    # normalize text columns
    for c in TEXT_COLS:
        if c in df.columns:
            df[c] = clean_text(df[c])

    # drop structurally invalid rows
    df = df.dropna(subset=CRITICAL_COLS)

    # drop non-informative / unused columns
    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(columns=[c])

    # time sanity: drop future events
    now_utc = pd.Timestamp.utcnow()
    df = df[df["event_time"] <= now_utc]

    rows_after = len(df)

    print("Rows before:", rows_before)
    print("Rows after :", rows_after)
    print("Dropped     :", rows_before - rows_after)

    print("\nFinal schema:")
    print(df.dtypes)

    print("\nEvent time range:")
    print(df["event_time"].min(), "→", df["event_time"].max())

    df.to_parquet(output_path, index=False)
    print(f"\nSaved cleaned data to: {output_path}")
