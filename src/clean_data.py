import pandas as pd

INPUT_FILE = "ubdated.csv"
OUTPUT_FILE = "cleaned_events.parquet"

CRITICAL_COLS = [
    "external_customerkey",
    "event_time",
    "interaction_type",
    "channel",
    "shop",
]

TEXT_COLS = [
    "external_customerkey",
    "interaction_type",
    "channel",
    "shop",
]

def _clean_text(s: pd.Series) -> pd.Series:
    s = s.astype("string")
    s = s.str.strip().str.lower()
    s = s.replace({"": pd.NA, "nan": pd.NA, "none": pd.NA, "null": pd.NA})
    return s

def main():
    df = pd.read_csv(INPUT_FILE)
    rows_before = len(df)

    # standardize column names
    df.columns = [c.lower() for c in df.columns]

    # parse timestamps
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce", utc=True)

    # clean text columns
    for c in TEXT_COLS:
        if c in df.columns:
            df[c] = _clean_text(df[c])

    # drop structurally invalid rows
    df = df.dropna(subset=CRITICAL_COLS)

    # drop non-informative / unused columns
    for col in ["amount", "incoming_outgoing"]:
        if col in df.columns:
            df = df.drop(columns=[col])

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

    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved cleaned data to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
