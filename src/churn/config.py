from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# -----------------------------
# Input / Output files
# -----------------------------
RAW_FILE = PROJECT_ROOT / "ubdated.csv"
CLEAN_FILE = PROJECT_ROOT / "cleaned_events.parquet"

# -----------------------------
# Cleaning configuration
# -----------------------------
CRITICAL_COLS = ["external_customerkey", "event_time", "interaction_type", "shop"]
TEXT_COLS = ["external_customerkey", "interaction_type", "channel", "shop"]
DROP_COLS = ["incoming_outgoing", "amount"]
