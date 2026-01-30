import pandas as pd
from pathlib import Path

# ---- config ----
DATA_PATH = Path(".")  # put your csv/parquet files here
FILE_NAME = "ubdated.csv"          # set later, e.g. "churn_1.csv"


def load_data(file_name: str) -> pd.DataFrame:
    file_path = DATA_PATH / file_name

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix in [".parquet", ".pq"]:
        df = pd.read_parquet(file_path)
    else:
        raise ValueError("Unsupported file type")

    print("✅ Data loaded successfully")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())

    return df


if __name__ == "__main__":
    if FILE_NAME is None:
        raise ValueError("Set FILE_NAME before running the script")

    load_data(FILE_NAME)
