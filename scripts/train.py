from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import joblib
import pyarrow.dataset as ds

from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from churn.features import build_customer_features

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*encountered in matmul.*")
np.seterr(over="ignore", divide="ignore", invalid="ignore")

# -------------------------
# Config
# -------------------------
ID_COL = "external_customerkey"
TARGET = "churn_30d"

CHURN_WINDOW_DAYS = 30
LOOKBACK_DAYS = 90

# -------------------------
# Pooled rolling training config
# -------------------------
ROLLING_STEP_DAYS = 30
MAX_POOL_SNAPSHOTS = 0  # 0 = use ALL available snapshots back to data start
N_EVAL_SNAPSHOTS = 12  # Evaluate on 12 months (full year)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIVITY_FILE = PROJECT_ROOT / "activity_events.parquet"
ACTIVITY_DS_DIR = PROJECT_ROOT / "activity_events_ds"

ART_DIR = PROJECT_ROOT / "artifacts"
MODEL_FILE = ART_DIR / "churn_model.joblib"
FEATURES_FILE = ART_DIR / "feature_list.json"
META_FILE = ART_DIR / "train_meta.json"
ROLLING_EVAL_FILE = ART_DIR / "rolling_eval.csv"
IMPORTANCE_FILE = ART_DIR / "feature_importance.csv"


# -------------------------
# Data loading
# -------------------------
def _months_in_range(start: pd.Timestamp, end: pd.Timestamp) -> list:
    start_m = pd.Period(start.to_pydatetime(), freq="M")
    end_m = pd.Period(end.to_pydatetime(), freq="M")
    return [p.strftime("%Y-%m") for p in pd.period_range(start_m, end_m, freq="M")]


def load_events(window_start: pd.Timestamp, window_end: pd.Timestamp) -> pd.DataFrame:
    cols = [ID_COL, "event_time", "interaction_type"]

    if ACTIVITY_DS_DIR.exists():
        months = _months_in_range(window_start, window_end)
        dataset = ds.dataset(
            str(ACTIVITY_DS_DIR),
            format="parquet",
            partitioning=ds.partitioning(field_names=["event_month"]),
        )
        table = dataset.to_table(columns=cols, filter=ds.field("event_month").isin(months))
        df = table.to_pandas()
        using_ds = True
    else:
        if not ACTIVITY_FILE.exists():
            raise FileNotFoundError("Missing activity data. Run steps 1→3.")
        df = pd.read_parquet(ACTIVITY_FILE, columns=cols)
        using_ds = False

    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce", utc=True)
    df = df.dropna(subset=[ID_COL, "event_time", "interaction_type"]).copy()
    df = df[(df["event_time"] > window_start) & (df["event_time"] <= window_end)].copy()

    df[ID_COL] = df[ID_COL].astype("string").str.strip().astype("category")
    df["interaction_type"] = df["interaction_type"].astype("string").str.strip().str.lower().astype("category")

    df.attrs["using_partitioned_ds"] = using_ds
    return df


def get_max_time() -> pd.Timestamp:
    if ACTIVITY_FILE.exists():
        s = pd.to_datetime(pd.read_parquet(ACTIVITY_FILE, columns=["event_time"])["event_time"], errors="coerce", utc=True)
        return s.max()
    if ACTIVITY_DS_DIR.exists():
        dataset = ds.dataset(str(ACTIVITY_DS_DIR), format="parquet")
        table = dataset.to_table(columns=["event_time"])
        s = pd.to_datetime(table.column("event_time").to_pandas(), errors="coerce", utc=True)
        return s.max()
    raise FileNotFoundError("No activity data found. Run steps 1→3.")


def get_min_time() -> pd.Timestamp:
    if ACTIVITY_FILE.exists():
        s = pd.to_datetime(pd.read_parquet(ACTIVITY_FILE, columns=["event_time"])["event_time"], errors="coerce", utc=True)
        return s.min()
    if ACTIVITY_DS_DIR.exists():
        dataset = ds.dataset(str(ACTIVITY_DS_DIR), format="parquet")
        table = dataset.to_table(columns=["event_time"])
        s = pd.to_datetime(table.column("event_time").to_pandas(), errors="coerce", utc=True)
        return s.min()
    raise FileNotFoundError("No activity data found. Run steps 1→3.")


# -------------------------
# Dataset building
# -------------------------
def make_dataset(snapshot_time: pd.Timestamp) -> pd.DataFrame:
    """
    Build features and labels for a single snapshot.
    Features: events in (T-90d, T]
    Label: no activity in (T, T+30d]
    """
    window_start = snapshot_time - pd.Timedelta(days=LOOKBACK_DAYS)
    window_end = snapshot_time + pd.Timedelta(days=CHURN_WINDOW_DAYS)

    ev = load_events(window_start, window_end)

    hist = ev[ev["event_time"] <= snapshot_time].copy()
    customers = hist[[ID_COL]].drop_duplicates()

    future = ev[(ev["event_time"] > snapshot_time) & (ev["event_time"] <= window_end)]
    active_future = set(future[ID_COL].dropna().unique())

    labels = customers.copy()
    labels[TARGET] = (~labels[ID_COL].isin(active_future)).astype(int)

    feats = build_customer_features(hist, snapshot_time=snapshot_time)

    out = feats.merge(labels, on=ID_COL, how="inner")
    num = out.select_dtypes(include=[np.number]).columns
    out[num] = out[num].replace([np.inf, -np.inf], np.nan).fillna(0)

    return out


def make_X(ds_: pd.DataFrame) -> pd.DataFrame:
    X = ds_.drop(columns=[ID_COL, TARGET], errors="ignore")
    X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0)

    if "recency_days" in X.columns:
        X["recency_days"] = X["recency_days"].clip(upper=365)

    for c in list(X.columns):
        if c.startswith(("n_events_last_", "active_days_last_", "cnt_", "n_session_events_", "n_email_engagement_", "n_orders_")):
            X[c] = np.log1p(X[c])

    X = X.loc[:, X.std(axis=0) != 0]
    return X


# -------------------------
# Pooled training dataset
# -------------------------
def build_pooled_train_dataset(test_snapshot: pd.Timestamp) -> pd.DataFrame:
    """
    Build training dataset by pooling multiple rolling snapshots.
    Uses all available snapshots from test_snapshot back to earliest possible.
    
    Args:
        test_snapshot: The test snapshot time (training uses earlier snapshots)
    
    Returns:
        Concatenated DataFrame with all training snapshots
    """
    min_time = get_min_time()
    # Earliest snapshot needs LOOKBACK_DAYS of history + CHURN_WINDOW_DAYS for labels
    earliest_allowed = min_time + pd.Timedelta(days=LOOKBACK_DAYS)
    
    parts = []
    snapshot_info = []
    k = 1
    
    while True:
        snap = test_snapshot - pd.Timedelta(days=k * ROLLING_STEP_DAYS)
        
        # Stop if we don't have enough history for this snapshot
        if snap < earliest_allowed:
            break
        
        # Stop if we've reached the max (when MAX_POOL_SNAPSHOTS > 0)
        if MAX_POOL_SNAPSHOTS > 0 and len(parts) >= MAX_POOL_SNAPSHOTS:
            break
        
        ds_ = make_dataset(snap)
        ds_["_snapshot_time"] = snap  # Track which snapshot each row came from
        parts.append(ds_)
        
        churn_rate = ds_[TARGET].mean()
        snapshot_info.append({
            "snapshot": snap,
            "rows": len(ds_),
            "churn_rate": churn_rate
        })
        
        k += 1
    
    if not parts:
        raise ValueError("No snapshots available for pooled training. Check data range.")
    
    # Print snapshot summary
    print(f"\nPooled training: {len(parts)} snapshots")
    print("-" * 50)
    for info in snapshot_info:
        print(f"  {info['snapshot'].date()}: {info['rows']:,} rows, churn={info['churn_rate']:.1%}")
    print("-" * 50)
    
    pooled = pd.concat(parts, ignore_index=True)
    print(f"Total pooled rows: {len(pooled):,}")
    print(f"Overall churn rate: {pooled[TARGET].mean():.1%}")
    
    return pooled


# -------------------------
# Feature importance
# -------------------------
def feature_importance(model: CalibratedClassifierCV, feature_names: list) -> pd.DataFrame:
    coefs = []
    for cc in model.calibrated_classifiers_:
        clf = cc.estimator.named_steps["clf"]
        coefs.append(clf.coef_.ravel())
    coef = np.mean(np.vstack(coefs), axis=0)

    df = pd.DataFrame({"feature": feature_names, "coef": coef})
    df["abs_coef"] = df["coef"].abs()
    return df.sort_values("abs_coef", ascending=False).reset_index(drop=True)


# -------------------------
# Evaluation
# -------------------------
def eval_snapshot(model, feature_list: list, snapshot_time: pd.Timestamp) -> dict:
    ds_ = make_dataset(snapshot_time)
    y = ds_[TARGET].astype(int)

    X = make_X(ds_)
    for c in feature_list:
        if c not in X.columns:
            X[c] = 0
    X = X[feature_list]

    proba = model.predict_proba(X)[:, 1]
    thr = 0.5
    y_hat = (proba >= thr).astype(int)

    # Optimize threshold for F1
    best = {"thr": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0}
    for t in np.linspace(0.01, 0.99, 99):
        yh = (proba >= t).astype(int)
        f1v = f1_score(y, yh, zero_division=0)
        if f1v > best["f1"]:
            best["thr"] = float(t)
            best["f1"] = float(f1v)
            best["precision"] = float(precision_score(y, yh, zero_division=0))
            best["recall"] = float(recall_score(y, yh, zero_division=0))
            best["accuracy"] = float(accuracy_score(y, yh))

    return {
        "snapshot_time": str(snapshot_time),
        "rows": int(len(ds_)),
        "churn_rate": float(y.mean()),
        "threshold": float(thr),
        "accuracy": float(accuracy_score(y, y_hat)),
        "precision": float(precision_score(y, y_hat, zero_division=0)),
        "recall": float(recall_score(y, y_hat, zero_division=0)),
        "f1": float(f1_score(y, y_hat, zero_division=0)),
        "threshold_opt": float(best["thr"]),
        "accuracy_opt": float(best["accuracy"]),
        "precision_opt": float(best["precision"]),
        "recall_opt": float(best["recall"]),
        "f1_opt": float(best["f1"]),
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
    }


# -------------------------
# Main
# -------------------------
def main():
    ART_DIR.mkdir(exist_ok=True)

    max_time = get_max_time()
    min_time = get_min_time()
    snap_test = max_time - pd.Timedelta(days=CHURN_WINDOW_DAYS)

    print("=" * 60)
    print("CHURN MODEL TRAINING")
    print("=" * 60)
    print(f"Data range: {min_time} → {max_time}")
    print(f"Test snapshot: {snap_test}")
    print(f"Config: LOOKBACK={LOOKBACK_DAYS}d, CHURN_WINDOW={CHURN_WINDOW_DAYS}d")
    print(f"Pooling: step={ROLLING_STEP_DAYS}d, max_snapshots={'ALL' if MAX_POOL_SNAPSHOTS == 0 else MAX_POOL_SNAPSHOTS}")

    # Build datasets
    train_ds = build_pooled_train_dataset(snap_test)
    test_ds = make_dataset(snap_test)

    # Remove snapshot tracking column before training
    if "_snapshot_time" in train_ds.columns:
        train_ds = train_ds.drop(columns=["_snapshot_time"])

    y_train = train_ds[TARGET].astype(int)
    y_test = test_ds[TARGET].astype(int)

    X_train = make_X(train_ds)
    X_test = make_X(test_ds)

    # Align features
    for c in X_train.columns:
        if c not in X_test.columns:
            X_test[c] = 0
    X_test = X_test[X_train.columns]

    print(f"\nTraining set: {len(X_train):,} rows, {X_train.shape[1]} features")
    print(f"Test set: {len(X_test):,} rows")
    print(f"Train churn rate: {y_train.mean():.1%}")
    print(f"Test churn rate: {y_test.mean():.1%}")

    # Train model
    print("\nTraining model...")
    base = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            max_iter=5000,
            tol=1e-3,
            class_weight="balanced",
            random_state=42,
        )),
    ])
    base.fit(X_train, y_train)

    model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    model.fit(X_train, y_train)

    # Save artifacts
    joblib.dump(model, MODEL_FILE)
    FEATURES_FILE.write_text(json.dumps(list(X_train.columns), indent=2), encoding="utf-8")

    # Rolling evaluation
    print("\nRunning rolling evaluation...")
    eval_snaps = []
    for k in range(N_EVAL_SNAPSHOTS):
        s = snap_test - pd.Timedelta(days=k * ROLLING_STEP_DAYS)
        if s + pd.Timedelta(days=CHURN_WINDOW_DAYS) <= max_time:
            eval_snaps.append(s)

    rolling = pd.DataFrame([eval_snapshot(model, list(X_train.columns), s) for s in eval_snaps])
    rolling = rolling.sort_values("snapshot_time", ascending=False)
    rolling.to_csv(ROLLING_EVAL_FILE, index=False)

    # Feature importance
    imp = feature_importance(model, list(X_train.columns))
    imp.to_csv(IMPORTANCE_FILE, index=False)

    # Metadata
    meta = {
        "max_time": str(max_time),
        "min_time": str(min_time),
        "snap_test": str(snap_test),
        "train_rows": int(len(train_ds)),
        "test_rows": int(len(test_ds)),
        "train_churn_rate": float(y_train.mean()),
        "test_churn_rate": float(y_test.mean()),
        "n_features": int(X_train.shape[1]),
        "n_train_snapshots": int(len([p for p in train_ds.columns if p != "_snapshot_time"]) if "_snapshot_time" not in train_ds.columns else 0),
        "pooling_config": {
            "rolling_step_days": ROLLING_STEP_DAYS,
            "max_pool_snapshots": MAX_POOL_SNAPSHOTS,
            "lookback_days": LOOKBACK_DAYS,
            "churn_window_days": CHURN_WINDOW_DAYS,
        },
        "using_partitioned_dataset": bool(ACTIVITY_DS_DIR.exists()),
    }
    META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Final output
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Saved model: {MODEL_FILE}")
    print(f"Saved features: {FEATURES_FILE}")
    print(f"Saved meta: {META_FILE}")
    print(f"Saved eval: {ROLLING_EVAL_FILE}")
    print(f"Saved importance: {IMPORTANCE_FILE}")
    print("\nRolling evaluation:")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(rolling[["snapshot_time", "rows", "churn_rate", "roc_auc", "pr_auc", "f1_opt"]].to_string(index=False))


if __name__ == "__main__":
    main()