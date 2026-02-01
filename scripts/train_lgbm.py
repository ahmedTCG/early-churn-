"""
LightGBM Churn Model Training
Alternative to train.py (SGDClassifier)
"""
import json
import warnings

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import pyarrow.dataset as ds

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from churn.config import (
    ACTIVITY_FILE,
    ACTIVITY_DS_DIR,
    ARTIFACTS_DIR,
    ID_COL,
    TIME_COL,
    EVENT_COL,
    TARGET,
    CHURN_WINDOW_DAYS,
    LOOKBACK_DAYS,
    ROLLING_STEP_DAYS,
    MAX_POOL_SNAPSHOTS,
    N_EVAL_SNAPSHOTS,
    MAX_RECENCY_DAYS,
)
from churn.features import build_customer_features

warnings.filterwarnings("ignore")

# Output files (separate from original model)
MODEL_FILE_LGBM = ARTIFACTS_DIR / "churn_model_lgbm.joblib"
FEATURES_FILE_LGBM = ARTIFACTS_DIR / "feature_list_lgbm.json"
META_FILE_LGBM = ARTIFACTS_DIR / "train_meta_lgbm.json"
EVAL_FILE_LGBM = ARTIFACTS_DIR / "rolling_eval_lgbm.csv"
IMPORTANCE_FILE_LGBM = ARTIFACTS_DIR / "feature_importance_lgbm.csv"

# LightGBM parameters
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "max_depth": 6,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 100,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "n_estimators": 500,
    "early_stopping_rounds": 50,
    "verbose": -1,
    "random_state": 42,
    "class_weight": "balanced",
}


def _months_in_range(start: pd.Timestamp, end: pd.Timestamp) -> list:
    start_m = pd.Period(start.to_pydatetime(), freq="M")
    end_m = pd.Period(end.to_pydatetime(), freq="M")
    return [p.strftime("%Y-%m") for p in pd.period_range(start_m, end_m, freq="M")]


def load_events(window_start: pd.Timestamp, window_end: pd.Timestamp) -> pd.DataFrame:
    cols = [ID_COL, TIME_COL, EVENT_COL]

    if ACTIVITY_DS_DIR.exists():
        months = _months_in_range(window_start, window_end)
        dataset = ds.dataset(
            str(ACTIVITY_DS_DIR),
            format="parquet",
            partitioning=ds.partitioning(field_names=["event_month"]),
        )
        table = dataset.to_table(columns=cols, filter=ds.field("event_month").isin(months))
        df = table.to_pandas()
    else:
        if not ACTIVITY_FILE.exists():
            raise FileNotFoundError("Missing activity data. Run steps 1→3.")
        df = pd.read_parquet(ACTIVITY_FILE, columns=cols)

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)
    df = df.dropna(subset=[ID_COL, TIME_COL, EVENT_COL]).copy()
    df = df[(df[TIME_COL] > window_start) & (df[TIME_COL] <= window_end)].copy()

    df[ID_COL] = df[ID_COL].astype("string").str.strip().astype("category")
    df[EVENT_COL] = df[EVENT_COL].astype("string").str.strip().str.lower().astype("category")

    return df


def get_max_time() -> pd.Timestamp:
    if ACTIVITY_FILE.exists():
        s = pd.to_datetime(pd.read_parquet(ACTIVITY_FILE, columns=[TIME_COL])[TIME_COL], errors="coerce", utc=True)
        return s.max()
    raise FileNotFoundError("No activity data found.")


def get_min_time() -> pd.Timestamp:
    if ACTIVITY_FILE.exists():
        s = pd.to_datetime(pd.read_parquet(ACTIVITY_FILE, columns=[TIME_COL])[TIME_COL], errors="coerce", utc=True)
        return s.min()
    raise FileNotFoundError("No activity data found.")


def make_dataset(snapshot_time: pd.Timestamp) -> pd.DataFrame:
    window_start = snapshot_time - pd.Timedelta(days=LOOKBACK_DAYS)
    window_end = snapshot_time + pd.Timedelta(days=CHURN_WINDOW_DAYS)

    ev = load_events(window_start, window_end)

    hist = ev[ev[TIME_COL] <= snapshot_time].copy()
    customers = hist[[ID_COL]].drop_duplicates()

    future = ev[(ev[TIME_COL] > snapshot_time) & (ev[TIME_COL] <= window_end)]
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
        X["recency_days"] = X["recency_days"].clip(upper=MAX_RECENCY_DAYS)

    return X


def build_pooled_train_dataset(test_snapshot: pd.Timestamp) -> pd.DataFrame:
    min_time = get_min_time()
    earliest_allowed = min_time + pd.Timedelta(days=LOOKBACK_DAYS)

    parts = []
    k = 1

    while True:
        snap = test_snapshot - pd.Timedelta(days=k * ROLLING_STEP_DAYS)

        if snap < earliest_allowed:
            break

        if MAX_POOL_SNAPSHOTS > 0 and len(parts) >= MAX_POOL_SNAPSHOTS:
            break

        ds_ = make_dataset(snap)
        parts.append(ds_)
        k += 1

    if not parts:
        raise ValueError("No snapshots available for pooled training.")

    print(f"Pooled training: {len(parts)} snapshots")
    return pd.concat(parts, ignore_index=True)


def feature_importance(model: lgb.LGBMClassifier, feature_names: list) -> pd.DataFrame:
    imp = model.feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": imp})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df


def eval_snapshot(model, feature_list: list, snapshot_time: pd.Timestamp) -> dict:
    ds_ = make_dataset(snapshot_time)
    y = ds_[TARGET].astype(int)

    X = make_X(ds_)
    for c in feature_list:
        if c not in X.columns:
            X[c] = 0
    X = X[feature_list]

    proba = model.predict_proba(X)[:, 1]
    y_hat = (proba >= 0.5).astype(int)

    best = {"thr": 0.5, "f1": -1.0}
    for t in np.linspace(0.01, 0.99, 99):
        yh = (proba >= t).astype(int)
        f1v = f1_score(y, yh, zero_division=0)
        if f1v > best["f1"]:
            best["thr"] = float(t)
            best["f1"] = float(f1v)

    return {
        "snapshot_time": str(snapshot_time),
        "rows": int(len(ds_)),
        "churn_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "f1_opt": float(best["f1"]),
        "threshold_opt": float(best["thr"]),
    }


def main():
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    max_time = get_max_time()
    min_time = get_min_time()
    snap_test = max_time - pd.Timedelta(days=CHURN_WINDOW_DAYS)

    print("=" * 60)
    print("LIGHTGBM CHURN MODEL TRAINING")
    print("=" * 60)
    print(f"Data range: {min_time} → {max_time}")
    print(f"Test snapshot: {snap_test}")

    train_ds = build_pooled_train_dataset(snap_test)
    test_ds = make_dataset(snap_test)

    y_train = train_ds[TARGET].astype(int)
    y_test = test_ds[TARGET].astype(int)

    X_train = make_X(train_ds)
    X_test = make_X(test_ds)

    for c in X_train.columns:
        if c not in X_test.columns:
            X_test[c] = 0
    X_test = X_test[X_train.columns]

    print(f"\nTraining set: {len(X_train):,} rows, {X_train.shape[1]} features")
    print(f"Test set: {len(X_test):,} rows")
    print(f"Train churn rate: {y_train.mean():.1%}")
    print(f"Test churn rate: {y_test.mean():.1%}")

    print("\nTraining LightGBM model...")
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
    )

    proba_test = model.predict_proba(X_test)[:, 1]
    roc_test = roc_auc_score(y_test, proba_test)
    pr_test = average_precision_score(y_test, proba_test)

    print(f"\nTest ROC-AUC: {roc_test:.4f}")
    print(f"Test PR-AUC:  {pr_test:.4f}")

    joblib.dump(model, MODEL_FILE_LGBM)
    FEATURES_FILE_LGBM.write_text(json.dumps(list(X_train.columns), indent=2), encoding="utf-8")

    print("\nRunning rolling evaluation...")
    eval_snaps = []
    for k in range(N_EVAL_SNAPSHOTS):
        s = snap_test - pd.Timedelta(days=k * ROLLING_STEP_DAYS)
        if s + pd.Timedelta(days=CHURN_WINDOW_DAYS) <= max_time:
            eval_snaps.append(s)

    rolling = pd.DataFrame([eval_snapshot(model, list(X_train.columns), s) for s in eval_snaps])
    rolling = rolling.sort_values("snapshot_time", ascending=False)
    rolling.to_csv(EVAL_FILE_LGBM, index=False)

    imp = feature_importance(model, list(X_train.columns))
    imp.to_csv(IMPORTANCE_FILE_LGBM, index=False)

    meta = {
        "model": "LightGBM",
        "max_time": str(max_time),
        "min_time": str(min_time),
        "snap_test": str(snap_test),
        "train_rows": int(len(train_ds)),
        "test_rows": int(len(test_ds)),
        "train_churn_rate": float(y_train.mean()),
        "test_churn_rate": float(y_test.mean()),
        "n_features": int(X_train.shape[1]),
        "roc_auc_test": float(roc_test),
        "pr_auc_test": float(pr_test),
        "lgbm_params": {k: v for k, v in LGBM_PARAMS.items() if k != "verbose"},
    }
    META_FILE_LGBM.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Saved model: {MODEL_FILE_LGBM}")
    print(f"Saved features: {FEATURES_FILE_LGBM}")
    print(f"Saved meta: {META_FILE_LGBM}")
    print(f"Saved eval: {EVAL_FILE_LGBM}")
    print(f"Saved importance: {IMPORTANCE_FILE_LGBM}")

    print("\nRolling evaluation:")
    print(rolling[["snapshot_time", "rows", "churn_rate", "roc_auc", "pr_auc", "f1_opt"]].to_string(index=False))

    print("\nTop 10 features:")
    print(imp.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
