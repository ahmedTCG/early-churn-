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
from sklearn.metrics import roc_auc_score, average_precision_score

from churn.features import build_customer_features

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*encountered in matmul.*")
np.seterr(over="ignore", divide="ignore", invalid="ignore")

# -------------------------
# Config (keep minimal)
# -------------------------
ID_COL = "external_customerkey"
TARGET = "churn_30d"

CHURN_WINDOW_DAYS = 30
LOOKBACK_DAYS = 90

ROLLING_STEP_DAYS = 30
N_POOL_SNAPSHOTS = 4       # training snapshots: T-30, T-60, ...
N_EVAL_SNAPSHOTS = 4       # eval snapshots: T, T-30, ...

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIVITY_FILE = PROJECT_ROOT / "activity_events.parquet"
ACTIVITY_DS_DIR = PROJECT_ROOT / "activity_events_ds"

ART_DIR = PROJECT_ROOT / "artifacts"
MODEL_FILE = ART_DIR / "churn_model.joblib"
FEATURES_FILE = ART_DIR / "feature_list.json"
META_FILE = ART_DIR / "train_meta.json"
ROLLING_EVAL_FILE = ART_DIR / "rolling_eval.csv"
IMPORTANCE_FILE = ART_DIR / "feature_importance.csv"


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
            partitioning=ds.partitioning(field_names=["event_month"]),  # directory partition
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


def make_dataset(snapshot_time: pd.Timestamp) -> pd.DataFrame:
    """
    Features: (T-90d, T]
    Label   : no activity in (T, T+30d]
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

    # drop constant features
    X = X.loc[:, X.std(axis=0) != 0]
    return X


def pooled_train_dataset(test_snapshot: pd.Timestamp) -> pd.DataFrame:
    parts = []
    for k in range(1, N_POOL_SNAPSHOTS + 1):
        snap = test_snapshot - pd.Timedelta(days=k * ROLLING_STEP_DAYS)
        parts.append(make_dataset(snap))
    return pd.concat(parts, ignore_index=True)


def feature_importance(model: CalibratedClassifierCV, feature_names: list) -> pd.DataFrame:
    coefs = []
    for cc in model.calibrated_classifiers_:
        clf = cc.estimator.named_steps["clf"]
        coefs.append(clf.coef_.ravel())
    coef = np.mean(np.vstack(coefs), axis=0)

    df = pd.DataFrame({"feature": feature_names, "coef": coef})
    df["abs_coef"] = df["coef"].abs()
    return df.sort_values("abs_coef", ascending=False).reset_index(drop=True)


def eval_snapshot(model, feature_list: list, snapshot_time: pd.Timestamp) -> dict:
    ds_ = make_dataset(snapshot_time)
    y = ds_[TARGET].astype(int)

    X = make_X(ds_)
    for c in feature_list:
        if c not in X.columns:
            X[c] = 0
    X = X[feature_list]

    proba = model.predict_proba(X)[:, 1]
    return {
        "snapshot_time": str(snapshot_time),
        "rows": int(len(ds_)),
        "churn_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
    }


def main():
    ART_DIR.mkdir(exist_ok=True)

    max_time = get_max_time()
    snap_test = max_time - pd.Timedelta(days=CHURN_WINDOW_DAYS)

    train_ds = pooled_train_dataset(snap_test)
    test_ds = make_dataset(snap_test)

    y_train = train_ds[TARGET].astype(int)
    y_test = test_ds[TARGET].astype(int)

    X_train = make_X(train_ds)
    X_test = make_X(test_ds)

    for c in X_train.columns:
        if c not in X_test.columns:
            X_test[c] = 0
    X_test = X_test[X_train.columns]

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

    # Save model + features
    joblib.dump(model, MODEL_FILE)
    FEATURES_FILE.write_text(json.dumps(list(X_train.columns), indent=2), encoding="utf-8")

    # Rolling eval
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

    # Meta
    meta = {
        "max_time": str(max_time),
        "snap_test": str(snap_test),
        "train_rows": int(len(train_ds)),
        "test_rows": int(len(test_ds)),
        "train_churn_rate": float(y_train.mean()),
        "test_churn_rate": float(y_test.mean()),
        "n_features": int(X_train.shape[1]),
        "using_partitioned_dataset": bool(ACTIVITY_DS_DIR.exists()),
    }
    META_FILE.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # One clean output
    print("Saved model :", MODEL_FILE)
    print("Saved feats :", FEATURES_FILE)
    print("Saved meta  :", META_FILE)
    print("Saved eval  :", ROLLING_EVAL_FILE)
    print("Saved imp   :", IMPORTANCE_FILE)
    print("\nRolling eval:")
    print(rolling)


if __name__ == "__main__":
    main()
