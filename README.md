# Customer Churn Prediction Pipeline

A production-ready machine learning pipeline for predicting customer churn based on engagement events.

## Overview

This pipeline processes customer interaction data, engineers behavioral features, trains a churn prediction model, and scores all customers with churn probabilities.

### Key Features

- **Rolling snapshot training**: Uses 32 historical snapshots for robust model training
- **Full customer coverage**: Scores all 200,000 customers including dormant ones
- **Interpretable buckets**: Customers segmented into actionable risk tiers
- **Centralized configuration**: All settings in one place (`src/churn/config.py`)

### Model Performance

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.891 |
| PR-AUC | 0.870 |
| F1 (optimized) | 0.848 |

## Quick Start

### Prerequisites
```bash
python >= 3.9
```

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd churn_2

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Run the Pipeline
```bash
# Run full pipeline (all 5 steps)
python scripts/run_pipeline.py

# Or run individual steps
python scripts/clean.py
python scripts/activity.py
python scripts/partition_activity.py
python scripts/train.py
python scripts/score.py
```

## Pipeline Steps
```
┌─────────────────┐
│ 3_years_churn.csv │  Raw input (21.7M events)
└────────┬────────┘
         ▼
┌─────────────────┐
│   clean.py      │  Parse timestamps, normalize text
└────────┬────────┘
         ▼
┌─────────────────┐
│  activity.py    │  Filter to engagement events (20.4M)
└────────┬────────┘
         ▼
┌─────────────────┐
│ partition_activity.py │  Partition by month (36 partitions)
└────────┬────────┘
         ▼
┌─────────────────┐
│   train.py      │  Train model (32 snapshots, 1.9M rows)
└────────┬────────┘
         ▼
┌─────────────────┐
│   score.py      │  Score all customers (200,000)
└────────┬────────┘
         ▼
┌─────────────────┐
│ customer_scores.parquet │  Final output
└─────────────────┘
```

## Project Structure
```
churn_2/
├── src/churn/
│   ├── config.py        # All configuration settings
│   ├── features.py      # Feature engineering logic
│   ├── cleaning.py      # Data cleaning utilities
│   └── __init__.py
├── scripts/
│   ├── run_pipeline.py  # Pipeline orchestrator
│   ├── clean.py         # Step 1: Clean raw data
│   ├── activity.py      # Step 2: Filter activity events
│   ├── partition_activity.py  # Step 3: Partition by month
│   ├── train.py         # Step 4: Train model
│   └── score.py         # Step 5: Score customers
├── artifacts/           # Model and metadata
│   ├── churn_model.joblib
│   ├── feature_list.json
│   ├── train_meta.json
│   ├── rolling_eval.csv
│   ├── feature_importance.csv
│   └── bucket_thresholds.json
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Configuration

All settings are centralized in `src/churn/config.py`:

### Churn Definition
```python
CHURN_WINDOW_DAYS = 30   # No activity in 30 days = churned
LOOKBACK_DAYS = 90       # Feature window (90 days of history)
```

### Scoring Thresholds
```python
THRESH_VERY_HIGH = 0.8   # >= 80% probability
THRESH_HIGH = 0.6        # >= 60% probability
THRESH_MEDIUM = 0.4      # >= 40% probability
```

### Activity Events
Events that count as customer engagement:
- `emarsys_open` - Email opens
- `emarsys_click` - Email clicks
- `order` - Purchases
- `emarsys_sessions_*` - Website sessions

## Output

### Customer Scores (`customer_scores.parquet`)

| Column | Description |
|--------|-------------|
| `external_customerkey` | Customer ID |
| `snapshot_time` | Scoring timestamp |
| `churn_probability` | Model probability (0-1) |
| `churn_bucket` | Risk tier (see below) |
| `churn_action` | 1 = needs retention action |

### Risk Buckets

| Bucket | Probability | Count | Action |
|--------|-------------|-------|--------|
| `very_high` | >= 0.8 | 135,937 (68%) | High priority retention |
| `high` | >= 0.6 | 16,468 (8%) | Retention campaigns |
| `medium` | >= 0.4 | 8,000 (4%) | Monitor |
| `low` | < 0.4 | 27,518 (14%) | Healthy customers |
| `weak_signal` | N/A | 12,077 (6%) | No engagement data |

### Artifacts

| File | Description |
|------|-------------|
| `churn_model.joblib` | Trained scikit-learn model |
| `feature_list.json` | List of 49 features used |
| `train_meta.json` | Training metadata and stats |
| `rolling_eval.csv` | 12-month evaluation metrics |
| `feature_importance.csv` | Feature coefficients |
| `bucket_thresholds.json` | Scoring thresholds used |

## Features

The model uses 49 engineered features:

### Top Predictive Features
1. `active_days_last_30d` - Days with activity in last 30 days
2. `recency_days` - Days since last activity
3. `cnt_emarsys_open_last_90d` - Email opens in last 90 days
4. `freq_trend_30_60` - Activity trend (30d vs 60d)

### Feature Categories
- **Recency**: Days since last activity
- **Frequency**: Event counts (30/60/90 day windows)
- **Engagement**: Email opens, clicks, sessions
- **Trends**: Activity change over time
- **Ratios**: Session share, email share, click-to-open

## Model Details

### Algorithm
- **Base model**: SGDClassifier (logistic regression)
- **Calibration**: CalibratedClassifierCV (sigmoid)
- **Scaling**: StandardScaler

### Training Strategy
- **Pooled rolling snapshots**: 32 monthly snapshots
- **Total training rows**: 1,946,215
- **Features**: 49 (after log transform and filtering)

### Evaluation
Rolling evaluation across 12 months shows stable performance:
- ROC-AUC range: 0.867 - 0.895
- Model generalizes well across seasons

## Usage Examples

### Load Scores in Python
```python
import pandas as pd

scores = pd.read_parquet('customer_scores.parquet')

# High-risk customers for retention
high_risk = scores[scores['churn_bucket'].isin(['very_high', 'high'])]
print(f"Customers needing attention: {len(high_risk):,}")

# Recently active but at risk
active_risk = scores[
    (scores['churn_bucket'] == 'very_high') & 
    (scores['churn_probability'] < 1.0)
]
```

### Change Churn Definition
Edit `src/churn/config.py`:
```python
CHURN_WINDOW_DAYS = 60  # Change to 60-day window
```
Then re-run: `python scripts/run_pipeline.py`

### Adjust Risk Thresholds
Edit `src/churn/config.py`:
```python
THRESH_VERY_HIGH = 0.75  # Lower threshold
THRESH_HIGH = 0.5
```
Then re-run: `python scripts/score.py`

## Troubleshooting

### Missing Data Error
```
FileNotFoundError: Missing activity_events.parquet
```
**Solution**: Run the full pipeline from the beginning:
```bash
python scripts/run_pipeline.py
```

### Memory Issues
For large datasets, the partitioned dataset (`activity_events_ds/`) enables efficient loading by month. The pipeline automatically uses partitions when available.

### Model Not Found
```
FileNotFoundError: Missing model artifacts
```
**Solution**: Run training first:
```bash
python scripts/train.py
```

## Contributing

1. All configuration changes go in `src/churn/config.py`
2. Feature engineering logic is in `src/churn/features.py`
3. Run full pipeline after changes to verify

## License

Internal use only.
