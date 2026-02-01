# Customer Churn Prediction Pipeline

ML pipeline for predicting customer churn based on engagement events.

## Model Performance

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.891 |
| PR-AUC | 0.870 |
| F1 | 0.848 |

## Quick Start
```bash
# Install
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Run pipeline
python scripts/run_pipeline.py
```

## Pipeline
```
3_years_churn.csv → clean.py → activity.py → partition_activity.py → train.py → score.py → customer_scores.parquet
```

## Data Source

Extract from Redshift (see `data/extract_customers.sql`):
```sql
SELECT external_customerkey, event_time, interaction_type
FROM poc_dw.customer_interactions_fact
WHERE event_time >= DATEADD(month, -36, CURRENT_DATE)
  AND incoming_outgoing = 'incoming'
```

## Project Structure
```
├── data/extract_customers.sql   # Redshift query
├── src/churn/
│   ├── config.py                # All settings
│   └── features.py              # Feature engineering
├── scripts/                     # Pipeline steps
├── artifacts/                   # Model & metadata
└── customer_scores.parquet      # Output
```

## Configuration

Edit `src/churn/config.py`:
```python
CHURN_WINDOW_DAYS = 30   # No activity = churned
LOOKBACK_DAYS = 90       # Feature window
THRESH_VERY_HIGH = 0.8   # Bucket thresholds
THRESH_HIGH = 0.6
THRESH_MEDIUM = 0.4
```

## Output

`customer_scores.parquet`:

| Column | Description |
|--------|-------------|
| `external_customerkey` | Customer ID |
| `churn_probability` | 0-1 score |
| `churn_bucket` | very_high / high / medium / low / weak_signal |
| `churn_action` | 1 = needs retention |

## Risk Buckets

| Bucket | Probability | Action |
|--------|-------------|--------|
| very_high | >= 0.8 | Priority retention |
| high | >= 0.6 | Retention campaigns |
| medium | >= 0.4 | Monitor |
| low | < 0.4 | Healthy |
| weak_signal | N/A | No engagement data |
