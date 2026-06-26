# Haier Datathon — Demand Forecasting

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LightGBM-02569B?style=for-the-badge&logo=lightgbm&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
</p>

> **Public Leaderboard: `0.96214` &nbsp;|&nbsp; Private Leaderboard: `0.95819`**

LightGBM-based demand forecasting solution for the **Haier Datathon** competition on Kaggle. Predicts monthly product sales quantities across markets by leveraging product lifecycle features, lag statistics, and rolling window aggregations.

---

## Competition Overview

| Field | Detail |
|---|---|
| Competition | Haier Datathon (Kaggle) |
| Task | Time-series demand forecasting |
| Metric | rWMAPE → Score = `1 / (1 + rWMAPE)` |
| Public Score | **0.96214** |
| Private Score | **0.95819** |

### Metric: rWMAPE

```
rWMAPE = (Σ|actual - predicted| + λ · |Σactual - Σpredicted|) / (Σactual + γ · Σpredicted)

Final Score = 1 / (1 + rWMAPE)   ← higher is better, max = 1.0
```

---

## Pipeline Architecture

```
train.csv + product.csv
        │
        ▼
┌──────────────────────────┐
│   Data Cleaning          │  fill_missing_values_no_drop()
│   Grid Expansion         │  expand_grid_train_only()
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│   Feature Engineering    │
│   ─ Lag features         │
│   ─ Rolling features     │
│   ─ Lifecycle features   │
│   ─ Time features        │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│   LightGBM Training      │  Cross-validation + early stopping
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│   Post-processing Rules  │  EOL, pre-production, near-EOL
└──────────┬───────────────┘
           │
           ▼
      submission.csv
```

---

## Feature Engineering

| Feature | Type | Description |
|---|---|---|
| `lag_1`, `lag_2`, `lag_3` | Lag | Sales 1–3 months ago |
| `lag_6`, `lag_12` | Lag | Sales 6 and 12 months ago |
| `rolling_mean_3/6/12` | Rolling | Moving average over 3/6/12 months |
| `rolling_std_3/6/12` | Rolling | Rolling standard deviation |
| `eol_urgency` | Lifecycle | Urgency score as product approaches end-of-life |
| `life_progress` | Lifecycle | Product lifecycle completion ratio (0–1) |
| `months_until_eol` | Lifecycle | Months remaining until EOL |
| `flag_eol_passed` | Lifecycle | Binary flag: EOL date has passed |
| `month`, `quarter` | Time | Calendar month and quarter |
| `month_sin`, `month_cos` | Time | Cyclical encoding for seasonality |

### Business Rules Applied After Prediction

| Condition | Adjustment |
|---|---|
| Product EOL already passed | Reduce predicted sales by 90% |
| Less than 3 months until EOL | Reduce predicted sales by 50% |
| Product not yet in production | Set prediction to 0 |

---

## Model Details

```python
LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8
)
```

---

## Project Structure

```
Haier_Datathon/
├── data/
│   ├── train.csv          # Historical sales data
│   ├── product.csv        # Product master (lifecycle dates, markets)
│   └── submission.csv     # Submission template
├── feature.py             # Data cleaning & feature engineering
├── model.py               # Model training & prediction pipeline
├── utils.py               # rWMAPE metric, CV evaluation, submission formatter
├── main.ipynb             # End-to-end notebook
└── requirements.txt
```

---

## Setup & Run

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

Download data from the Kaggle competition page and place `train.csv`, `product.csv`, and `submission.csv` inside the `data/` directory. Then run `main.ipynb` to generate predictions.

---

## Key Modules

### `feature.py`
- `fill_missing_values_no_drop()` — imputes gaps without removing rows
- `expand_grid_train_only()` — expands sparse time series to a full monthly grid
- `add_advanced_lifecycle_features()` — computes EOL proximity and lifecycle stage

### `model.py`
- `prepare_features()` — assembles all lag, rolling, lifecycle, and time features
- `train_model()` — trains LightGBM with early stopping
- `train_and_predict_full()` — full end-to-end pipeline

### `utils.py`
- `calculate_rwmape()` — official competition metric
- `evaluate_model_cv()` — cross-validation scorer
- `prepare_submission_format()` — converts predictions to submission format

---

> Data is sourced from the Kaggle Haier Datathon competition. Copyright belongs to Haier. Commercial use requires explicit permission.
