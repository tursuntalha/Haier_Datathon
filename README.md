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

**Project Vision:** Demand forecasting at scale. The lifecycle-aware feature engineering (EOL urgency, life_progress, months_until_eol) and rWMAPE metric optimization developed here are directly applicable to supply chain and inventory management systems. Any business managing products with defined lifecycles — electronics, pharmaceuticals, fashion — faces exactly this problem. The business rules layer (post-prediction adjustments for EOL products) demonstrates how domain knowledge integrates with ML in real industrial settings.

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
│   ├── train.csv                # Historical sales data
│   ├── product.csv              # Product master (lifecycle dates, markets)
│   └── submission.csv           # Submission template
├── feature.py                   # Data cleaning & feature engineering
├── model.py                     # Model training & prediction pipeline
├── utils.py                     # rWMAPE metric, CV evaluation, submission formatter
├── main.ipynb                   # End-to-end notebook
├── requirements.txt
│
├── anomaly/
│   ├── __init__.py
│   └── anomaly_detection.py     # Item 3: 3σ outlier & zero-sales streak detection
│
├── probabilistic/
│   ├── __init__.py
│   └── quantile_forecast.py     # Item 2: LightGBM quantile regression (p10/p50/p90)
│
├── experiments/
│   ├── __init__.py
│   ├── tft_model.py             # Item 1: Temporal Fusion Transformer
│   └── patch_tst_model.py       # Item 6: PatchTST architecture
│
├── api/
│   ├── __init__.py
│   └── forecast_api.py          # Item 5: FastAPI service
│
└── dashboard/
    ├── __init__.py
    └── dashboard.py             # Item 4: Streamlit dashboard
```

---

## Setup & Run

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

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

## Beyond the Datathon

A 0.96+ score demonstrates that the core pipeline is solid. The next evolution takes this from a competition notebook to a production-grade forecasting system:

- [x] **Temporal Fusion Transformer (TFT)** — Replace LightGBM with TFT (PyTorch Forecasting library) for a true sequence-aware model. TFT handles multi-horizon forecasting natively and provides interpretable attention weights. Compare against LightGBM on the same CV folds.  [`experiments/tft_model.py`](experiments/tft_model.py)
- [x] **Probabilistic Forecasting** — Switch from point estimates to prediction intervals using LightGBM quantile regression (train separate models for p10, p50, p90). More useful for inventory decisions: "We're 90% confident sales will be between X and Y units."  [`probabilistic/quantile_forecast.py`](probabilistic/quantile_forecast.py)
- [x] **Anomaly Detection Layer** — Add a pre-processing step that flags statistically anomalous sales months (3σ outliers, zero-sales streaks) and treats them differently during training. Prevents model from learning from stockout or data entry errors.  [`anomaly/anomaly_detection.py`](anomaly/anomaly_detection.py)
- [x] **Interactive Forecasting Dashboard** — Build a Streamlit or Grafana dashboard that visualizes: actual vs predicted sales per product, forecast confidence intervals, lifecycle stage indicators, and alert flags for products approaching EOL.  [`dashboard/dashboard.py`](dashboard/dashboard.py)
- [x] **REST API Service** — Wrap the trained model as a FastAPI service. Input: product ID + target month. Output: predicted demand + confidence interval + EOL flag. Simulate integration with a real ERP system by calling the API from a mock order management system.  [`api/forecast_api.py`](api/forecast_api.py)
- [x] **PatchTST Experiment** — Test PatchTST (Patch Time Series Transformer), a recent architecture that treats time series like NLP tokens. Compare to both LightGBM and TFT on this dataset.  [`experiments/patch_tst_model.py`](experiments/patch_tst_model.py)

> Data is sourced from the Kaggle Haier Datathon competition. Copyright belongs to Haier. Commercial use requires explicit permission.
