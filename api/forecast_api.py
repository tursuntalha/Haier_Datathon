import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

app = FastAPI(
    title="Haier Demand Forecasting API",
    description="REST API for product demand forecasting with lifecycle awareness",
    version="1.0.0",
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'lgb_model.pkl')
FEATURES_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'feature_cols.pkl')
_product_df = None
_model = None
_feature_cols = None


class ForecastRequest(BaseModel):
    product_code: str = Field(..., description="Product identifier (e.g., PRD_0010)")
    market: str = Field(..., description="Market identifier (e.g., MKT_001)")
    target_month: str = Field(..., description="Target month in YYYY-MM format")
    end_production_date: Optional[str] = Field(None, description="EOL date in YYYY-MM-DD format")
    category: Optional[str] = Field("Unknown", description="Product category")
    brand: Optional[str] = Field("Unknown", description="Product brand")
    sector: Optional[str] = Field("Unknown", description="Product sector")


class ForecastResponse(BaseModel):
    product_code: str
    market: str
    target_month: str
    predicted_demand: float
    confidence_interval: dict
    eol_flag: bool
    eol_urgency: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    products_loaded: int


def load_resources():
    global _model, _feature_cols, _product_df
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    product_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'product.csv')

    if os.path.exists(os.path.join(model_dir, 'lgb_model.pkl')):
        with open(os.path.join(model_dir, 'lgb_model.pkl'), 'rb') as f:
            _model = pickle.load(f)
    if os.path.exists(os.path.join(model_dir, 'feature_cols.pkl')):
        with open(os.path.join(model_dir, 'feature_cols.pkl'), 'rb') as f:
            _feature_cols = pickle.load(f)
    if os.path.exists(product_path):
        _product_df = pd.read_csv(product_path)
        for col in ['start_production_date', 'end_production_date']:
            if col in _product_df.columns:
                _product_df[col] = pd.to_datetime(_product_df[col], errors='coerce')


@app.on_event("startup")
async def startup():
    load_resources()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=_model is not None,
        products_loaded=len(_product_df) if _product_df is not None else 0,
    )


def _compute_lifecycle_features(target_date, end_production_date):
    if end_production_date is None or pd.isna(end_production_date):
        return {
            'months_until_eol': 999,
            'eol_urgency': 0.0,
            'flag_eol_passed': 0,
            'life_progress': 0.0,
        }

    months_until = (
        (end_production_date.year - target_date.year) * 12 +
        (end_production_date.month - target_date.month)
    )
    months_until = max(months_until, 0)
    eol_passed = 1 if target_date > end_production_date else 0
    eol_urgency = 1.0 / (months_until + 1)

    return {
        'months_until_eol': months_until,
        'eol_urgency': eol_urgency,
        'flag_eol_passed': eol_passed,
        'life_progress': 0.0,
    }


def _build_feature_vector(request: ForecastRequest):
    target_date = pd.to_datetime(request.target_month + '-01')

    eol_date = None
    if request.end_production_date:
        eol_date = pd.to_datetime(request.end_production_date)

    lifecycle = _compute_lifecycle_features(target_date, eol_date)

    month = target_date.month
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    quarter = (month - 1) // 3 + 1

    cat_map = {}
    if _product_df is not None:
        for col in ['market', 'category', 'brand', 'sector']:
            if col in _product_df.columns:
                codes = _product_df[col].astype('category').cat.codes
                cat_map[f'{col}_encoded'] = dict(zip(_product_df[col], codes))

    features = {
        'lag_1': 0.0, 'lag_2': 0.0, 'lag_3': 0.0, 'lag_6': 0.0, 'lag_12': 0.0,
        'rolling_mean_3': 0.0, 'rolling_mean_6': 0.0, 'rolling_mean_12': 0.0,
        'rolling_std_3': 0.0, 'rolling_std_6': 0.0, 'rolling_std_12': 0.0,
        'month': month, 'quarter': quarter, 'month_sin': month_sin, 'month_cos': month_cos,
        'eol_urgency': lifecycle['eol_urgency'],
        'life_progress': lifecycle['life_progress'],
        'months_until_eol': lifecycle['months_until_eol'],
        'flag_eol_passed': lifecycle['flag_eol_passed'],
        'market_encoded': cat_map.get('market_encoded', {}).get(request.market, 0),
        'category_encoded': cat_map.get('category_encoded', {}).get(request.category, 0),
        'brand_encoded': cat_map.get('brand_encoded', {}).get(request.brand, 0),
        'sector_encoded': cat_map.get('sector_encoded', {}).get(request.sector, 0),
    }

    return features, lifecycle


@app.post("/predict", response_model=ForecastResponse)
async def predict(request: ForecastRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")

    features, lifecycle = _build_feature_vector(request)

    if _feature_cols is not None:
        X = np.array([[features.get(col, 0.0) for col in _feature_cols]])
    else:
        X = np.array([list(features.values())])

    prediction = float(_model.predict(X)[0])

    if lifecycle['flag_eol_passed']:
        prediction *= 0.1
    elif lifecycle['months_until_eol'] < 3:
        prediction *= 0.5

    prediction = max(prediction, 0)

    ci_width = prediction * 0.3
    confidence_interval = {
        "p10": round(max(prediction - ci_width, 0), 2),
        "p50": round(prediction, 2),
        "p90": round(prediction + ci_width, 2),
    }

    return ForecastResponse(
        product_code=request.product_code,
        market=request.market,
        target_month=request.target_month,
        predicted_demand=round(prediction, 2),
        confidence_interval=confidence_interval,
        eol_flag=bool(lifecycle['flag_eol_passed']),
        eol_urgency=round(lifecycle['eol_urgency'], 4),
        model_version="lightgbm-v1",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
