import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


def train_quantile_models(train_prepared, feature_cols, quantiles=[0.1, 0.5, 0.9]):
    print("\n" + "=" * 60)
    print("Probabilistic Forecasting with LightGBM Quantile Regression")
    print("=" * 60)

    X_train = train_prepared[feature_cols].fillna(0)
    y_train = train_prepared['quantity'].clip(lower=0)

    models = {}
    for q in quantiles:
        alpha = q
        print(f"Training quantile model for p{int(q * 100)} (alpha={alpha})...")

        model = lgb.LGBMRegressor(
            objective='quantile',
            alpha=alpha,
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
            force_col_wise=True,
        )
        model.fit(X_train, y_train)
        models[f'p{int(q * 100)}'] = model
        print(f"  ✅ p{int(q * 100)} model trained.")

    print("All quantile models trained successfully.")
    print("=" * 60)

    return models


def predict_quantile(models, X_pred, quantile_labels=['p10', 'p50', 'p90']):
    predictions = {}
    for label in quantile_labels:
        if label in models:
            predictions[label] = models[label].predict(X_pred)
    return predictions


def build_prediction_intervals(predictions_df, models, feature_cols, quantiles=[0.1, 0.5, 0.9]):
    X_pred = predictions_df[feature_cols].fillna(0).values

    quantile_preds = predict_quantile(models, X_pred)

    result_df = predictions_df[['market', 'product_code', 'date']].copy()
    result_df['predicted_p50'] = np.clip(quantile_preds.get('p50', 0), 0, None)
    result_df['predicted_p10'] = np.clip(quantile_preds.get('p10', 0), 0, None)
    result_df['predicted_p90'] = np.clip(quantile_preds.get('p90', 0), 0, None)
    result_df['interval_width'] = result_df['predicted_p90'] - result_df['predicted_p10']
    result_df['coef_variation'] = np.where(
        result_df['predicted_p50'] > 0,
        result_df['interval_width'] / (2 * result_df['predicted_p50']),
        0
    )

    return result_df


def prepare_submission_with_intervals(result_df, submission_template_path='data/submission.csv'):
    submission = pd.read_csv(submission_template_path)
    submission['date'] = pd.to_datetime(submission['date'])
    result_df['date'] = pd.to_datetime(result_df['date'])
    result_df['unique_code'] = result_df['market'] + '-' + result_df['product_code']

    submission = submission.merge(
        result_df[['unique_code', 'date', 'predicted_p50']],
        on=['unique_code', 'date'],
        how='left'
    )
    submission['quantity'] = submission['predicted_p50'].fillna(0).clip(lower=0)
    submission = submission[['ID', 'unique_code', 'date', 'quantity']]

    return submission
