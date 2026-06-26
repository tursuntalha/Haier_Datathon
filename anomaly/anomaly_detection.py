import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def detect_statistical_outliers(df, group_cols=['product_code'], value_col='quantity', z_threshold=3):
    print("\n" + "=" * 60)
    print("Anomaly Detection Layer")
    print("=" * 60)

    result = df.copy()
    result['_is_anomaly'] = False
    result['_anomaly_reason'] = ''
    result['_anomaly_score'] = 0.0

    for group_name, group_df in result.groupby(group_cols):
        idx = group_df.index
        values = group_df[value_col].values

        if len(values) < 4:
            continue

        z_scores = np.abs(stats.zscore(values, nan_policy='omit'))
        rolling_mean = pd.Series(values).rolling(window=3, min_periods=2).mean().values
        rolling_std = pd.Series(values).rolling(window=3, min_periods=2).std().values

        global_outliers = z_scores > z_threshold

        local_anomalies = np.zeros(len(values), dtype=bool)
        for i in range(len(values)):
            if i >= 3 and rolling_std[i] > 0:
                deviation = abs(values[i] - rolling_mean[i])
                if deviation > z_threshold * rolling_std[i]:
                    local_anomalies[i] = True

        result.loc[idx, '_is_anomaly'] = global_outliers | local_anomalies
        result.loc[idx, '_anomaly_score'] = z_scores

        for i in idx[global_outliers]:
            result.at[i, '_anomaly_reason'] = f'3-sigma global outlier (z={z_scores[result.index.get_loc(i)]:.2f})'
        for i in idx[local_anomalies & ~global_outliers]:
            result.at[i, '_anomaly_reason'] = f'Local outlier (z={z_scores[result.index.get_loc(i)]:.2f})'

    n_anomalies = result['_is_anomaly'].sum()
    print(f"  Statistical outliers flagged: {n_anomalies} ({n_anomalies / len(result) * 100:.2f}%)")

    return result


def detect_zero_sales_streaks(df, group_cols=['product_code'], streak_length=3, value_col='quantity'):
    result = df.copy()
    if '_is_anomaly' not in result.columns:
        result['_is_anomaly'] = False
        result['_anomaly_reason'] = ''
        result['_anomaly_score'] = 0.0

    streak_count = 0
    for group_name, group_df in result.sort_values(['product_code', 'date']).groupby(group_cols):
        idx = group_df.index
        values = group_df[value_col].values

        consecutive_zeros = 0
        for i in range(len(values)):
            if values[i] == 0:
                consecutive_zeros += 1
            else:
                consecutive_zeros = 0

            if consecutive_zeros >= streak_length:
                abs_idx = idx[i]
                prev_sales = values[i - streak_length] if i >= streak_length else 0
                if prev_sales > 0:
                    if not result.at[abs_idx, '_is_anomaly']:
                        result.at[abs_idx, '_is_anomaly'] = True
                        result.at[abs_idx, '_anomaly_reason'] = f'Zero-sales streak ({streak_length}+ months)'
                        result.at[abs_idx, '_anomaly_score'] = 1.0
                        streak_count += 1

    print(f"  Zero-sales streaks (>= {streak_length} months) flagged: {streak_count}")
    return result


def replace_anomalies(df, value_col='quantity', strategy='clip', clip_percentile=99):
    result = df.copy()

    n_anomalies = result['_is_anomaly'].sum()
    if n_anomalies == 0:
        print("  No anomalies to replace.")
        return result

    if strategy == 'clip':
        upper_bound = np.percentile(result.loc[~result['_is_anomaly'], value_col].fillna(0), clip_percentile)
        result.loc[result['_is_anomaly'], value_col] = result.loc[result['_is_anomaly'], value_col].clip(upper=upper_bound)
        print(f"  Clipped {n_anomalies} anomalies to p{clip_percentile} bound ({upper_bound:.2f})")

    elif strategy == 'median':
        for group_name, group_df in result[result['_is_anomaly']].groupby('product_code'):
            median_val = result.loc[result['product_code'] == group_name, value_col].median()
            result.loc[result['product_code'] == group_name, value_col] = median_val
        print(f"  Replaced {n_anomalies} anomalies with product median")

    elif strategy == 'rolling_mean':
        for idx in result[result['_is_anomaly']].index:
            product = result.at[idx, 'product_code']
            product_data = result[result['product_code'] == product].sort_values('date')
            prod_idx = product_data.index.get_loc(idx)
            if prod_idx >= 3:
                replacement = product_data[value_col].iloc[prod_idx - 3:prod_idx].mean()
            else:
                replacement = product_data[value_col].iloc[:prod_idx].mean() if prod_idx > 0 else 0
            result.at[idx, value_col] = max(replacement, 0)
        print(f"  Replaced {n_anomalies} anomalies with rolling mean")

    return result


def run_anomaly_pipeline(train_df, z_threshold=3, streak_length=3, replacement_strategy='clip'):
    print("Running anomaly detection pipeline...")

    result = detect_statistical_outliers(train_df, z_threshold=z_threshold)
    result = detect_zero_sales_streaks(result, streak_length=streak_length)
    result = replace_anomalies(result, strategy=replacement_strategy)

    anomaly_flags = result[['_is_anomaly', '_anomaly_reason', '_anomaly_score']].copy()
    clean_df = result.drop(columns=['_is_anomaly', '_anomaly_reason', '_anomaly_score'])

    return clean_df, anomaly_flags
