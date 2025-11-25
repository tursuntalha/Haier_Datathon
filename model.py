import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')


def create_lag_features(df, lags=[1, 2, 3, 6, 12]):
    """
    Gecikmeli özellikler oluştur
    """
    df = df.sort_values(['market', 'product_code', 'date'])
    
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby(['market', 'product_code'])['quantity'].shift(lag)
    
    return df


def create_rolling_features(df, windows=[3, 6, 12]):
    """
    Hareketli ortalama özellikleri oluştur
    """
    df = df.sort_values(['market', 'product_code', 'date'])
    
    for window in windows:
        df[f'rolling_mean_{window}'] = df.groupby(['market', 'product_code'])['quantity'].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        df[f'rolling_std_{window}'] = df.groupby(['market', 'product_code'])['quantity'].transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
    
    return df


def create_time_features(df):
    """
    Zaman bazlı özellikler oluştur
    """
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df


def prepare_features(train_df, product_df):
    """
    Tüm özellikleri hazırla
    """
    print("🔧 Özellikler hazırlanıyor...")
    
    # Product bilgilerini ekle
    train_df = train_df.merge(product_df, on='product_code', how='left')
    
    # Zaman özellikleri
    train_df = create_time_features(train_df)
    
    # Lag özellikleri
    train_df = create_lag_features(train_df, lags=[1, 2, 3, 6, 12])
    
    # Rolling özellikleri
    train_df = create_rolling_features(train_df, windows=[3, 6, 12])
    
    # Kategorik değişkenleri encode et
    for col in ['market', 'category', 'brand', 'sector']:
        if col in train_df.columns:
            train_df[f'{col}_encoded'] = train_df[col].astype('category').cat.codes
    
    print("✅ Özellikler hazır!")
    return train_df


def train_model(train_df, product_df, forecast_dates):
    """
    Model eğit ve tahmin yap
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Eğitim verisi
    product_df : pd.DataFrame
        Ürün master verisi
    forecast_dates : list
        Tahmin yapılacak tarihler
    
    Returns:
    --------
    pd.DataFrame : Tahminler (market, product_code, date, quantity)
    """
    print("\n" + "="*60)
    print("🚀 Model Eğitimi Başlıyor")
    print("="*60)
    
    # Özellikleri hazırla
    train_prepared = prepare_features(train_df.copy(), product_df)
    
    # Feature sütunları
    feature_cols = [
        'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
        'rolling_mean_3', 'rolling_mean_6', 'rolling_mean_12',
        'rolling_std_3', 'rolling_std_6', 'rolling_std_12',
        'month', 'quarter', 'month_sin', 'month_cos',
        'eol_urgency', 'life_progress', 'months_until_eol',
        'flag_eol_passed', 'market_encoded', 'category_encoded', 
        'brand_encoded', 'sector_encoded'
    ]
    
    # Mevcut sütunları filtrele
    feature_cols = [col for col in feature_cols if col in train_prepared.columns]
    
    # NaN'ları doldur
    train_prepared[feature_cols] = train_prepared[feature_cols].fillna(0)
    
    # Eğitim verisi
    X_train = train_prepared[feature_cols]
    y_train = train_prepared['quantity'].clip(lower=0)
    
    print(f"📊 Eğitim verisi: {len(X_train)} satır, {len(feature_cols)} özellik")
    
    # Model eğit (LightGBM)
    print("🤖 LightGBM modeli eğitiliyor...")
    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
        force_col_wise=True
    )
    model.fit(X_train, y_train)
    print("✅ Model eğitimi tamamlandı!")
    
    # Tahmin için veri hazırla - BATCH PREDICTION
    print(f"\n📅 {len(forecast_dates)} ay için tahmin yapılıyor (Batch mode)...")
    
    # Her market-product kombinasyonu için
    unique_combinations = train_df[['market', 'product_code']].drop_duplicates()
    
    # Her kombinasyon için son satırı al
    last_rows = train_prepared.groupby(['market', 'product_code']).tail(1).copy()
    
    # end_production_date'i sakla (eğer varsa)
    if 'end_production_date' in last_rows.columns:
        eol_dates = last_rows[['market', 'product_code', 'end_production_date']].copy()
    else:
        # Product'tan al
        eol_dates = product_df[['product_code', 'end_production_date']].copy()
    
    # Tüm tahmin satırlarını bir kerede oluştur
    forecast_rows = []
    for forecast_date in forecast_dates:
        temp_df = last_rows.copy()
        temp_df['date'] = pd.to_datetime(forecast_date)
        
        # Zaman özelliklerini güncelle
        temp_df['year'] = temp_df['date'].dt.year
        temp_df['month'] = temp_df['date'].dt.month
        temp_df['quarter'] = temp_df['date'].dt.quarter
        temp_df['month_sin'] = np.sin(2 * np.pi * temp_df['month'] / 12)
        temp_df['month_cos'] = np.cos(2 * np.pi * temp_df['month'] / 12)
        
        # EOL bilgisini merge et
        if 'end_production_date' not in temp_df.columns:
            temp_df = temp_df.merge(eol_dates, on='product_code', how='left')
        
        # EOL özelliklerini güncelle
        temp_df['end_production_date'] = pd.to_datetime(temp_df['end_production_date'], errors='coerce')
        months_until = (temp_df['end_production_date'].dt.year - temp_df['date'].dt.year) * 12 + \
                      (temp_df['end_production_date'].dt.month - temp_df['date'].dt.month)
        temp_df['months_until_eol'] = months_until.clip(lower=0).fillna(999)
        temp_df['eol_urgency'] = 1 / (temp_df['months_until_eol'] + 1)
        temp_df['flag_eol_passed'] = (temp_df['date'] > temp_df['end_production_date']).astype(int).fillna(0)
        
        forecast_rows.append(temp_df)
    
    # Tüm tahmin satırlarını birleştir
    all_forecast_df = pd.concat(forecast_rows, ignore_index=True)
    
    # Feature matrix oluştur
    X_pred = all_forecast_df[feature_cols].fillna(0).values
    
    # BATCH PREDICTION - Tek seferde tüm tahminler
    print(f"🚀 Batch prediction yapılıyor: {len(X_pred)} satır...")
    predictions = model.predict(X_pred)
    
    # Tahminleri dataframe'e ekle
    all_forecast_df['quantity'] = predictions
    
    # EOL mantığını uygula (vektörize)
    eol_passed_mask = all_forecast_df['flag_eol_passed'] == 1
    eol_near_mask = (all_forecast_df['months_until_eol'] < 3) & (~eol_passed_mask)
    
    all_forecast_df.loc[eol_passed_mask, 'quantity'] *= 0.1  # %90 azalt
    all_forecast_df.loc[eol_near_mask, 'quantity'] *= 0.5    # %50 azalt
    
    # Negatif değerleri 0 yap
    all_forecast_df['quantity'] = all_forecast_df['quantity'].clip(lower=0)
    
    # Sadece gerekli sütunları al
    predictions_df = all_forecast_df[['market', 'product_code', 'date', 'quantity']].copy()
    
    print(f"✅ Tahminler tamamlandı: {len(predictions_df)} satır")
    
    # Feature importance göster
    get_feature_importance(model, feature_cols, top_n=15)
    
    print("="*60 + "\n")
    
    return predictions_df


def get_feature_importance(model, feature_names, top_n=20):
    """
    Model'in feature importance'ını göster
    
    Parameters:
    -----------
    model : trained model
        Eğitilmiş LightGBM modeli
    feature_names : list
        Özellik isimleri
    top_n : int
        Gösterilecek en önemli N özellik
    
    Returns:
    --------
    pd.DataFrame : Feature importance tablosu
    """
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)
    
    print(f"\n🔝 En Önemli {top_n} Özellik:")
    print("="*50)
    for idx, row in importance_df.iterrows():
        print(f"{row['feature']:30s} : {row['importance']:8.0f}")
    print("="*50)
    
    return importance_df


def train_and_predict_full(train_df, product_df, submission_template_path='data/submission.csv'):
    """
    Tüm veri ile model eğit ve submission için tahmin yap
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Tüm eğitim verisi
    product_df : pd.DataFrame
        Ürün master verisi
    submission_template_path : str
        Submission template yolu
    
    Returns:
    --------
    pd.DataFrame : Tahminler
    """
    # Submission template'den forecast tarihlerini al
    submission = pd.read_csv(submission_template_path)
    submission['date'] = pd.to_datetime(submission['date'])
    forecast_dates = sorted(submission['date'].unique())
    
    print(f"\n🎯 Forecast Dönemi: {forecast_dates[0].strftime('%Y-%m')} - {forecast_dates[-1].strftime('%Y-%m')}")
    
    # Model eğit ve tahmin yap
    predictions = train_model(train_df, product_df, forecast_dates)
    
    return predictions
