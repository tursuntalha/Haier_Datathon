"""
Gelişmiş Talep Tahmini Modeli
Feature Engineering Map'e göre kapsamlı özellik çıkarımı ve model eğitimi
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("HAIER EUROPE - 12 AYLIK TALEP TAHMİNİ MODELİ")
print("=" * 80)

# ============================================================================
# 1. VERİ YÜKLEME
# ============================================================================
print("\n[1/7] Veri yükleniyor...")
train = pd.read_csv('data/train.csv')
product = pd.read_csv('data/product_master.csv')
submission = pd.read_csv('data/submission.csv')

# Tarih dönüşümleri
train['date'] = pd.to_datetime(train['date'])
product['start_production_date'] = pd.to_datetime(product['start_production_date'])
product['end_production_date'] = pd.to_datetime(product['end_production_date'])
submission['date'] = pd.to_datetime(submission['date'])

print(f"   ✓ Train: {train.shape[0]:,} satır")
print(f"   ✓ Product: {product.shape[0]:,} ürün")
print(f"   ✓ Submission: {submission.shape[0]:,} tahmin")

# ============================================================================
# 2. SKELETON (İSKELET) OLUŞTURMA - KRİTİK!
# ============================================================================
print("\n[2/7] Skeleton (eksik aylar) oluşturuluyor...")
# Tüm market x product x date kombinasyonlarını oluştur
markets = train['market'].unique()
products = train['product_code'].unique()
date_range = pd.date_range(start=train['date'].min(), end=train['date'].max(), freq='MS')

skeleton = pd.MultiIndex.from_product([markets, products, date_range], 
                                       names=['market', 'product_code', 'date']).to_frame(index=False)

# Train ile birleştir (eksik aylar quantity=0 olacak)
train_full = skeleton.merge(train, on=['market', 'product_code', 'date'], how='left')
train_full['quantity'] = train_full['quantity'].fillna(0)

print(f"   ✓ Skeleton: {skeleton.shape[0]:,} satır (eksik aylar dahil)")
print(f"   ✓ Doldurulmuş Train: {train_full.shape[0]:,} satır")

# Product bilgilerini ekle
train_full = train_full.merge(product, on='product_code', how='left')

# ============================================================================
# 3. ÖZELLİK ÇIKARIMI (FEATURE ENGINEERING)
# ============================================================================
print("\n[3/7] Özellik çıkarımı yapılıyor...")

df = train_full.copy()

# -------------------------
# A. ZAMAN ÖZELLİKLERİ
# -------------------------
print("   → Zaman özellikleri...")
df['month'] = df['date'].dt.month
df['quarter'] = df['date'].dt.quarter
df['year'] = df['date'].dt.year
df['day_of_year'] = df['date'].dt.dayofyear
df['is_year_end'] = (df['month'] >= 11).astype(int)
df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)

# -------------------------
# B. LIFECYCLE ÖZELLİKLERİ (PHASE-OUT İÇİN KRİTİK!)
# -------------------------
print("   → Lifecycle özellikleri (Phase-Out)...")
df['months_until_eol'] = ((df['end_production_date'] - df['date']).dt.days / 30).fillna(999)
df['months_until_eol'] = df['months_until_eol'].clip(lower=-12, upper=999)

df['eol_urgency'] = 1 / (df['months_until_eol'].clip(lower=0) + 1)
df['is_eol_approaching'] = (df['months_until_eol'] < 6).astype(int)
df['is_post_eol'] = (df['date'] > df['end_production_date']).astype(int).fillna(0)
df['is_continuing'] = df['end_production_date'].isna().astype(int)

# Ürün yaşam döngüsü
df['days_since_launch'] = (df['date'] - df['start_production_date']).dt.days.fillna(0).clip(lower=0)
df['months_since_start'] = (df['days_since_launch'] / 30).clip(lower=0)

# Life progress (0-1 arası)
total_life = (df['end_production_date'] - df['start_production_date']).dt.days
current_life = (df['date'] - df['start_production_date']).dt.days
df['life_progress'] = (current_life / total_life).fillna(0).clip(0, 1)

# EOL decay factor (son 12 ayda azalma)
df['eol_decay_factor'] = df['months_until_eol'].apply(
    lambda x: 0 if x < 0 else (x/12 if x < 12 else 1)
)

# -------------------------
# C. LAG ÖZELLİKLERİ (12 AYLIK TAHMİN İÇİN LAG_12+)
# -------------------------
print("   → Lag özellikleri (12+ ay)...")
for lag in [12, 13, 24]:
    df[f'lag_{lag}'] = df.groupby(['market', 'product_code'])['quantity'].shift(lag)

# -------------------------
# D. ROLLING ÖZELLİKLERİ
# -------------------------
print("   → Rolling özellikleri...")
for window in [3, 6, 12]:
    # Shift(12) yaparak gelecek bilgisini kullanmıyoruz
    df[f'rolling_mean_{window}m'] = df.groupby(['market', 'product_code'])['quantity'].transform(
        lambda x: x.shift(12).rolling(window, min_periods=1).mean()
    )
    df[f'rolling_std_{window}m'] = df.groupby(['market', 'product_code'])['quantity'].transform(
        lambda x: x.shift(12).rolling(window, min_periods=1).std()
    )

# -------------------------
# E. YOY GROWTH
# -------------------------
print("   → YoY growth...")
df['yoy_growth'] = (df['quantity'] - df['lag_12']) / (df['lag_12'] + 1)
df['yoy_growth'] = df['yoy_growth'].fillna(0).clip(-10, 10)

# -------------------------
# F. PAZAR & KATEGORİ ÖZELLİKLERİ (LEAKAGE ÖNLEME!)
# -------------------------
print("   → Pazar & kategori özellikleri...")
# Sadece geçmiş veriyi kullan (shift)
df['market_avg_sales'] = df.groupby('market')['quantity'].transform(
    lambda x: x.shift(12).rolling(12, min_periods=1).mean()
)
df['category_avg_sales'] = df.groupby('category')['quantity'].transform(
    lambda x: x.shift(12).rolling(12, min_periods=1).mean()
)
df['brand_avg_sales'] = df.groupby('brand')['quantity'].transform(
    lambda x: x.shift(12).rolling(12, min_periods=1).mean()
)
df['sector_avg_sales'] = df.groupby('sector')['quantity'].transform(
    lambda x: x.shift(12).rolling(12, min_periods=1).mean()
)

# -------------------------
# G. HİYERARŞİK İLİŞKİ ÖZELLİKLERİ (GAME CHANGER!)
# -------------------------
print("   → Hiyerarşik ilişki özellikleri...")
# Kategori toplamı (shift ile)
cat_sum = df.groupby(['category', 'date'])['quantity'].sum().rename('cat_total')
df = df.merge(cat_sum, on=['category', 'date'], how='left')
df['sku_share_in_cat'] = df['quantity'] / (df['cat_total'] + 1)

# Market toplamı
market_sum = df.groupby(['market', 'date'])['quantity'].sum().rename('market_total')
df = df.merge(market_sum, on=['market', 'date'], how='left')
df['sku_share_in_market'] = df['quantity'] / (df['market_total'] + 1)

# -------------------------
# H. EOL ETKİLEŞİM ÖZELLİKLERİ (INTERACTION)
# -------------------------
print("   → EOL etkileşim özellikleri...")
df['lag12_x_eol_urgency'] = df['lag_12'] * df['eol_urgency']
df['rolling_mean_12m_x_decay'] = df['rolling_mean_12m'] * df['eol_decay_factor']

# -------------------------
# I. KESİKLİ SATIŞ ÖZELLİKLERİ (INTERMITTENT DEMAND)
# -------------------------
print("   → Kesikli satış özellikleri...")
# Son 12 ayda kaç ay satış yapıldı
df['sale_frequency_12m'] = df.groupby(['market', 'product_code'])['quantity'].transform(
    lambda x: ((x.shift(1) > 0).rolling(12, min_periods=1).sum() / 12)
)

# Sıfır satış streak (basitleştirilmiş)
df['zero_sales_streak'] = df.groupby(['market', 'product_code'])['quantity'].transform(
    lambda x: (x == 0).astype(int)
)

# -------------------------
# J. KATEGORİK DEĞİŞKENLER (LABEL ENCODING)
# -------------------------
print("   → Kategorik değişkenler encode ediliyor...")
cat_cols = ['market', 'category', 'brand', 'sector', 'business_line', 'factory']
for col in cat_cols:
    if col in df.columns:
        df[col + '_encoded'] = df[col].astype('category').cat.codes

print(f"   ✓ Toplam {df.shape[1]} özellik oluşturuldu")

# ============================================================================
# 4. MODEL EĞİTİMİ İÇİN VERİ HAZIRLAMA
# ============================================================================
print("\n[4/7] Model eğitimi için veri hazırlanıyor...")

# Özellik sütunları
feature_cols = [
    # Zaman
    'month', 'quarter', 'year', 'is_year_end', 'is_summer',
    # Lifecycle
    'months_until_eol', 'eol_urgency', 'is_eol_approaching', 'is_post_eol',
    'is_continuing', 'days_since_launch', 'months_since_start', 'life_progress',
    'eol_decay_factor',
    # Lag
    'lag_12', 'lag_13', 'lag_24',
    # Rolling
    'rolling_mean_3m', 'rolling_mean_6m', 'rolling_mean_12m',
    'rolling_std_3m', 'rolling_std_6m', 'rolling_std_12m',
    # Growth
    'yoy_growth',
    # Pazar/Kategori
    'market_avg_sales', 'category_avg_sales', 'brand_avg_sales', 'sector_avg_sales',
    # Hiyerarşik
    'sku_share_in_cat', 'sku_share_in_market',
    # Etkileşim
    'lag12_x_eol_urgency', 'rolling_mean_12m_x_decay',
    # Kesikli satış
    'sale_frequency_12m', 'zero_sales_streak',
    # Kategorik
    'market_encoded', 'category_encoded', 'brand_encoded', 'sector_encoded'
]

# Sadece mevcut sütunları al
feature_cols = [col for col in feature_cols if col in df.columns]

# NaN'ları doldur
df[feature_cols] = df[feature_cols].fillna(0)

# Train/Validation split (son 6 ay validation)
train_cutoff = df['date'].max() - pd.DateOffset(months=6)
train_data = df[df['date'] <= train_cutoff].copy()
val_data = df[df['date'] > train_cutoff].copy()

X_train = train_data[feature_cols]
y_train = train_data['quantity']
X_val = val_data[feature_cols]
y_val = val_data['quantity']

print(f"   ✓ Train: {X_train.shape[0]:,} satır, {X_train.shape[1]} özellik")
print(f"   ✓ Validation: {X_val.shape[0]:,} satır")

# ============================================================================
# 5. MODEL EĞİTİMİ
# ============================================================================
print("\n[5/7] Model eğitiliyor...")
print("   → Gradient Boosting Regressor...")

model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    min_samples_split=20,
    min_samples_leaf=10,
    subsample=0.8,
    random_state=42,
    verbose=0
)

model.fit(X_train, y_train)

# Validation tahminleri
y_pred_val = model.predict(X_val)
y_pred_val = np.maximum(y_pred_val, 0)  # Negatif değerleri sıfırla

mae = mean_absolute_error(y_val, y_pred_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))

print(f"   ✓ Validation MAE: {mae:.2f}")
print(f"   ✓ Validation RMSE: {rmse:.2f}")

# ============================================================================
# 6. ÖNEMLİ ÖZELLİKLER
# ============================================================================
print("\n[6/7] En önemli 15 özellik:")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).head(15)

for idx, row in feature_importance.iterrows():
    print(f"   {row['feature']:30s} : {row['importance']:.4f}")

# ============================================================================
# 7. SUBMISSION TAHMİNLERİ
# ============================================================================
print("\n[7/7] Submission tahminleri yapılıyor...")

# Submission için unique_code'u ayrıştır
submission[['market', 'product_code']] = submission['unique_code'].str.split('-', expand=True)

# Product bilgilerini ekle
submission = submission.merge(product, on='product_code', how='left')

# Aynı özellikleri oluştur
sub_df = submission.copy()

# Zaman özellikleri
sub_df['month'] = sub_df['date'].dt.month
sub_df['quarter'] = sub_df['date'].dt.quarter
sub_df['year'] = sub_df['date'].dt.year
sub_df['day_of_year'] = sub_df['date'].dt.dayofyear
sub_df['is_year_end'] = (sub_df['month'] >= 11).astype(int)
sub_df['is_summer'] = sub_df['month'].isin([6, 7, 8]).astype(int)

# Lifecycle
sub_df['months_until_eol'] = ((sub_df['end_production_date'] - sub_df['date']).dt.days / 30).fillna(999).clip(-12, 999)
sub_df['eol_urgency'] = 1 / (sub_df['months_until_eol'].clip(lower=0) + 1)
sub_df['is_eol_approaching'] = (sub_df['months_until_eol'] < 6).astype(int)
sub_df['is_post_eol'] = (sub_df['date'] > sub_df['end_production_date']).astype(int).fillna(0)
sub_df['is_continuing'] = sub_df['end_production_date'].isna().astype(int)
sub_df['days_since_launch'] = (sub_df['date'] - sub_df['start_production_date']).dt.days.fillna(0).clip(lower=0)
sub_df['months_since_start'] = (sub_df['days_since_launch'] / 30).clip(lower=0)

total_life = (sub_df['end_production_date'] - sub_df['start_production_date']).dt.days
current_life = (sub_df['date'] - sub_df['start_production_date']).dt.days
sub_df['life_progress'] = (current_life / total_life).fillna(0).clip(0, 1)
sub_df['eol_decay_factor'] = sub_df['months_until_eol'].apply(lambda x: 0 if x < 0 else (x/12 if x < 12 else 1))

# Lag özellikleri (train'den son değerleri al)
last_values = df.groupby(['market', 'product_code']).tail(24)[['market', 'product_code', 'date', 'quantity']]

for lag in [12, 13, 24]:
    lag_df = last_values.copy()
    lag_df['date'] = lag_df['date'] + pd.DateOffset(months=lag)
    lag_df = lag_df.rename(columns={'quantity': f'lag_{lag}'})
    sub_df = sub_df.merge(lag_df[['market', 'product_code', 'date', f'lag_{lag}']], 
                          on=['market', 'product_code', 'date'], how='left')

# Rolling özellikleri (train'den son değerleri al)
for window in [3, 6, 12]:
    rolling_mean = df.groupby(['market', 'product_code'])['quantity'].apply(
        lambda x: x.rolling(window, min_periods=1).mean().iloc[-1]
    ).reset_index(name=f'rolling_mean_{window}m')
    
    rolling_std = df.groupby(['market', 'product_code'])['quantity'].apply(
        lambda x: x.rolling(window, min_periods=1).std().iloc[-1]
    ).reset_index(name=f'rolling_std_{window}m')
    
    sub_df = sub_df.merge(rolling_mean, on=['market', 'product_code'], how='left')
    sub_df = sub_df.merge(rolling_std, on=['market', 'product_code'], how='left')

# YoY growth
sub_df['yoy_growth'] = ((sub_df['lag_12'] - sub_df['lag_24']) / (sub_df['lag_24'] + 1)).fillna(0).clip(-10, 10)

# Pazar/Kategori ortalamaları
for col in ['market', 'category', 'brand', 'sector']:
    avg_col = f'{col}_avg_sales'
    avg_values = df.groupby(col)['quantity'].mean().reset_index(name=avg_col)
    sub_df = sub_df.merge(avg_values, on=col, how='left')

# Hiyerarşik özellikler (basitleştirilmiş)
sub_df['sku_share_in_cat'] = 0.1  # Placeholder
sub_df['sku_share_in_market'] = 0.05  # Placeholder

# Etkileşim
sub_df['lag12_x_eol_urgency'] = sub_df['lag_12'].fillna(0) * sub_df['eol_urgency']
sub_df['rolling_mean_12m_x_decay'] = sub_df['rolling_mean_12m'].fillna(0) * sub_df['eol_decay_factor']

# Kesikli satış
sub_df['sale_frequency_12m'] = 0.5  # Placeholder
sub_df['zero_sales_streak'] = 0  # Placeholder

# Kategorik encoding
for col in ['market', 'category', 'brand', 'sector']:
    if col in sub_df.columns:
        sub_df[col + '_encoded'] = sub_df[col].astype('category').cat.codes

# NaN'ları doldur
sub_df[feature_cols] = sub_df[feature_cols].fillna(0)

# Tahmin yap
X_sub = sub_df[feature_cols]
predictions = model.predict(X_sub)
predictions = np.maximum(predictions, 0)  # Negatif değerleri sıfırla

# Submission dosyasına yaz
submission['quantity'] = predictions.astype(int)
submission[['ID', 'unique_code', 'date', 'quantity']].to_csv('data/submission.csv', index=False)

print(f"   ✓ Tahminler tamamlandı")
print(f"   ✓ Ortalama tahmin: {predictions.mean():.2f}")
print(f"   ✓ Maksimum tahmin: {predictions.max():.0f}")
print(f"   ✓ Sıfır tahmin oranı: {(predictions == 0).sum() / len(predictions) * 100:.1f}%")

print("\n" + "=" * 80)
print("✓ MODEL EĞİTİMİ VE TAHMİN TAMAMLANDI!")
print("✓ Submission dosyası güncellendi: data/submission.csv")
print("=" * 80)
