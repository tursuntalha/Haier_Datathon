# 📚 Modül Özeti ve Fonksiyon Referansı

## 🎯 Genel Bakış

Bu proje, Haier Europe Datathon yarışması için 12 aylık talep tahmini yapan bir makine öğrenmesi pipeline'ı içerir.

## 📦 Modüller

### 1️⃣ `feature.py` - Veri Ön İşleme ve Özellik Mühendisliği

#### `fill_missing_values_no_drop(train_df, product_df)`
**Amaç**: Product master verisini temizler ve eksik tarihleri doldurur

**Parametreler**:
- `train_df`: Eğitim verisi (market, product_code, date, quantity)
- `product_df`: Ürün master verisi

**Döndürür**: `(train_df, product_df)` - Temizlenmiş dataframe'ler

**Yaptığı İşlemler**:
- Tarih sütunlarını datetime'a çevirir
- Eksik start_production_date'leri ilk satış tarihinden doldurur
- Grup bazlı doldurma yapar (category, brand, sector)
- `is_continuing` bayrağı ekler (end_production_date boşsa 1)

---

#### `expand_grid_train_only(train_df)`
**Amaç**: Zaman serisi için eksik ay-ürün kombinasyonlarını ekler

**Parametreler**:
- `train_df`: Eğitim verisi

**Döndürür**: `train_df` - Genişletilmiş dataframe

**Yaptığı İşlemler**:
- Her market-product kombinasyonu için tüm tarihleri oluşturur (Cartesian product)
- Eksik ayları NaN ile doldurur (sonraki adımda filtrelenecek)

---

#### `add_advanced_lifecycle_features(train_df, product_df)`
**Amaç**: Ürün yaşam döngüsü özellikleri ekler ve üretim öncesi satırları temizler

**Parametreler**:
- `train_df`: Genişletilmiş eğitim verisi
- `product_df`: Ürün master verisi

**Döndürür**: `train_df` - Özellikler eklenmiş dataframe

**Yaptığı İşlemler**:
1. **Üretim Öncesi Temizlik**: Üretim başlamadan önceki yapay satırları siler
2. **Özellik Ekleme**:
   - `flag_pre_production_sale`: Üretim öncesi satış bayrağı (anomali tespiti)
   - `flag_eol_passed`: End-of-life tarihi geçmiş mi?
   - `months_until_eol`: EOL'a kaç ay kaldı (999 = devam eden)
   - `eol_urgency`: Phase-out aciliyeti (1 / (months_until_eol + 1))
   - `life_progress`: Ürün yaşam döngüsü ilerlemesi (0-1 arası)

---

### 2️⃣ `utils.py` - Skor Hesaplama ve Değerlendirme

#### `calculate_rwmape(y_true, y_pred, lambda_mass=0.5, gamma_forecast=0.1, epsilon=1e-10)`
**Amaç**: Regularized WMAPE hesaplar

**Parametreler**:
- `y_true`: Gerçek değerler
- `y_pred`: Tahmin değerleri
- `lambda_mass`: Toplam hacim cezası katsayısı (default: 0.5)
- `gamma_forecast`: Küçük tahmin cezası (default: 0.1)
- `epsilon`: Sayısal kararlılık sabiti (default: 1e-10)

**Döndürür**: `float` - rWMAPE skoru

**Formül**:
```
rWMAPE = (Σ|y_true - y_pred| + λ * |Σy_true - Σy_pred|) / (Σy_true + γ * Σy_pred + ε)
```

---

#### `calculate_group_rwmape(df, group_col='unique_code', true_col='quantity_true', pred_col='quantity_pred')`
**Amaç**: Grup bazlı rWMAPE hesaplar (yarışma metriği)

**Parametreler**:
- `df`: Gerçek ve tahmin değerlerini içeren dataframe
- `group_col`: Gruplama sütunu (default: 'unique_code')
- `true_col`: Gerçek değer sütunu
- `pred_col`: Tahmin değer sütunu

**Döndürür**: `(avg_score, group_details)` - Ortalama skor ve grup detayları

**Mantık**:
- Her grup için ayrı rWMAPE hesaplar
- İki toplam da sıfırsa → grubu atla
- Gerçek sıfır ama tahmin varsa → ceza = 1.0
- Diğer durumlarda → normal rWMAPE

---

#### `calculate_competition_score(rwmape)`
**Amaç**: Yarışma skorunu hesaplar

**Parametreler**:
- `rwmape`: rWMAPE değeri

**Döndürür**: `float` - Competition score (0-1 arası, yüksek = iyi)

**Formül**:
```
Score = 1 / (1 + rWMAPE)
```

---

#### `create_time_series_splits(train_df, n_splits=3, test_months=12)`
**Amaç**: Zaman serisi için çapraz doğrulama split'leri oluşturur

**Parametreler**:
- `train_df`: Eğitim verisi (date sütunu olmalı)
- `n_splits`: Kaç fold (default: 3)
- `test_months`: Test için kaç ay (default: 12)

**Döndürür**: `list` - (train_dates, val_dates) tuple'larının listesi

**Mantık**:
- Son tarihten geriye doğru test_months kadar ayırır
- Her fold için farklı validation dönemi

---

#### `prepare_submission_format(predictions_df, submission_template_path='data/submission.csv')`
**Amaç**: Tahminleri yarışma formatına dönüştürür

**Parametreler**:
- `predictions_df`: Tahminler (market, product_code, date, quantity)
- `submission_template_path`: Template dosya yolu

**Döndürür**: `pd.DataFrame` - Submission formatında dataframe

**Yaptığı İşlemler**:
- unique_code oluşturur (market-product_code)
- Template ile merge eder
- Negatif değerleri 0 yapar
- ID, unique_code, date, quantity sütunlarını döndürür

---

#### `evaluate_model_cv(train_df, product_df, model_func, n_splits=3, test_months=12)`
**Amaç**: Model performansını çapraz doğrulama ile değerlendirir

**Parametreler**:
- `train_df`: Eğitim verisi
- `product_df`: Ürün master verisi
- `model_func`: Model fonksiyonu (signature: `model_func(train_data, product_data, forecast_dates) -> predictions_df`)
- `n_splits`: Kaç fold
- `test_months`: Test için kaç ay

**Döndürür**: `dict` - CV sonuçları
```python
{
    'fold_rwmapes': [0.34, 0.35, 0.33],
    'fold_competition_scores': [0.74, 0.73, 0.75],
    'avg_rwmape': 0.34,
    'avg_competition_score': 0.74,
    'std_rwmape': 0.01,
    'std_competition_score': 0.01
}
```

---

### 3️⃣ `model.py` - Model Eğitimi ve Tahmin

#### `create_lag_features(df, lags=[1, 2, 3, 6, 12])`
**Amaç**: Gecikmeli özellikler oluşturur

**Parametreler**:
- `df`: Dataframe
- `lags`: Gecikme değerleri (default: [1, 2, 3, 6, 12])

**Döndürür**: `df` - Lag özellikleri eklenmiş dataframe

**Eklenen Sütunlar**: `lag_1`, `lag_2`, `lag_3`, `lag_6`, `lag_12`

---

#### `create_rolling_features(df, windows=[3, 6, 12])`
**Amaç**: Hareketli ortalama özellikleri oluşturur

**Parametreler**:
- `df`: Dataframe
- `windows`: Pencere boyutları (default: [3, 6, 12])

**Döndürür**: `df` - Rolling özellikleri eklenmiş dataframe

**Eklenen Sütunlar**: 
- `rolling_mean_3`, `rolling_mean_6`, `rolling_mean_12`
- `rolling_std_3`, `rolling_std_6`, `rolling_std_12`

---

#### `create_time_features(df)`
**Amaç**: Zaman bazlı özellikler oluşturur

**Parametreler**:
- `df`: Dataframe (date sütunu olmalı)

**Döndürür**: `df` - Zaman özellikleri eklenmiş dataframe

**Eklenen Sütunlar**:
- `year`: Yıl
- `month`: Ay (1-12)
- `quarter`: Çeyrek (1-4)
- `month_sin`: Mevsimsellik için sinüs (sin(2π * month / 12))
- `month_cos`: Mevsimsellik için kosinüs (cos(2π * month / 12))

---

#### `prepare_features(train_df, product_df)`
**Amaç**: Tüm özellikleri hazırlar

**Parametreler**:
- `train_df`: Eğitim verisi
- `product_df`: Ürün master verisi

**Döndürür**: `train_df` - Tüm özellikler eklenmiş dataframe

**Yaptığı İşlemler**:
1. Product bilgilerini merge eder
2. Zaman özellikleri ekler
3. Lag özellikleri ekler
4. Rolling özellikleri ekler
5. Kategorik değişkenleri encode eder (market, category, brand, sector)

---

#### `train_model(train_df, product_df, forecast_dates)`
**Amaç**: Model eğitir ve tahmin yapar

**Parametreler**:
- `train_df`: Eğitim verisi
- `product_df`: Ürün master verisi
- `forecast_dates`: Tahmin yapılacak tarihler (list of datetime)

**Döndürür**: `pd.DataFrame` - Tahminler (market, product_code, date, quantity)

**Model**: LightGBM (LGBMRegressor)
- n_estimators: 200
- learning_rate: 0.05
- max_depth: 7
- num_leaves: 31
- min_child_samples: 20
- subsample: 0.8
- colsample_bytree: 0.8
- random_state: 42

**Özel Mantık**:
- EOL geçmiş ürünler için tahmin × 0.1 (90% azaltma)
- EOL'a 3 aydan az kalan ürünler için tahmin × 0.5 (50% azaltma)
- Negatif tahminler 0'a çekilir

---

#### `train_and_predict_full(train_df, product_df, submission_template_path='data/submission.csv')`
**Amaç**: Tüm veri ile final model eğitir ve submission için tahmin yapar

**Parametreler**:
- `train_df`: Tüm eğitim verisi
- `product_df`: Ürün master verisi
- `submission_template_path`: Submission template yolu

**Döndürür**: `pd.DataFrame` - Tahminler

**Kullanım**:
```python
predictions = train_and_predict_full(train, product)
```

---

## 🔄 Pipeline Akışı

```
1. Veri Yükleme
   ↓
2. Veri Temizleme (feature.py)
   - fill_missing_values_no_drop()
   - expand_grid_train_only()
   - add_advanced_lifecycle_features()
   ↓
3. Çapraz Doğrulama (utils.py + model.py)
   - evaluate_model_cv()
   ↓
4. Final Model Eğitimi (model.py)
   - train_and_predict_full()
   ↓
5. Submission Oluşturma (utils.py)
   - prepare_submission_format()
   ↓
6. Doğrulama ve Kaydetme
```

## 📊 Kullanılan Özellikler

### Zaman Özellikleri
- year, month, quarter
- month_sin, month_cos (mevsimsellik)

### Lag Özellikleri
- lag_1, lag_2, lag_3, lag_6, lag_12

### Rolling Özellikleri
- rolling_mean_3, rolling_mean_6, rolling_mean_12
- rolling_std_3, rolling_std_6, rolling_std_12

### Lifecycle Özellikleri
- eol_urgency (phase-out aciliyeti)
- life_progress (ürün yaşam döngüsü)
- months_until_eol (EOL'a kalan ay)
- flag_eol_passed (EOL geçmiş mi?)

### Kategorik Özellikler
- market_encoded
- category_encoded
- brand_encoded
- sector_encoded

---

## 🎯 Örnek Kullanım

```python
# 1. Veri yükle
train = pd.read_csv("data/train.csv")
product = pd.read_csv("data/product_master.csv")

# 2. Ön işleme
from feature import *
train, product = fill_missing_values_no_drop(train, product)
train = expand_grid_train_only(train)
train = add_advanced_lifecycle_features(train, product)

# 3. CV ile değerlendir
from utils import evaluate_model_cv
from model import train_model
cv_results = evaluate_model_cv(train, product, train_model, n_splits=3)

# 4. Final model ve submission
from model import train_and_predict_full
from utils import prepare_submission_format
predictions = train_and_predict_full(train, product)
submission = prepare_submission_format(predictions)
submission.to_csv('data/my_submission.csv', index=False)
```

---

**Not**: Tüm fonksiyonlar docstring'ler içerir. Detaylı bilgi için `help(function_name)` kullanabilirsiniz.
