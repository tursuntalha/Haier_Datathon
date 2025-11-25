# 🚀 Datathon Projesi Kullanım Kılavuzu

## 📁 Proje Yapısı

```
├── data/
│   ├── train.csv              # Eğitim verisi
│   ├── product_master.csv     # Ürün master verisi
│   └── submission.csv         # Submission template
├── feature.py                 # Veri ön işleme ve özellik mühendisliği
├── utils.py                   # Skor hesaplama ve değerlendirme fonksiyonları
├── model.py                   # Model eğitimi ve tahmin fonksiyonları
├── main.ipynb                 # Ana notebook (tüm pipeline)
└── requirements.txt           # Gerekli kütüphaneler
```

## 🔧 Kurulum

```bash
# Virtual environment oluştur
python -m venv venv

# Aktif et (Windows)
venv\Scripts\activate

# Kütüphaneleri yükle
pip install -r requirements.txt
```

## 📊 Kullanım

### 1. Jupyter Notebook'u Başlat

```bash
jupyter notebook main.ipynb
```

### 2. Hücreleri Sırayla Çalıştır

#### Hücre 1: Veri Yükleme
```python
import pandas as pd

train = pd.read_csv("data/train.csv")
product = pd.read_csv("data/product_master.csv")
sub = pd.read_csv("data/submission.csv")
```

#### Hücre 2: Veri Ön İşleme
```python
from feature import fill_missing_values_no_drop
from feature import expand_grid_train_only
from feature import add_advanced_lifecycle_features

# 1. Temizlik
train, product = fill_missing_values_no_drop(train, product)

# 2. Genişletme (Zaman serisi onarımı)
train = expand_grid_train_only(train)

# 3. Özellik Türetme
train = add_advanced_lifecycle_features(train, product)
```

#### Hücre 3: Çapraz Doğrulama
```python
from utils import evaluate_model_cv
from model import train_model

# Model performansını değerlendir
cv_results = evaluate_model_cv(
    train_df=train,
    product_df=product,
    model_func=train_model,
    n_splits=3,
    test_months=12
)
```

#### Hücre 4: Final Model ve Submission
```python
from model import train_and_predict_full
from utils import prepare_submission_format

# Final model eğit
final_predictions = train_and_predict_full(
    train_df=train,
    product_df=product,
    submission_template_path='data/submission.csv'
)

# Submission oluştur
submission_df = prepare_submission_format(
    predictions_df=final_predictions,
    submission_template_path='data/submission.csv'
)

# Kaydet
submission_df.to_csv('data/my_submission.csv', index=False)
```

## 📈 Modüller

### `feature.py`
- **fill_missing_values_no_drop()**: Eksik değerleri doldurur, product master'ı temizler
- **expand_grid_train_only()**: Zaman serisi grid'ini genişletir
- **add_advanced_lifecycle_features()**: Lifecycle özellikleri ekler (EOL urgency, life progress, vb.)

### `utils.py`
- **calculate_rwmape()**: Regularized WMAPE hesaplar
- **calculate_group_rwmape()**: Grup bazlı rWMAPE hesaplar
- **calculate_competition_score()**: Yarışma skorunu hesaplar (1/(1+rWMAPE))
- **create_time_series_splits()**: Çapraz doğrulama split'leri oluşturur
- **prepare_submission_format()**: Tahminleri submission formatına dönüştürür
- **evaluate_model_cv()**: Model performansını çapraz doğrulama ile değerlendirir

### `model.py`
- **create_lag_features()**: Gecikmeli özellikler oluşturur
- **create_rolling_features()**: Hareketli ortalama özellikleri oluşturur
- **create_time_features()**: Zaman bazlı özellikler oluşturur
- **prepare_features()**: Tüm özellikleri hazırlar
- **train_model()**: Model eğitir ve tahmin yapar
- **train_and_predict_full()**: Tüm veri ile final model eğitir

## 🎯 Özellikler

### Veri Ön İşleme
- ✅ Eksik tarih doldurma
- ✅ Üretim öncesi satırları temizleme
- ✅ Zaman serisi grid genişletme

### Feature Engineering
- ✅ Lag features (1, 2, 3, 6, 12 ay)
- ✅ Rolling features (3, 6, 12 ay ortalama/std)
- ✅ Zaman özellikleri (ay, çeyrek, sinüs/kosinüs)
- ✅ Lifecycle özellikleri (EOL urgency, life progress)
- ✅ Kategorik encoding (market, category, brand, sector)

### Model
- ✅ LightGBM Regressor (hızlı ve güçlü)
- ✅ Phase-out ürünler için özel mantık
- ✅ EOL geçmiş ürünler için tahmin azaltma

### Değerlendirme
- ✅ Regularized WMAPE (rWMAPE)
- ✅ Grup bazlı skorlama
- ✅ 3-fold çapraz doğrulama
- ✅ Competition score hesaplama

## 📊 Skor Metrikleri

### rWMAPE (Regularized Weighted Mean Absolute Percentage Error)

```
rWMAPE = (Σ|y_true - y_pred| + λ * |Σy_true - Σy_pred|) / (Σy_true + γ * Σy_pred + ε)
```

- **λ (lambda_mass)**: Toplam hacim cezası (0.5)
- **γ (gamma_forecast)**: Küçük tahmin cezası (0.1)
- **ε (epsilon)**: Sayısal kararlılık (1e-10)

### Competition Score

```
Score = 1 / (1 + rWMAPE)
```

- Yüksek skor = daha iyi
- Baseline ≈ 1.0
- Skor > 1 → baseline'dan iyi
- Skor < 1 → baseline'dan kötü

## 🔍 Çıktılar

### Çapraz Doğrulama Çıktısı
```
============================================================
🔄 Çapraz Doğrulama Başlıyor (3 fold)
============================================================

📊 Fold 1/3
   Train: 2022-01 - 2023-10
   Val:   2023-11 - 2024-10
   ✅ rWMAPE: 0.3456
   ✅ Competition Score: 0.7432

...

============================================================
📈 Çapraz Doğrulama Sonuçları
============================================================
Ortalama rWMAPE: 0.3456 (±0.0234)
Ortalama Competition Score: 0.7432 (±0.0156)
============================================================
```

### Submission Dosyası
- **Dosya**: `data/my_submission.csv`
- **Format**: ID, unique_code, date, quantity
- **Satır sayısı**: Submission template ile aynı
- **Doğrulama**: Negatif değer yok, NaN yok

## 💡 İpuçları

1. **Hızlı Test**: İlk önce küçük bir veri alt kümesi ile test edin
2. **Hiperparametre Tuning**: `model.py`'deki LGBMRegressor parametrelerini ayarlayın
3. **Feature Selection**: Önemli özellikleri seçmek için feature importance kullanın
4. **Ensemble**: Farklı modellerin tahminlerini birleştirin
5. **Phase-out Logic**: EOL mantığını iş kurallarına göre ayarlayın

## 🐛 Sorun Giderme

### Memory Error
- Veri boyutunu küçültün veya batch processing kullanın

### Düşük Skor
- Daha fazla feature ekleyin
- Hiperparametreleri optimize edin
- Ensemble yöntemleri deneyin

### Uzun Eğitim Süresi
- n_estimators'ı azaltın
- Daha az feature kullanın
- Veri örneklemesi yapın

## 📞 Destek

Sorularınız için proje README.md dosyasına bakın veya kod içindeki docstring'leri inceleyin.

---

**Başarılar! 🎉**
