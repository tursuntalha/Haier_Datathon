# 🏆 Haier Datathon - Talep Tahmini Projesi

Bu proje, [Kaggle Haier Datathon] yarışması için geliştirilmiş bir talep tahmin (demand forecasting) çözümüdür. Ürün yaşam döngüsü (lifecycle) özelliklerini kullanarak gelecek aylardaki satış miktarlarını tahmin eder.

## 🎯 Yarışma Sonuçları

Bu çözüm yarışma kapsamında aşağıdaki skorları elde etmiştir:

- **Public Leaderboard**: 0.96214
- **Private Leaderboard**: 0.95819

## 📊 Proje Hakkında

Haier Datathon yarışmasında amaç, farklı pazarlardaki ürünlerin gelecek aylardaki satış miktarlarını tahmin etmektir. Bu proje, LightGBM tabanlı bir makine öğrenmesi modeli kullanarak:

- Gecikmeli (lag) özellikler
- Hareketli ortalamalar (rolling features)
- Ürün yaşam döngüsü özellikleri (EOL, lifecycle progress)
- Zaman bazlı özellikler (mevsimsellik, trend)

ile tahminleme yapar.

### 🎯 Metrik: rWMAPE

Yarışmada kullanılan metrik **Regularized Weighted Mean Absolute Percentage Error (rWMAPE)**'dir:

```
rWMAPE = (Σ|gerçek - tahmin| + λ * |Σgerçek - Σtahmin|) / (Σgerçek + γ * Σtahmin)
```

Yarışma skoru: `1 / (1 + rWMAPE)` (yüksek skor = daha iyi)

## 📁 Proje Yapısı

```
haier_datathon/
│
├── data/                      # Veri dosyaları (Kaggle'dan indirilecek)
│   ├── train.csv             # Eğitim verisi (geçmiş satışlar)
│   ├── product.csv           # Ürün master verisi
│   └── submission.csv        # Submission template
│
├── feature.py                # Veri temizleme ve özellik mühendisliği
├── model.py                  # Model eğitimi ve tahmin
├── utils.py                  # Metrik hesaplama ve yardımcı fonksiyonlar
├── main.ipynb                # Jupyter notebook 
│
├── requirements.txt          # Python bağımlılıkları
└── README.md                 # Bu dosya
```

## 🚀 Kurulum

### 1. Gereksinimler

- Python 3.8+
- pip veya conda

### 2. Bağımlılıkları Yükleyin

```bash
# Virtual environment oluşturun (önerilen)
python -m venv venv

# Virtual environment'ı aktif edin
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Bağımlılıkları yükleyin
pip install -r requirements.txt
```

### 3. Veri Setini İndirin

Kaggle'dan veri setini indirip `data/` klasörüne yerleştirin:

1. [Haier Datathon] sayfasına gidin
2. Data sekmesinden şu dosyaları indirin:
   - `train.csv`
   - `product.csv`
   - `submission.csv`
3. İndirilen dosyaları `data/` klasörüne kopyalayın

## 💻 Kullanım

```bash
python main.ipynb
```

Script otomatik olarak:
1. Veriyi yükler ve temizler
2. Özellikleri oluşturur
3. Modeli eğitir
4. Tahminleri yapar
5. `submission.csv` dosyasını oluşturur

## 📦 Modüller

### `feature.py`
Veri temizleme ve özellik mühendisliği:
- `fill_missing_values_no_drop()`: Eksik değerleri doldurur
- `expand_grid_train_only()`: Zaman serisini genişletir
- `add_advanced_lifecycle_features()`: Yaşam döngüsü özellikleri ekler

### `model.py`
Model eğitimi ve tahmin:
- `prepare_features()`: Tüm özellikleri hazırlar
- `train_model()`: LightGBM modelini eğitir
- `train_and_predict_full()`: Tam pipeline'ı çalıştırır

### `utils.py`
Yardımcı fonksiyonlar:
- `calculate_rwmape()`: rWMAPE metriğini hesaplar
- `calculate_group_rwmape()`: Grup bazlı skor hesaplar
- `prepare_submission_format()`: Submission formatına dönüştürür
- `evaluate_model_cv()`: Cross-validation ile model değerlendirir

## 🔧 Özellik Mühendisliği

### 1. Lag Features (Gecikmeli Özellikler)
```python
lag_1, lag_2, lag_3, lag_6, lag_12  # 1, 2, 3, 6, 12 ay önceki satışlar
```

### 2. Rolling Features (Hareketli Ortalamalar)
```python
rolling_mean_3, rolling_mean_6, rolling_mean_12  # 3, 6, 12 aylık ortalamalar
rolling_std_3, rolling_std_6, rolling_std_12     # 3, 6, 12 aylık standart sapmalar
```

### 3. Lifecycle Features (Yaşam Döngüsü)
```python
eol_urgency          # Ürün sonu yaklaşma aciliyeti
life_progress        # Ürün yaşam döngüsü ilerlemesi (0-1)
months_until_eol     # Ürün sonuna kalan ay sayısı
flag_eol_passed      # Ürün sonu geçti mi?
```

### 4. Time Features (Zaman Özellikleri)
```python
month, quarter       # Ay ve çeyrek
month_sin, month_cos # Mevsimsellik için sinüs/kosinüs dönüşümü
```

## 📈 Model Detayları

### LightGBM Hiperparametreleri

```python
n_estimators=200
learning_rate=0.05
max_depth=7
num_leaves=31
min_child_samples=20
subsample=0.8
colsample_bytree=0.8
```

### Özel Kurallar

1. **EOL Geçmiş Ürünler**: Satış %90 azaltılır
2. **EOL'a 3 Ay Kala**: Satış %50 azaltılır
3. **Üretim Öncesi**: Satış 0 olarak işaretlenir

## 📊 Sonuçlar

Model çıktıları:
- `submission.csv`: Kaggle'a yüklenecek tahmin dosyası
- Feature importance tablosu
- Cross-validation skorları (opsiyonel)

## 🤝 Katkıda Bulunma

Bu proje Haier Datathon yarışması için geliştirilmiştir. Geliştirmeler için:

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit edin (`git commit -m 'Add amazing feature'`)
4. Push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📝 Lisans

Bu proje eğitim amaçlıdır ve Haier Datathon yarışması kapsamında geliştirilmiştir.



**Not**: Veri seti Kaggle üzerinden alınmıştır ve telif hakları Haier'e aittir. Ticari kullanım için izin alınmalıdır.
