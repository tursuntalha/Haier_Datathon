# Feature Engineering Haritası - Talep Tahmini Projesi

## 📊 Proje Amacı
Haier Europe ürünleri için 12 aylık talep tahmini (SKU ve kategori seviyesinde), phase-out ürünleri doğru şekilde ele alma.

---

## 🎯 Kritik Özellikler (Priority Order)

### 1️⃣ **ZAMAN SERİSİ ÖZELLİKLERİ** (En Önemli)
Talep tahmini için temel:

| Özellik | Açıklama | Nasıl Çıkarılır |
|---------|----------|-----------------|
| **Trend** | Uzun dönem yükseliş/düşüş | Rolling mean (3-6-12 ay) |
| **Seasonality** | Mevsimsel desen | Ay, çeyrek, yıl içi pattern |
| **Lag Features** | Geçmiş satışlar | quantity_lag_1, lag_3, lag_6, lag_12 |
| **Rolling Mean** | Hareketli ortalama | 3-ay, 6-ay, 12-ay ortalaması |
| **Rolling Std** | Volatilite | Satış değişkenliği |
| **YoY Growth** | Yıl-yıl büyüme | (t - t-12) / t-12 |

**Kod Örneği:**
```python
train['quantity_lag_1'] = train.groupby(['market', 'product_code'])['quantity'].shift(1)
train['quantity_lag_12'] = train.groupby(['market', 'product_code'])['quantity'].shift(12)
train['rolling_mean_3m'] = train.groupby(['market', 'product_code'])['quantity'].transform(
    lambda x: x.rolling(3, min_periods=1).mean()
)
train['yoy_growth'] = (train['quantity'] - train['quantity_lag_12']) / (train['quantity_lag_12'] + 1)
```

---

### 2️⃣ **ÜRÜN LİFESİKL ÖZELLİKLERİ** (Çok Önemli)
Phase-out ürünleri tanımlamak için kritik:

| Özellik | Açıklama | Nasıl Çıkarılır |
|---------|----------|-----------------|
| **Months Until EOL** | EOL'a kalan ay | (end_date - current_date) / 30 |
| **EOL Urgency** | Phase-out aciliyeti | 1 / (months_until_eol + 1) |
| **Life Progress** | Ürün yaşam döngüsü % | (current_date - start_date) / (end_date - start_date) |
| **Days Since Launch** | Piyasada kaç gün | (current_date - start_production_date).days |
| **Is EOL Passed** | EOL geçti mi? | (current_date > end_production_date).astype(int) |
| **Is Continuing** | Devam eden ürün mü? | end_production_date.isna().astype(int) |

**Kod Örneği:**
```python
train['months_until_eol'] = (train['end_production_date'].dt.year - train['date'].dt.year) * 12 + \
                            (train['end_production_date'].dt.month - train['date'].dt.month)
train['eol_urgency'] = 1 / (train['months_until_eol'].clip(lower=0) + 1)
train['life_progress'] = ((train['date'] - train['start_production_date']).dt.days / 
                          (train['end_production_date'] - train['start_production_date']).dt.days).clip(0, 1)
```

---

### 3️⃣ **PAZAR & KATEGORİ ÖZELLİKLERİ** (Önemli)
Pazar dinamiklerini ve ürün gruplarını yakalamak:

| Özellik | Açıklama | Nasıl Çıkarılır |
|---------|----------|-----------------|
| **Market Avg Sales** | Pazarın ortalama satışı | train.groupby('market')['quantity'].mean() |
| **Category Avg Sales** | Kategori ortalaması | train.groupby('category')['quantity'].mean() |
| **Brand Avg Sales** | Marka ortalaması | train.groupby('brand')['quantity'].mean() |
| **Sector Avg Sales** | Sektör ortalaması | train.groupby('sector')['quantity'].mean() |
| **Market Seasonality** | Pazara göre mevsimsellik | Pazarın ay bazında pattern'i |
| **Product in Market** | Ürün pazarda kaç ay? | Pazarında satış yapılan ay sayısı |

**Kod Örneği:**
```python
train['market_avg_sales'] = train.groupby('market')['quantity'].transform('mean')
train['category_avg_sales'] = train.groupby('category')['quantity'].transform('mean')
train['market_seasonality'] = train.groupby(['market', train['date'].dt.month])['quantity'].transform('mean')
```

---

### 4️⃣ **ZAMAN ÖZELLİKLERİ** (Temel)
Takvim tabanlı desenler:

| Özellik | Açıklama | Nasıl Çıkarılır |
|---------|----------|-----------------|
| **Month** | Ay (1-12) | date.dt.month |
| **Quarter** | Çeyrek (1-4) | date.dt.quarter |
| **Day of Year** | Yıl içi gün (1-365) | date.dt.dayofyear |
| **Is Year End** | Yıl sonu mu? (Nov-Dec) | (date.dt.month >= 11).astype(int) |
| **Is Summer** | Yaz mevsimi mi? | (date.dt.month.isin([6,7,8])).astype(int) |
| **Months Since Start** | Başlangıçtan kaç ay | (date - start_production_date).dt.days / 30 |

**Kod Örneği:**
```python
train['month'] = train['date'].dt.month
train['quarter'] = train['date'].dt.quarter
train['is_year_end'] = (train['date'].dt.month >= 11).astype(int)
train['months_since_start'] = (train['date'] - train['start_production_date']).dt.days / 30
```

---

### 5️⃣ **ANOMALI & KALITE ÖZELLİKLERİ** (Ek)
Veri kalitesi ve anomali tespiti:

| Özellik | Açıklama | Nasıl Çıkarılır |
|---------|----------|-----------------|
| **Flag Pre-Production Sale** | Üretim öncesi satış | (date < start_production_date).astype(int) |
| **Zero Sales Streak** | Ardışık 0 satış | Kaç ay 0 satış? |
| **Spike Detection** | Satış sıçraması | quantity > (mean + 2*std) |
| **Missing Data Flag** | Eksik veri | quantity.isna().astype(int) |

---

## 📈 Önerilen Feature Extraction Sırası

```
1. Tarih Özellikleri (month, quarter, etc.)
   ↓
2. Lifecycle Özellikleri (months_until_eol, eol_urgency, etc.)
   ↓
3. Lag & Rolling Features (lag_1, lag_12, rolling_mean_3m, etc.)
   ↓
4. Pazar & Kategori Özellikleri (market_avg, category_avg, etc.)
   ↓
5. Türetilmiş Özellikler (yoy_growth, seasonality_index, etc.)
   ↓
6. Anomali Flagları (pre_production_sale, spike_detection, etc.)
```

---

## 🔧 Hangi Özellikleri Çıkarmalısın?

### ✅ **MUTLAKA ÇIKARMALıSıN** (High Impact)
1. **Lag Features** (lag_1, lag_3, lag_6, lag_12) - Zaman serisi modelleri buna bağlı
2. **Rolling Mean** (3m, 6m, 12m) - Trend ve smoothing
3. **EOL Urgency** - Phase-out ürünleri tanımlamak için kritik
4. **Months Until EOL** - Model EOL'a yaklaşan ürünleri bilmeli
5. **Month & Quarter** - Mevsimsellik
6. **Market & Category Averages** - Baseline tahmin

### ⭐ **ÖNERİLİ** (Medium Impact)
7. **YoY Growth** - Büyüme trendi
8. **Life Progress** - Ürün yaşam döngüsü
9. **Rolling Std** - Volatilite
10. **Days Since Launch** - Ürün maturity

### 💡 **İSTEĞE BAĞLI** (Nice to Have)
11. **Spike Detection** - Anomali tespiti
12. **Zero Sales Streak** - Satış durması
13. **Sector & Brand Averages** - Ek context

---

## 📝 Özet
- **Zaman Serisi**: Lag ve rolling features olmadan model başarısız olur
- **Lifecycle**: Phase-out ürünleri doğru tahmin etmek için şart
- **Pazar Dinamikleri**: Kategoriler arası farkları yakalamak için gerekli
- **Takvim**: Mevsimsellik ve yıl sonu etkilerini modele vermek için

**Başla:** Lag + Rolling + EOL Urgency + Tarih özellikleri ile. Sonra diğerlerini ekle.
