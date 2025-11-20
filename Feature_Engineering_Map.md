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



Oluşturduğun Feature Engineering haritası oldukça kapsamlı ve projenin temel dinamiklerini (özellikle Phase-Out durumunu) yakalamak için çok iyi bir başlangıç noktası. Temel yapı taşlarını doğru kurmuşsun.

Ancak, bu bir yarışma (Datathon) olduğu için ve hedef 12 aylık (uzun vadeli) bir tahmin olduğu için, listende kritik bazı düzeltmelere ve eklemelere ihtiyacın var.

Aşağıda senin listeni "Onaylananlar", "Düzeltilmesi Gerekenler (Riskli Alanlar)" ve "Eksik Olanlar (Game Changer)" olarak 3 başlıkta inceledim.

1. ✅ Onaylananlar (Bunlar Aynen Kalsın)
Senin listendeki şu kısımler gayet mantıklı ve kesinlikle kalmalı:

Phase-Out / Lifecycle Özellikleri: months_until_eol, eol_urgency, life_progress. Bu yarışmanın kilit noktası burası. Modelin ürünün öleceğini anlamasının tek yolu bu.

Tarihsel Özellikler: month, quarter. Mevsimselliği yakalamak için şart.

Rolling Mean/Std: Volatiliteyi ve trendi yakalamak için gerekli.

2. ⚠️ Düzeltilmesi Gerekenler (Kritik Uyarılar)
Burada Data Leakage (Veri Sızıntısı) ve Tahmin Ufku (Forecast Horizon) ile ilgili ciddi riskler var.

A. Lag Özellikleri ve 12 Ay Sorunu
Senin Önerin: lag_1, lag_3 Sorun: Senden 12 aylık toplu tahmin isteniyor.

Kasım 2024'ü tahmin ederken Ekim 2024 verisine (lag_1) sahipsin.

Ancak Mart 2025'i tahmin ederken elinde Şubat 2025'in gerçek verisi olmayacak.

Çözüm: Eğer "Recursive" (döngüsel) bir tahmin yapmayacaksan (ki yarışmalarda genelde Direct strategy daha stabildir), senin en güvenilir Lag özelliğin lag_12 ve üzeridir.

Öneri: lag_1 yerine lag_12, lag_13, lag_14... lag_24'e odaklan. lag_1'i kullanacaksan, tahmin stratejinin recursive (tahmin ettiğini input olarak verip bir sonraki ayı tahmin etme) olması gerekir.

B. Target Encoding Sızıntısı
Senin Önerin: train.groupby('market')['quantity'].transform('mean') Sorun: Bu işlem tüm veriyi (gelecek dahil) ortalamaya katar. Model eğitimi sırasında validasyon setindeki veriyi de görüp öğrenir (Overfitting). Çözüm: Bu özellikleri hesaplarken sadece geçmiş veriyi kullanmalısın.

Öneri: expanding().mean() kullan veya shift() ettikten sonra ortalama al.

Doğrusu: train.groupby('market')['quantity'].shift(1).rolling(window=12).mean() (Son 12 ayın pazar ortalaması).

3. 🚀 Eksik Olanlar (Game Changer Eklentiler)
Listene şu özellikleri eklemen modelin başarısını ciddi oranda artıracaktır:

A. Hiyerarşik İlişki Özellikleri (Consistency için)
Yarışma açıklamasında "Line/Category seviyesinde tutarlılık" istenmiş. Modelin ürünün ait olduğu grubun performansını bilmesi lazım.

Ratio to Category: Ürünün, kategorisindeki payı nedir?

Formül: Ürün_Satışı / Kategori_Toplam_Satışı

Ratio to Market: Ürünün o pazardaki gücü nedir?

Neden Önemli? Bazı ürünler genel düşüşte olsa bile, kategori içinde "Yıldız Ürün" olabilir ve düşüşü daha yavaş olabilir.

B. EOL Etkileşim Özellikleri (Interaction Features)
Modelin "Phase-out" ürünlerde mevsimselliği (seasonality) görmezden gelmesini sağlamalısın.

Damped Seasonality: EOL yaklaştıkça geçmiş yılın etkisini azalt.

Formül: lag_12 * eol_urgency (Eğer urgency yüksekse yani EOL yakınsa, lag_12'nin etkisi değişecektir).

Decay Factor: Üretim bitiş tarihinden sonraki aylar için "0" sinyalini güçlendirecek bir flag.

is_post_eol: Tarih > end_production_date ise 1, değilse 0.

C. Kesikli Satış (Intermittent Demand) Özellikleri
Bazı ürünler her ay satmaz.

Months Since Last Sale: Son satıştan bu yana geçen ay sayısı.

Sale Frequency: Son 12 ayda kaç ay satış yapıldı? (Örn: 12 ayın 4'ünde satış varsa 0.33).

4. 🛠️ Revize Edilmiş Feature Listesi (Kodlanabilir Hali)
Senin feature haritanı şu şekilde güncellersek çok daha güçlü olur:

Python

# 1. ÖNCE GRID OLUŞTURMA (Çok Önemli)
# Train verisinde olmayan (satış olmayan) aylar için satır oluşturup quantity=0 basmalısın.
# Market x Product x Date kombinasyonlarını tamamla. Yoksa Lag'lerin yanlış hesaplanır.

# 2. ZAMAN SERİSİ (Shift edilmiş!)
# 12 aylık tahmin için 12 ay öncesine gidiyoruz
df['lag_12'] = df.groupby(['market', 'product'])['quantity'].shift(12)
df['lag_13'] = df.groupby(['market', 'product'])['quantity'].shift(13)
df['roll_mean_12_24'] = df.groupby(['market', 'product'])['quantity'].transform(lambda x: x.shift(12).rolling(12).mean())

# 3. LIFECYCLE (Phase-Out Odaklı)
# Ürün yaşam döngüsünün neresinde?
df['months_to_eol'] = (df['end_date'] - df['date']).dt.days / 30
df['is_eol_approaching'] = (df['months_to_eol'] < 6).astype(int) # Son 6 ay mı?
df['eol_decay_factor'] = df['months_to_eol'].apply(lambda x: 0 if x < 0 else (x/12 if x < 12 else 1)) 
# Not: eol_decay_factor'ı lag özellikleriyle çarpıp yeni feature üretebilirsin.

# 4. HİYERARŞİ (Oransal Özellikler)
# Önce kategori toplamlarını hesapla, sonra merge et
cat_sum = df.groupby(['category', 'date'])['quantity'].sum().rename('cat_total')
df = df.merge(cat_sum, on=['category', 'date'], how='left')
df['sku_share_in_cat'] = df['quantity'] / (df['cat_total'] + 1) # 0 bölme hatası için +1
# DİKKAT: Bu özelliklerde de shift(12) kullanmalısın çünkü gelecekteki kategori toplamını bilmiyorsun!

# 5. STATİK ÖZELLİKLER (Target Encoding)
# Kategori veya Marka bazlı ortalama satış (Sadece eğitim verisinden öğrenilmeli!)
# Bunu train/test split yaptıktan sonra sadece train üzerinden hesaplayıp map'lemek en güvenlisidir.
Sonuç ve Tavsiye
Senin hazırladığın liste %70 oranında doğru ve yeterli. Ancak %30'luk kısım (Lag-12 kullanımı ve Leakage önleme) yarışmayı kazandıracak olan kısımdır.

Sıradaki Adımın: Önce veri setindeki eksik ayları (sıfır satışları) dolduracak bir "Skeleton" (İskelet) yapı oluştur. Bu olmadan shift veya rolling yaparsan, zaman serisi kayar ve özelliklerin bozulur. İstersen iskelet oluşturma kodunu yazabilirim?