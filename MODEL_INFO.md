# 🤖 Model Bilgileri - LightGBM

## 🚀 Neden LightGBM?

LightGBM (Light Gradient Boosting Machine), Microsoft tarafından geliştirilen, hızlı ve yüksek performanslı bir gradient boosting framework'üdür.

### ✅ Avantajları

1. **Hız**: Gradient Boosting'e göre 10-20x daha hızlı
2. **Bellek Verimliliği**: Daha az RAM kullanır
3. **Doğruluk**: Genellikle daha iyi tahmin performansı
4. **Büyük Veri**: Milyonlarca satır ile çalışabilir
5. **Kategorik Destek**: Kategorik değişkenleri doğrudan işler
6. **Overfitting Kontrolü**: Leaf-wise büyüme stratejisi

### 🎯 Kullanılan Hiperparametreler

```python
lgb.LGBMRegressor(
    n_estimators=200,        # Ağaç sayısı (daha fazla = daha iyi öğrenme)
    learning_rate=0.05,      # Öğrenme hızı (küçük = daha stabil)
    max_depth=7,             # Maksimum ağaç derinliği
    num_leaves=31,           # Yaprak sayısı (2^max_depth - 1)
    min_child_samples=20,    # Yaprakta minimum örnek sayısı
    subsample=0.8,           # Satır örnekleme oranı (overfitting önler)
    colsample_bytree=0.8,    # Sütun örnekleme oranı (overfitting önler)
    random_state=42,         # Tekrarlanabilirlik
    verbose=-1,              # Sessiz mod
    force_col_wise=True      # Sütun bazlı histogram (hızlı)
)
```

## 📊 Hiperparametre Açıklamaları

### `n_estimators` (200)
- **Ne yapar**: Kaç tane ağaç oluşturulacağını belirler
- **Daha yüksek**: Daha iyi öğrenme ama daha yavaş
- **Daha düşük**: Daha hızlı ama underfitting riski
- **Önerilen aralık**: 100-500

### `learning_rate` (0.05)
- **Ne yapar**: Her ağacın katkısını kontrol eder
- **Daha yüksek**: Daha hızlı öğrenme ama overfitting riski
- **Daha düşük**: Daha stabil ama daha fazla ağaç gerekir
- **Önerilen aralık**: 0.01-0.1

### `max_depth` (7)
- **Ne yapar**: Ağaçların maksimum derinliği
- **Daha yüksek**: Daha karmaşık modeller, overfitting riski
- **Daha düşük**: Daha basit modeller, underfitting riski
- **Önerilen aralık**: 3-10

### `num_leaves` (31)
- **Ne yapar**: Her ağaçtaki maksimum yaprak sayısı
- **Formül**: Genellikle 2^max_depth - 1
- **Daha yüksek**: Daha karmaşık modeller
- **Önerilen aralık**: 20-100

### `min_child_samples` (20)
- **Ne yapar**: Bir yaprakta olması gereken minimum örnek sayısı
- **Daha yüksek**: Overfitting'i önler ama underfitting riski
- **Daha düşük**: Daha detaylı öğrenme ama overfitting riski
- **Önerilen aralık**: 10-50

### `subsample` (0.8)
- **Ne yapar**: Her ağaç için kullanılacak satır oranı
- **0.8**: Her ağaç için rastgele %80 satır kullanılır
- **Avantaj**: Overfitting'i önler, model çeşitliliği sağlar
- **Önerilen aralık**: 0.6-1.0

### `colsample_bytree` (0.8)
- **Ne yapar**: Her ağaç için kullanılacak sütun oranı
- **0.8**: Her ağaç için rastgele %80 özellik kullanılır
- **Avantaj**: Overfitting'i önler, feature çeşitliliği
- **Önerilen aralık**: 0.6-1.0

## 🔧 Hiperparametre Tuning Önerileri

### Hızlı Prototipleme
```python
lgb.LGBMRegressor(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=5,
    num_leaves=15
)
```

### Dengeli (Mevcut)
```python
lgb.LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=31
)
```

### Yüksek Performans
```python
lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=10,
    num_leaves=63,
    min_child_samples=10
)
```

## 📈 Feature Importance

LightGBM otomatik olarak feature importance hesaplar. Model eğitiminden sonra:

```python
# En önemli 15 özelliği göster
get_feature_importance(model, feature_names, top_n=15)
```

**Importance Türleri**:
- **Gain**: Özelliğin modele katkısı (default)
- **Split**: Özelliğin kaç kez kullanıldığı

## 🎯 Phase-Out Mantığı

Model tahminlerini yaptıktan sonra, ürün yaşam döngüsüne göre düzeltme yapılır:

```python
# EOL geçmişse
if flag_eol_passed == 1:
    prediction *= 0.1  # %90 azalt

# EOL'a 3 aydan az kaldıysa
elif months_until_eol < 3:
    prediction *= 0.5  # %50 azalt
```

Bu mantık, phase-out ürünlerin satışlarının doğal olarak azalmasını simüle eder.

## 🔍 Model Performansı İzleme

### Çapraz Doğrulama Sonuçları
```
Fold 1: rWMAPE = 0.34, Score = 0.74
Fold 2: rWMAPE = 0.35, Score = 0.73
Fold 3: rWMAPE = 0.33, Score = 0.75
---
Ortalama: rWMAPE = 0.34 (±0.01)
```

### İyi Bir Skor Nedir?
- **rWMAPE < 0.3**: Mükemmel
- **rWMAPE 0.3-0.4**: Çok iyi
- **rWMAPE 0.4-0.5**: İyi
- **rWMAPE 0.5-0.7**: Orta
- **rWMAPE > 0.7**: Zayıf

### Competition Score
- **Score > 1.0**: Baseline'dan iyi
- **Score = 1.0**: Baseline seviyesi
- **Score < 1.0**: Baseline'dan kötü

## 🚀 İleri Seviye Optimizasyonlar

### 1. Optuna ile Hiperparametre Tuning
```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 5, 10),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
    }
    # Model eğit ve skor döndür
    ...

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
```

### 2. Early Stopping
```python
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=False
)
```

### 3. Kategorik Özellikler
```python
model.fit(
    X_train, y_train,
    categorical_feature=['market', 'category', 'brand']
)
```

### 4. Custom Loss Function
```python
def custom_loss(y_true, y_pred):
    # Özel loss fonksiyonu
    ...

model = lgb.LGBMRegressor(objective=custom_loss)
```

## 📚 Kaynaklar

- [LightGBM Dokümantasyon](https://lightgbm.readthedocs.io/)
- [LightGBM Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
- [LightGBM Python API](https://lightgbm.readthedocs.io/en/latest/Python-API.html)

---

**Not**: Hiperparametreleri değiştirirken her zaman çapraz doğrulama ile test edin!
