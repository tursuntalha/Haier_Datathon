import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 1. ADIM: PRODUCT TEMİZLİĞİ (AYNI KALABİLİR + Ufak Revize)
# ---------------------------------------------------------
def fill_missing_values_no_drop(train_df, product_df):
    print("🧹 1. ADIM: Product Temizliği...")
    
    # Tarih dönüşümleri
    for col in ['date']:
        train_df[col] = pd.to_datetime(train_df[col], errors='coerce')
    for col in ['start_production_date', 'end_production_date']:
        product_df[col] = pd.to_datetime(product_df[col], errors='coerce')

    # --- A. Start Date Eksiklerini Satış Verisinden Tamamlama ---
    min_sale_dates = train_df.groupby("product_code")["date"].min().reset_index()
    min_sale_dates.rename(columns={'date': 'first_sale_date'}, inplace=True)

    product_df = pd.merge(product_df, min_sale_dates, on='product_code', how='left')

    # Start date boşsa, ilk satıştan 1 ay öncesini yaz
    product_df['start_production_date'] = product_df['start_production_date'].fillna(
        product_df['first_sale_date'] - pd.DateOffset(months=1)
    )
    
    # --- B. Gruplama ile Doldurma ---
    GROUPING_COLS = ['category', 'brand', 'sector']
    for col in GROUPING_COLS:
        if col in product_df.columns:
            product_df[col] = product_df[col].fillna('Unknown')

    # Kendi grubunun en eski tarihini al
    product_df['start_production_date'] = product_df.groupby(GROUPING_COLS)['start_production_date'].transform(
        lambda x: x.fillna(x.min())
    )

    # Hala boşsa veri setindeki global min tarihi ver (Hardcode yerine dinamik)
    global_min_date = train_df['date'].min()
    product_df['start_production_date'] = product_df['start_production_date'].fillna(global_min_date)

    # --- C. Bayraklar ---
    product_df['is_continuing'] = product_df['end_production_date'].isna().astype(int)
    
    # Gereksiz sütunu at
    product_df.drop(columns=['first_sale_date'], inplace=True)

    return train_df, product_df

# ---------------------------------------------------------
# 2. ADIM: TRAIN GRID GENİŞLETME (STANDART)
# ---------------------------------------------------------
def expand_grid_train_only(train_df):
    print(f"🚀 2. ADIM: Train Grid Genişletme (Giriş: {len(train_df)})...")
    
    train_df['date'] = pd.to_datetime(train_df['date'])
    unique_keys = train_df[['market', 'product_code']].drop_duplicates()
    all_dates = sorted(train_df['date'].unique())
    
    # Cartesian Product
    idx = pd.MultiIndex.from_product(
        [unique_keys.index, all_dates], 
        names=['key_index', 'date']
    )
    
    grid = pd.DataFrame(index=idx).reset_index()
    grid = grid.merge(unique_keys, left_on='key_index', right_index=True).drop(columns=['key_index'])
    
    train_expanded = pd.merge(grid, train_df, on=['market', 'product_code', 'date'], how='left')
    
    # Henüz 0 ile doldurma yapmıyoruz! Önce filtreleme yapacağız.
    # Ancak NaN kalanlar teknik olarak 0 olacak, şimdilik dursun.
    
    return train_expanded

# ---------------------------------------------------------
# 3. ADIM: GELİŞMİŞ FEATURES & PRE-PRODUCTION FİLTRESİ (GÜNCELLENDİ)
# ---------------------------------------------------------
def add_advanced_lifecycle_features(train_df, product_df):
    print("🧬 3. ADIM: Lifecycle Özellikleri & Gürültü Temizliği...")
    
    # Product verilerini ana tabloya ekle (Merge işlemi en başta yapılmalı)
    prod_cols = ['product_code', 'start_production_date', 'end_production_date']
    train_df = train_df.merge(product_df[prod_cols], on='product_code', how='left')
    
    # --- A. PRE-PRODUCTION TEMİZLİĞİ (KRİTİK ADIM) ---
    # Eğer miktar NaN ise (yani grid'den gelme boş satırsa) VE tarih < üretim başlangıcı ise -> SİL
    # Ama eğer o tarihte gerçekten satış varsa (Quantity > 0), veri hatasıdır ama silme, tut.
    
    rows_before = len(train_df)
    
    # Koşul: Quantity NaN (yani satış yok) VE Tarih < Start Date
    drop_condition = (train_df['quantity'].isna()) & (train_df['date'] < train_df['start_production_date'])
    train_df = train_df[~drop_condition].copy()
    
    # Şimdi kalan NaN'ları (yani üretim başladıktan sonraki satışsız ayları) 0 yap
    train_df['quantity'] = train_df['quantity'].fillna(0)
    
    print(f"✂️  Üretim öncesi {rows_before - len(train_df)} adet 'yapay' satır silindi.")

    # --- B. Özellik Mühendisliği ---
    
    # 1. Flag: Üretim Öncesi Satış (Anomali tespiti için)
    train_df['flag_pre_production_sale'] = (train_df['date'] < train_df['start_production_date']).astype(int)

    # Üretim tarihi geçmiş mi? (Yani ürün ölü mü?)
    train_df['flag_eol_passed'] = (train_df['date'] > train_df['end_production_date']).astype(int)

# Eğer flag_eol_passed = 1 ise, modelin satışı 0 tahmin etmesi kolaylaşır.

    # 2. Months Until EOL
    # Not: .dt erişimcisi için sütunların datetime olduğundan emin olalım
    train_df['months_until_eol'] = (train_df['end_production_date'].dt.year - train_df['date'].dt.year) * 12 + \
                                   (train_df['end_production_date'].dt.month - train_df['date'].dt.month)
    
    # NaN (Devam eden ürünler) -> 999
    safe_eol = train_df['months_until_eol'].fillna(999).clip(lower=0)

    # 3. EOL Urgency (Phase-out sinyali)
    train_df['eol_urgency'] = 1 / (safe_eol + 1)
    
    # 4. Life Progress
    total_lifespan = (train_df['end_production_date'].dt.year - train_df['start_production_date'].dt.year) * 12 + \
                     (train_df['end_production_date'].dt.month - train_df['start_production_date'].dt.month)
    
    passed_time = (train_df['date'].dt.year - train_df['start_production_date'].dt.year) * 12 + \
                  (train_df['date'].dt.month - train_df['start_production_date'].dt.month)
                  
    train_df['life_progress'] = passed_time / total_lifespan.replace(0, 1)
    train_df['life_progress'] = train_df['life_progress'].clip(0, 1).fillna(0)

    # Temizlik
    train_df['months_until_eol'] = train_df['months_until_eol'].fillna(999)
    
    # Memory Optimization (İsteğe bağlı ama önerilir)
    # train_df['quantity'] = pd.to_numeric(train_df['quantity'], downcast='integer')
    
    return train_df