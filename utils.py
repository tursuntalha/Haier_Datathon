import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

def calculate_rwmape(y_true, y_pred, lambda_mass=0.5, gamma_forecast=0.1, epsilon=1e-10):
    """
    Regularized Weighted Mean Absolute Percentage Error (rWMAPE)
    
    Parameters:
    -----------
    y_true : array-like
        Gerçek değerler
    y_pred : array-like
        Tahmin edilen değerler
    lambda_mass : float
        Toplam hacim uyumsuzluğu cezası katsayısı
    gamma_forecast : float
        Küçük tahminlere duyarlılık katsayısı
    epsilon : float
        Sayısal kararlılık için küçük sabit
    
    Returns:
    --------
    float : rWMAPE skoru
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Toplam hacim farkı cezası
    mass_penalty = lambda_mass * abs(y_true.sum() - y_pred.sum())
    
    # Pay: Mutlak hatalar toplamı + hacim cezası
    numerator = np.sum(np.abs(y_true - y_pred)) + mass_penalty
    
    # Payda: Gerçek + tahmin toplamı + gamma düzeltmesi
    denominator = np.sum(y_true) + gamma_forecast * np.sum(y_pred) + epsilon
    
    rwmape = numerator / denominator
    
    return rwmape


def calculate_group_rwmape(df, group_col='unique_code', true_col='quantity_true', pred_col='quantity_pred'):
    """
    Grup bazlı rWMAPE hesaplama
    
    Parameters:
    -----------
    df : pd.DataFrame
        Gerçek ve tahmin değerlerini içeren dataframe
    group_col : str
        Gruplama sütunu (örn: unique_code)
    true_col : str
        Gerçek değer sütunu
    pred_col : str
        Tahmin değer sütunu
    
    Returns:
    --------
    float : Ortalama grup rWMAPE skoru
    dict : Grup bazlı detaylı skorlar
    """
    group_scores = []
    group_details = {}
    
    for group_name, group_df in df.groupby(group_col):
        y_true = group_df[true_col].values
        y_pred = group_df[pred_col].values
        
        total_true = y_true.sum()
        total_pred = y_pred.sum()
        
        # Her iki toplam da sıfırsa, bu grubu atla
        if total_true == 0 and total_pred == 0:
            continue
        
        # Gerçek sıfır ama tahmin varsa, ceza = 1.0
        if total_true == 0 and total_pred > 0:
            group_scores.append(1.0)
            group_details[group_name] = {'rwmape': 1.0, 'penalty': True}
            continue
        
        # Normal rWMAPE hesapla
        rwmape = calculate_rwmape(y_true, y_pred)
        group_scores.append(rwmape)
        group_details[group_name] = {'rwmape': rwmape, 'penalty': False}
    
    # Ortalama grup skoru
    avg_score = np.mean(group_scores) if group_scores else 0.0
    
    return avg_score, group_details


def calculate_competition_score(rwmape):
    """
    Yarışma skoru: 1 / (1 + rWMAPE)
    Yüksek skor = daha iyi
    
    Parameters:
    -----------
    rwmape : float
        rWMAPE değeri
    
    Returns:
    --------
    float : Yarışma skoru (0-1 arası, 1'e yakın daha iyi)
    """
    return 1.0 / (1.0 + rwmape)


def create_time_series_splits(train_df, n_splits=3, test_months=12):
    """
    Zaman serisi için çapraz doğrulama split'leri oluştur
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Eğitim verisi (date sütunu olmalı)
    n_splits : int
        Kaç fold oluşturulacak
    test_months : int
        Test seti için kaç ay ayrılacak
    
    Returns:
    --------
    list : (train_dates, val_dates) tuple'larının listesi
    """
    train_df = train_df.copy()
    train_df['date'] = pd.to_datetime(train_df['date'])
    
    all_dates = sorted(train_df['date'].unique())
    
    splits = []
    total_dates = len(all_dates)
    
    # Her split için test_months kadar validation ayır
    for i in range(n_splits):
        # Son tarihten geriye doğru test_months kadar ayır
        val_end_idx = total_dates - (i * test_months)
        val_start_idx = max(0, val_end_idx - test_months)
        
        if val_start_idx >= val_end_idx or val_start_idx < test_months:
            break
        
        train_dates = all_dates[:val_start_idx]
        val_dates = all_dates[val_start_idx:val_end_idx]
        
        if len(train_dates) > 0 and len(val_dates) > 0:
            splits.append((train_dates, val_dates))
    
    return splits


def prepare_submission_format(predictions_df, submission_template_path='data/submission.csv'):
    """
    Tahminleri yarışma formatına dönüştür
    
    Parameters:
    -----------
    predictions_df : pd.DataFrame
        Tahminler (market, product_code, date, quantity sütunları olmalı)
    submission_template_path : str
        Submission template dosya yolu
    
    Returns:
    --------
    pd.DataFrame : Yarışma formatında submission dataframe
    """
    # Template'i yükle
    submission = pd.read_csv(submission_template_path)
    
    # unique_code oluştur
    predictions_df['unique_code'] = predictions_df['market'] + '-' + predictions_df['product_code']
    predictions_df['date'] = pd.to_datetime(predictions_df['date'])
    
    # Template ile merge et
    submission['date'] = pd.to_datetime(submission['date'])
    
    # Tahminleri merge et
    submission = submission.merge(
        predictions_df[['unique_code', 'date', 'quantity']],
        on=['unique_code', 'date'],
        how='left',
        suffixes=('', '_pred')
    )
    
    # Quantity'yi güncelle
    submission['quantity'] = submission['quantity_pred'].fillna(0)
    
    # Negatif değerleri 0 yap
    submission['quantity'] = submission['quantity'].clip(lower=0)
    
    # Sadece gerekli sütunları tut
    submission = submission[['ID', 'unique_code', 'date', 'quantity']]
    
    return submission


def evaluate_model_cv(train_df, product_df, model_func, n_splits=3, test_months=12):
    """
    Model performansını çapraz doğrulama ile değerlendir
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Eğitim verisi
    product_df : pd.DataFrame
        Ürün master verisi
    model_func : callable
        Model eğitim ve tahmin fonksiyonu
        Signature: model_func(train_data, product_data, forecast_dates) -> predictions_df
    n_splits : int
        Kaç fold
    test_months : int
        Test için kaç ay
    
    Returns:
    --------
    dict : CV sonuçları (scores, avg_score, avg_competition_score)
    """
    splits = create_time_series_splits(train_df, n_splits, test_months)
    
    cv_scores = []
    competition_scores = []
    
    print(f"\n{'='*60}")
    print(f"🔄 Çapraz Doğrulama Başlıyor ({n_splits} fold)")
    print(f"{'='*60}\n")
    
    for fold_idx, (train_dates, val_dates) in enumerate(splits, 1):
        print(f"📊 Fold {fold_idx}/{len(splits)}")
        print(f"   Train: {train_dates[0].strftime('%Y-%m')} - {train_dates[-1].strftime('%Y-%m')}")
        print(f"   Val:   {val_dates[0].strftime('%Y-%m')} - {val_dates[-1].strftime('%Y-%m')}")
        
        # Train/Val split
        train_fold = train_df[train_df['date'].isin(train_dates)].copy()
        val_fold = train_df[train_df['date'].isin(val_dates)].copy()
        
        # Model eğit ve tahmin yap
        predictions = model_func(train_fold, product_df, val_dates)
        
        # Değerlendirme için merge
        val_fold['unique_code'] = val_fold['market'] + '-' + val_fold['product_code']
        predictions['unique_code'] = predictions['market'] + '-' + predictions['product_code']
        
        eval_df = val_fold.merge(
            predictions[['unique_code', 'date', 'quantity']],
            on=['unique_code', 'date'],
            how='left',
            suffixes=('_true', '_pred')
        )
        
        eval_df['quantity_pred'] = eval_df['quantity_pred'].fillna(0)
        
        # Skor hesapla
        rwmape, _ = calculate_group_rwmape(eval_df, 'unique_code', 'quantity_true', 'quantity_pred')
        comp_score = calculate_competition_score(rwmape)
        
        cv_scores.append(rwmape)
        competition_scores.append(comp_score)
        
        print(f"   ✅ rWMAPE: {rwmape:.4f}")
        print(f"   ✅ Competition Score: {comp_score:.4f}\n")
    
    avg_rwmape = np.mean(cv_scores)
    avg_comp_score = np.mean(competition_scores)
    
    print(f"{'='*60}")
    print(f"📈 Çapraz Doğrulama Sonuçları")
    print(f"{'='*60}")
    print(f"Ortalama rWMAPE: {avg_rwmape:.4f} (±{np.std(cv_scores):.4f})")
    print(f"Ortalama Competition Score: {avg_comp_score:.4f} (±{np.std(competition_scores):.4f})")
    print(f"{'='*60}\n")
    
    return {
        'fold_rwmapes': cv_scores,
        'fold_competition_scores': competition_scores,
        'avg_rwmape': avg_rwmape,
        'avg_competition_score': avg_comp_score,
        'std_rwmape': np.std(cv_scores),
        'std_competition_score': np.std(competition_scores)
    }
