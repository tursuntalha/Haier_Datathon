import pandas as pd
import numpy as np
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import warnings
warnings.filterwarnings('ignore')


def prepare_tft_data(train_df, product_df, forecast_horizon=12):
    train = train_df.copy()
    train['date'] = pd.to_datetime(train['date'])
    product = product_df.copy()

    train = train.merge(product[['product_code', 'category', 'brand', 'sector']], on='product_code', how='left')
    for col in ['category', 'brand', 'sector']:
        if col in train.columns:
            train[col] = train[col].fillna('Unknown')

    train['unique_id'] = train['market'] + '_' + train['product_code']
    train = train.sort_values(['unique_id', 'date'])
    train['time_idx'] = train.groupby('unique_id').cumcount()

    train['sales'] = train['quantity'].clip(lower=0)
    train['sales_log'] = np.log1p(train['sales'])

    for col in ['category', 'brand', 'sector', 'market']:
        if col in train.columns:
            train[col] = train[col].astype(str)

    max_encoder_length = max(train.groupby('unique_id').size().min(), 12)
    training_cutoff = train['time_idx'].max() - forecast_horizon

    training = train[train['time_idx'] <= training_cutoff]
    validation = train[train['time_idx'] > training_cutoff]

    if len(training) < 100 or len(validation) < 10:
        return None, None, None, None, None

    try:
        training_dataset = TimeSeriesDataSet(
            training,
            time_idx='time_idx',
            target='sales_log',
            group_ids=['unique_id'],
            max_encoder_length=max_encoder_length,
            max_prediction_length=forecast_horizon,
            static_categoricals=['market', 'category', 'brand', 'sector'],
            time_varying_known_categoricals=[],
            time_varying_known_reals=['time_idx'],
            time_varying_unknown_reals=['sales_log'],
            target_normalizer=GroupNormalizer(groups=['unique_id'], transformation='softplus'),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        validation_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset, validation, predict=False, stop_randomization=True
        )

        train_dataloader = training_dataset.to_dataloader(train=True, batch_size=64, num_workers=0)
        val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

        return training_dataset, validation_dataset, train_dataloader, val_dataloader, training
    except Exception as e:
        print(f"TFT data preparation failed: {e}")
        return None, None, None, None, None


def train_tft(train_dataloader, val_dataloader, training_dataset, max_epochs=30):
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=0.03,
        hidden_size=32,
        attention_head_size=2,
        dropout=0.15,
        hidden_continuous_size=16,
        output_size=7,
        loss=QuantileLoss(),
        reduce_on_plateau_patience=4,
    )

    early_stop_callback = EarlyStopping(monitor='val_loss', patience=5, mode='min')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1)

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator='auto',
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback, checkpoint_callback],
        enable_progress_bar=True,
    )

    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    return tft, trainer


def predict_tft(tft, training_dataset, val_dataloader, forecast_horizon=12):
    best_tft = tft

    raw_predictions = best_tft.predict(val_dataloader, mode='raw', return_x=True)
    x, y_pred = raw_predictions

    predictions_list = []
    for idx in range(len(y_pred.output)):
        pred = y_pred.output[idx, :, :].cpu().numpy()
        median_idx = 3
        pred_median = np.expm1(pred[:, median_idx])
        pred_lower = np.expm1(pred[:, 1])
        pred_upper = np.expm1(pred[:, 5])

        predictions_list.append({
            'prediction': pred_median,
            'lower': pred_lower,
            'upper': pred_upper,
        })

    return predictions_list


def compare_with_lgbm(train_df, product_df, lgbm_predictions):
    print("\n" + "=" * 60)
    print("TFT vs LightGBM Comparison")
    print("=" * 60)

    result = prepare_tft_data(train_df, product_df)
    training_dataset, validation_dataset, train_dataloader, val_dataloader, training = result

    if training_dataset is None:
        print("TFT training data could not be prepared. Data may be insufficient.")
        return None

    print(f"Training samples: {len(training)}")
    print(f"Training dataset prepared. Groups: {len(training_dataset.groups)}")

    tft, trainer = train_tft(train_dataloader, val_dataloader, training_dataset)

    predictions = predict_tft(tft, training_dataset, val_dataloader)

    print("\nTFT Training completed. Compare metrics against LightGBM on same CV folds.")
    print("=" * 60)

    return {
        'model': tft,
        'trainer': trainer,
        'predictions': predictions,
    }
