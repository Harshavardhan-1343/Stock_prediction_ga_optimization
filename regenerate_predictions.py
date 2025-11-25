"""
Regenerate optimized model predictions from best GA chromosome
"""

import json
import torch
import numpy as np
import os
import logging
import sys

logging.basicConfig(level=logging. INFO)
logger = logging.getLogger(__name__)

try:
    import config
    results_dir = config.RESULTS_DIR
except:
    results_dir = 'results'

from data.data_collector import StockDataCollector
from data.data_preprocessor import DataPreprocessor
from models.model_trainer_torch import ModelTrainer
from utils.visualization import plot_predictions, plot_multiple_predictions

logger.info("="*80)
logger.info("🔄 REGENERATING OPTIMIZED MODEL PREDICTIONS")
logger.info("="*80)

# Load GA results
ga_results_file = os.path.join(results_dir, 'ga_results_20251126_010310.json')

logger.info(f"\n📂 Loading GA results from: {ga_results_file}")

with open(ga_results_file, 'r') as f:
    results = json.load(f)

best_chromosome = results['best_chromosome']

logger.info("\n🏆 Best Hyperparameters:")
for key, value in best_chromosome['genes']. items():
    logger.info(f"   {key}: {value}")

# Load and preprocess data
logger.info("\n📦 Loading and preprocessing data...")

tickers = config.BANK_TICKERS
collector = StockDataCollector(tickers)
raw_data = collector.load_raw_data()

ticker = tickers[0]  # Use first ticker
logger.info(f"   Using ticker: {ticker}")

preprocessor = DataPreprocessor()
df = raw_data[ticker]. copy()
processed_df = preprocessor.preprocess_data(df)

# Split data
train_df, val_df, test_df = preprocessor.split_data(processed_df)
train_df, val_df, test_df, scalers = preprocessor.scale_features(
    train_df, val_df, test_df, target_col='close'
)

# Prepare data
feature_cols = [col for col in train_df.columns 
               if col not in ['date', 'ticker', 'close']]

X_train = train_df[feature_cols]. values
y_train = train_df['close'].values
X_val = val_df[feature_cols].values
y_val = val_df['close'].values
X_test = test_df[feature_cols].values
y_test = test_df['close']. values

# Create trainer with best hyperparameters
logger.info("\n🎓 Training model with best hyperparameters...")

hyperparameters = best_chromosome['genes']. copy()
hyperparameters['epochs'] = 100

trainer = ModelTrainer(hyperparameters)

lookback = best_chromosome['genes']['lookback_window']
prepared_data = trainer.prepare_data(
    (X_train, y_train),
    (X_val, y_val),
    (X_test, y_test),
    lookback
)

# Train model
trainer.train_model(prepared_data, verbose=1)

# Get predictions
logger.info("\n📊 Generating predictions...")
predictions = trainer.get_predictions(prepared_data)

# Inverse transform
target_scaler = scalers['target']

y_train_true = target_scaler.inverse_transform(
    prepared_data['y_train'].cpu().numpy(). reshape(-1, 1)
)
y_train_pred = target_scaler. inverse_transform(predictions['train'])

y_val_true = target_scaler.inverse_transform(
    prepared_data['y_val'].cpu().numpy().reshape(-1, 1)
)
y_val_pred = target_scaler.inverse_transform(predictions['val'])

y_test_true = target_scaler.inverse_transform(
    prepared_data['y_test'].cpu().numpy().reshape(-1, 1)
)
y_test_pred = target_scaler.inverse_transform(predictions['test'])

# Calculate metrics for test set
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

rmse = np.sqrt(mean_squared_error(y_test_true, y_test_pred))
mae = mean_absolute_error(y_test_true, y_test_pred)
r2 = r2_score(y_test_true, y_test_pred)
mape = np.mean(np.abs((y_test_true - y_test_pred) / (y_test_true + 1e-10))) * 100

logger.info("\n" + "="*80)
logger.info("📊 TEST SET PERFORMANCE METRICS")
logger.info("="*80)
logger.info(f"   RMSE:  {rmse:.6f}")
logger.info(f"   MAE:   {mae:.6f}")
logger.info(f"   R²:    {r2:.6f}")
logger.info(f"   MAPE:  {mape:.2f}%")
logger. info("="*80)

# Generate plots
logger.info("\n📊 Generating prediction plots...")

# Plot 1: All datasets
predictions_dict = {
    'train': (y_train_true, y_train_pred),
    'validation': (y_val_true, y_val_pred),
    'test': (y_test_true, y_test_pred)
}

plot_path_all = os.path.join(results_dir, 'optimized_predictions_regenerated.png')
plot_multiple_predictions(predictions_dict, save_path=plot_path_all)
logger.info(f"✅ Saved: {plot_path_all}")

# Plot 2: Test set only
plot_path_test = os.path.join(results_dir, 'optimized_test_predictions_regenerated.png')
plot_predictions(y_test_true, y_test_pred, 
                title="Optimized Model - Test Set Predictions",
                save_path=plot_path_test)
logger.info(f"✅ Saved: {plot_path_test}")

# Clean up GPU
if torch.cuda.is_available():
    torch.cuda.empty_cache()

logger.info("\n" + "="*80)
logger.info("✅ PREDICTIONS REGENERATED SUCCESSFULLY!")
logger.info("="*80)
logger.info(f"\n📁 New plots saved in: {os.path.abspath(results_dir)}")
logger.info("   - optimized_predictions_regenerated.png (all datasets)")
logger.info("   - optimized_test_predictions_regenerated.png (test only)")