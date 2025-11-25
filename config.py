"""
Configuration file for Stock Prediction with GA Optimization
"""

import os
from datetime import datetime, timedelta

# ==================== PROJECT SETTINGS ====================
PROJECT_NAME = "Stock Prediction with GA Optimization"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Create directories if they don't exist
for directory in [DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ==================== DATA SETTINGS ====================
# Indian Banks
BANK_TICKERS = [
    'HDFCBANK.NS',    # HDFC Bank
    'ICICIBANK.NS',   # ICICI Bank
    'SBIN.NS',        # State Bank of India
    'AXISBANK.NS',    # Axis Bank
    'KOTAKBANK.NS'    # Kotak Mahindra Bank
]

# Date Range (8 years)
END_DATE = datetime.now().strftime('%Y-%m-%d')
START_DATE = (datetime.now() - timedelta(days=8*365)).strftime('%Y-%m-%d')

# Data Split Ratios
TRAIN_RATIO = 0.70  # 5.6 years
VAL_RATIO = 0.15    # 1.2 years
TEST_RATIO = 0.15   # 1.2 years

# ==================== FEATURE ENGINEERING ====================
# Technical Indicators
INDICATORS = {
    'SMA': [5, 10, 20, 50],           # Simple Moving Averages
    'EMA': [12, 26],                   # Exponential Moving Averages
    'RSI': [14],                       # Relative Strength Index
    'MACD': True,                      # MACD indicator
    'BOLLINGER': [20, 2],              # Bollinger Bands (period, std)
    'STOCHASTIC': [14, 3],             # Stochastic Oscillator
    'ATR': [14],                       # Average True Range
    'ADX': [14],                       # Average Directional Index
    'CCI': [20],                       # Commodity Channel Index
    'WILLIAMS_R': [14]                 # Williams %R
}

# Lag Features
LAG_PERIODS = [1, 5, 10, 20]

# Rolling Statistics Windows
ROLLING_WINDOWS = [7, 14, 30]

# ==================== MODEL SETTINGS ====================
# LSTM Model Configuration
LSTM_CONFIG = {
    'activation': 'tanh',
    'recurrent_activation': 'sigmoid',
    'return_sequences_middle': True,
    'return_sequences_last': False
}

# Training Settings
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-7
VERBOSE = 1

# ==================== GENETIC ALGORITHM SETTINGS ====================
GA_CONFIG = {
    'population_size': 25,
    'generations': 20,
    'crossover_rate': 0.8,
    'mutation_rate': 0.15,
    'tournament_size': 3,
    'elitism_count': 2,
    'max_workers': 4  # Parallel processing
}

# Chromosome Genes (Search Space)
GENE_SPACE = {
    'n_lstm_layers': [1, 2, 3],
    'neurons_layer1': [32, 64, 128, 256],
    'neurons_layer2': [32, 64, 128, 256],
    'neurons_layer3': [32, 64, 128, 256],
    'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01],
    'batch_size': [16, 32, 64, 128],
    'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
    'lookback_window': [30, 60, 90, 120],
    'optimizer': ['adam', 'rmsprop', 'nadam'],
    'dense_units': [16, 32, 64],
    'use_bidirectional': [True, False]
}

# Fitness Function Weights
FITNESS_WEIGHTS = {
    'rmse_weight': 0.40,
    'mae_weight': 0.30,
    'r2_weight': 0.20,
    'complexity_weight': 0.10
}

# ==================== VISUALIZATION SETTINGS ====================
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
FIGSIZE = (14, 8)
DPI = 100

COLORS = {
    'train': '#1f77b4',
    'val': '#ff7f0e',
    'test': '#2ca02c',
    'prediction': '#d62728',
    'actual': '#9467bd'
}

# ==================== LOGGING SETTINGS ====================
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = os.path.join(LOGS_DIR, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

# ==================== SEED FOR REPRODUCIBILITY ====================
RANDOM_SEED = 42

# ==================== EVALUATION METRICS ====================
METRICS_TO_TRACK = [
    'rmse',
    'mae',
    'mape',
    'r2_score',
    'directional_accuracy',
    'max_error',
    'explained_variance'
]

# ==================== ADVANCED SETTINGS ====================
# GPU Settings
USE_GPU = True
GPU_MEMORY_LIMIT = 4096  # MB

# Mixed Precision Training
USE_MIXED_PRECISION = False

# Model Checkpointing
SAVE_BEST_ONLY = True
SAVE_WEIGHTS_ONLY = False

# Early Stopping
MONITOR_METRIC = 'val_loss'
MONITOR_MODE = 'min'

print(f"✅ Configuration loaded successfully!")
print(f"📅 Data Range: {START_DATE} to {END_DATE}")
print(f"🏦 Banks: {', '.join(BANK_TICKERS)}")