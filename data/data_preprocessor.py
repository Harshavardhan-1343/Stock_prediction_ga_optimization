"""
Data Preprocessor with Feature Engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Tuple, List, Dict
import logging
import ta
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocesses stock data with feature engineering
    """
    
    def __init__(self):
        self.scalers = {}
        self.feature_columns = []
        
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the dataframe
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators
        """
        logger.info("🔧 Adding technical indicators...")
        
        df = df.copy()
        
        # Simple Moving Averages
        for period in config.INDICATORS['SMA']:
            df[f'sma_{period}'] = ta.trend.sma_indicator(df['close'], window=period)
        
        # Exponential Moving Averages
        for period in config.INDICATORS['EMA']:
            df[f'ema_{period}'] = ta.trend.ema_indicator(df['close'], window=period)
        
        # RSI
        for period in config.INDICATORS['RSI']:
            df[f'rsi_{period}'] = ta.momentum.rsi(df['close'], window=period)
        
        # MACD
        if config.INDICATORS['MACD']:
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        period, std = config.INDICATORS['BOLLINGER']
        bollinger = ta.volatility.BollingerBands(df['close'], window=period, window_dev=std)
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_low'] = bollinger.bollinger_lband()
        df['bb_mid'] = bollinger.bollinger_mavg()
        df['bb_width'] = bollinger.bollinger_wband()
        
        # Stochastic Oscillator
        period, smooth = config.INDICATORS['STOCHASTIC']
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], 
                                                  window=period, smooth_window=smooth)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # ATR (Average True Range)
        for period in config.INDICATORS['ATR']:
            df[f'atr_{period}'] = ta.volatility.average_true_range(df['high'], df['low'], 
                                                                    df['close'], window=period)
        
        # ADX (Average Directional Index)
        for period in config.INDICATORS['ADX']:
            df[f'adx_{period}'] = ta.trend.adx(df['high'], df['low'], df['close'], window=period)
        
        # CCI (Commodity Channel Index)
        for period in config.INDICATORS['CCI']:
            df[f'cci_{period}'] = ta.trend.cci(df['high'], df['low'], df['close'], window=period)
        
        # Williams %R
        for period in config.INDICATORS['WILLIAMS_R']:
            df[f'williams_r_{period}'] = ta.momentum.williams_r(df['high'], df['low'], 
                                                                 df['close'], lbp=period)
        
        # OBV (On Balance Volume)
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        
        logger.info(f"✅ Added technical indicators. New shape: {df.shape}")
        return df
    
    def add_lag_features(self, df: pd.DataFrame, target_col: str = 'close') -> pd.DataFrame:
        """
        Add lag features
        
        Args:
            df: Input DataFrame
            target_col: Column to create lags for
            
        Returns:
            DataFrame with lag features
        """
        logger.info("🔧 Adding lag features...")
        
        df = df.copy()
        
        for lag in config.LAG_PERIODS:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        logger.info(f"✅ Added lag features. New shape: {df.shape}")
        return df
    
    def add_rolling_statistics(self, df: pd.DataFrame, target_col: str = 'close') -> pd.DataFrame:
        """
        Add rolling statistics
        
        Args:
            df: Input DataFrame
            target_col: Column to calculate statistics for
            
        Returns:
            DataFrame with rolling statistics
        """
        logger.info("🔧 Adding rolling statistics...")
        
        df = df.copy()
        
        for window in config.ROLLING_WINDOWS:
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
            df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window=window).min()
            df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window=window).max()
        
        # Price momentum
        df['momentum_1'] = df[target_col].pct_change(periods=1)
        df['momentum_5'] = df[target_col].pct_change(periods=5)
        df['momentum_10'] = df[target_col].pct_change(periods=10)
        
        logger.info(f"✅ Added rolling statistics. New shape: {df.shape}")
        return df
    
    def add_inter_bank_features(self, all_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Add inter-bank correlation features
        
        Args:
            all_data: Dictionary of DataFrames for each bank
            
        Returns:
            Updated dictionary with correlation features
        """
        logger.info("🔧 Adding inter-bank correlation features...")
        
        # Get all close prices
        close_prices = {}
        for ticker, df in all_data.items():
            close_prices[ticker] = df.set_index('date')['close']
        
        # Create a combined dataframe
        combined_close = pd.DataFrame(close_prices)
        
        # Calculate rolling correlation with other banks
        for ticker in all_data.keys():
            df = all_data[ticker].copy()
            df = df.set_index('date')
            
            other_banks = [t for t in all_data.keys() if t != ticker]
            
            for other_ticker in other_banks:
                # 30-day rolling correlation
                df[f'corr_{other_ticker.replace(".NS", "")}'] = (
                    combined_close[ticker].rolling(window=30)
                    .corr(combined_close[other_ticker])
                )
            
            df = df.reset_index()
            all_data[ticker] = df
        
        logger.info("✅ Added inter-bank correlation features")
        return all_data
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete preprocessing pipeline
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("🚀 Starting preprocessing pipeline...")
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Add technical indicators
        df = self.add_technical_indicators(df)
        
        # Add lag features
        df = self.add_lag_features(df, 'close')
        
        # Add rolling statistics
        df = self.add_rolling_statistics(df, 'close')
        
        # Drop NaN values (created by indicators and lags)
        initial_len = len(df)
        df = df.dropna()
        logger.info(f"Dropped {initial_len - len(df)} rows with NaN values")
        
        logger.info(f"✅ Preprocessing complete. Final shape: {df.shape}")
        return df
    
    def scale_features(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                      test_df: pd.DataFrame, target_col: str = 'close') -> Tuple:
        """
        Scale features using MinMaxScaler
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            target_col: Target column name
            
        Returns:
            Tuple of scaled dataframes and scalers
        """
        logger.info("🔧 Scaling features...")
        
        # Separate target from features
        feature_cols = [col for col in train_df.columns 
                       if col not in ['date', 'ticker', target_col]]
        
        self.feature_columns = feature_cols
        
        # Feature scaler
        self.scalers['features'] = MinMaxScaler()
        train_df[feature_cols] = self.scalers['features'].fit_transform(train_df[feature_cols])
        val_df[feature_cols] = self.scalers['features'].transform(val_df[feature_cols])
        test_df[feature_cols] = self.scalers['features'].transform(test_df[feature_cols])
        
        # Target scaler
        self.scalers['target'] = MinMaxScaler()
        train_df[[target_col]] = self.scalers['target'].fit_transform(train_df[[target_col]])
        val_df[[target_col]] = self.scalers['target'].transform(val_df[[target_col]])
        test_df[[target_col]] = self.scalers['target'].transform(test_df[[target_col]])
        
        logger.info("✅ Scaling complete")
        return train_df, val_df, test_df, self.scalers
    
    def create_sequences(self, data: np.ndarray, target: np.ndarray, 
                        lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM
        
        Args:
            data: Feature array
            target: Target array
            lookback: Number of timesteps to look back
            
        Returns:
            Tuple of X (sequences) and y (targets)
        """
        X, y = [], []
        
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(target[i])
        
        return np.array(X), np.array(y)
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of train, validation, and test DataFrames
        """
        logger.info("🔧 Splitting data...")
        
        n = len(df)
        train_size = int(n * config.TRAIN_RATIO)
        val_size = int(n * config.VAL_RATIO)
        
        train_df = df.iloc[:train_size].copy()
        val_df = df.iloc[train_size:train_size+val_size].copy()
        test_df = df.iloc[train_size+val_size:].copy()
        
        logger.info(f"✅ Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return train_df, val_df, test_df


if __name__ == "__main__":
    # Test the preprocessor
    from data_collector import StockDataCollector
    
    # Collect data
    collector = StockDataCollector()
    data = collector.fetch_all_data()
    
    # Preprocess one ticker
    preprocessor = DataPreprocessor()
    ticker = config.BANK_TICKERS[0]
    df = data[ticker]
    
    processed_df = preprocessor.preprocess_data(df)
    print(f"\n📊 Processed data shape: {processed_df.shape}")
    print(f"📋 Feature columns: {len(preprocessor.feature_columns)}")
    print(f"Columns: {processed_df.columns.tolist()}")