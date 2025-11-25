"""
Model Trainer with Training Pipeline
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
import logging
import time
import os
from .lstm_model import LSTMModel
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Handles the complete model training pipeline
    """
    
    def __init__(self, hyperparameters: dict):
        """
        Initialize trainer
        
        Args:
            hyperparameters: Dictionary of hyperparameters
        """
        self.hyperparameters = hyperparameters
        self.model = None
        self.training_time = 0
        self.metrics = {}
        
    def prepare_data(self, train_data: Tuple, val_data: Tuple, test_data: Tuple,
                    lookback: int) -> Dict:
        """
        Prepare data for training
        
        Args:
            train_data: Tuple of (features, target) for training
            val_data: Tuple of (features, target) for validation
            test_data: Tuple of (features, target) for testing
            lookback: Lookback window
            
        Returns:
            Dictionary with prepared datasets
        """
        logger.info("📦 Preparing data for training...")
        
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data
        
        # Create sequences
        from data.data_preprocessor import DataPreprocessor
        preprocessor = DataPreprocessor()
        
        X_train_seq, y_train_seq = preprocessor.create_sequences(X_train, y_train, lookback)
        X_val_seq, y_val_seq = preprocessor.create_sequences(X_val, y_val, lookback)
        X_test_seq, y_test_seq = preprocessor.create_sequences(X_test, y_test, lookback)
        
        logger.info(f"✅ Data prepared!")
        logger.info(f"   Train: {X_train_seq.shape}, Val: {X_val_seq.shape}, Test: {X_test_seq.shape}")
        
        return {
            'X_train': X_train_seq,
            'y_train': y_train_seq,
            'X_val': X_val_seq,
            'y_val': y_val_seq,
            'X_test': X_test_seq,
            'y_test': y_test_seq
        }
    
    def train_model(self, data_dict: Dict) -> LSTMModel:
        """
        Train LSTM model
        
        Args:
            data_dict: Dictionary with prepared datasets
            
        Returns:
            Trained LSTM model
        """
        logger.info("🚀 Starting model training...")
        
        # Get input shape
        input_shape = (data_dict['X_train'].shape[1], data_dict['X_train'].shape[2])
        
        # Create model
        self.model = LSTMModel(input_shape, self.hyperparameters)
        self.model.build_model()
        
        # Train model
        start_time = time.time()
        
        history = self.model.train(
            data_dict['X_train'],
            data_dict['y_train'],
            data_dict['X_val'],
            data_dict['y_val'],
            epochs=config.EPOCHS,
            batch_size=self.hyperparameters.get('batch_size', 32)
        )
        
        self.training_time = time.time() - start_time
        
        logger.info(f"✅ Training complete in {self.training_time:.2f} seconds")
        
        return self.model
    
    def evaluate_model(self, data_dict: Dict) -> Dict:
        """
        Evaluate trained model
        
        Args:
            data_dict: Dictionary with prepared datasets
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        logger.info("📊 Evaluating model...")
        
        # Make predictions
        y_train_pred = self.model.predict(data_dict['X_train'])
        y_val_pred = self.model.predict(data_dict['X_val'])
        y_test_pred = self.model.predict(data_dict['X_test'])
        
        # Calculate metrics
        from utils.metrics import calculate_metrics
        
        train_metrics = calculate_metrics(data_dict['y_train'], y_train_pred)
        val_metrics = calculate_metrics(data_dict['y_val'], y_val_pred)
        test_metrics = calculate_metrics(data_dict['y_test'], y_test_pred)
        
        self.metrics = {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics,
            'training_time': self.training_time,
            'total_params': self.model.count_parameters()
        }
        
        logger.info("✅ Evaluation complete!")
        logger.info(f"   Val RMSE: {val_metrics['rmse']:.4f}")
        logger.info(f"   Val MAE: {val_metrics['mae']:.4f}")
        logger.info(f"   Val R²: {val_metrics['r2_score']:.4f}")
        
        return self.metrics
    
    def get_predictions(self, data_dict: Dict) -> Dict:
        """
        Get predictions for all datasets
        
        Args:
            data_dict: Dictionary with prepared datasets
            
        Returns:
            Dictionary with predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        predictions = {
            'train': self.model.predict(data_dict['X_train']),
            'val': self.model.predict(data_dict['X_val']),
            'test': self.model.predict(data_dict['X_test'])
        }
        
        return predictions
    
    def save_model(self, filename: str):
        """
        Save trained model
        
        Args:
            filename: Filename to save model
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        filepath = os.path.join(config.MODELS_DIR, filename)
        self.model.save_model(filepath)
        
    def load_model(self, filename: str):
        """
        Load saved model
        
        Args:
            filename: Filename to load model from
        """
        filepath = os.path.join(config.MODELS_DIR, filename)
        
        # Create a dummy model first
        input_shape = (self.hyperparameters.get('lookback_window', 60), 50)
        self.model = LSTMModel(input_shape, self.hyperparameters)
        self.model.load_model(filepath)


if __name__ == "__main__":
    # Test the trainer
    hyperparameters = {
        'n_lstm_layers': 2,
        'neurons_layer1': 128,
        'neurons_layer2': 64,
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'batch_size': 32,
        'lookback_window': 60
    }
    
    trainer = ModelTrainer(hyperparameters)
    print("✅ Trainer initialized successfully!")