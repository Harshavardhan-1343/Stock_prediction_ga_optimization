"""
LSTM Model Architecture
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import numpy as np
import logging
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMModel:
    """
    LSTM Model for stock price prediction
    """
    
    def __init__(self, input_shape: tuple, hyperparameters: dict):
        """
        Initialize LSTM model
        
        Args:
            input_shape: Shape of input data (timesteps, features)
            hyperparameters: Dictionary of hyperparameters from chromosome
        """
        self.input_shape = input_shape
        self.hyperparameters = hyperparameters
        self.model = None
        self.history = None
        
    def build_model(self) -> keras.Model:
        """
        Build LSTM model architecture
        
        Returns:
            Compiled Keras model
        """
        logger.info("🏗️  Building LSTM model...")
        
        # Extract hyperparameters
        n_layers = self.hyperparameters.get('n_lstm_layers', 2)
        neurons_1 = self.hyperparameters.get('neurons_layer1', 128)
        neurons_2 = self.hyperparameters.get('neurons_layer2', 64)
        neurons_3 = self.hyperparameters.get('neurons_layer3', 32)
        dropout = self.hyperparameters.get('dropout_rate', 0.2)
        learning_rate = self.hyperparameters.get('learning_rate', 0.001)
        optimizer_name = self.hyperparameters.get('optimizer', 'adam')
        dense_units = self.hyperparameters.get('dense_units', 32)
        use_bidirectional = self.hyperparameters.get('use_bidirectional', False)
        
        # Input layer
        inputs = layers.Input(shape=self.input_shape)
        x = inputs
        
        # LSTM layers
        neurons_list = [neurons_1, neurons_2, neurons_3]
        
        for i in range(n_layers):
            return_sequences = (i < n_layers - 1)  # Return sequences for all but last layer
            
            if use_bidirectional:
                x = layers.Bidirectional(
                    layers.LSTM(
                        neurons_list[i],
                        return_sequences=return_sequences,
                        activation=config.LSTM_CONFIG['activation'],
                        recurrent_activation=config.LSTM_CONFIG['recurrent_activation']
                    )
                )(x)
            else:
                x = layers.LSTM(
                    neurons_list[i],
                    return_sequences=return_sequences,
                    activation=config.LSTM_CONFIG['activation'],
                    recurrent_activation=config.LSTM_CONFIG['recurrent_activation']
                )(x)
            
            x = layers.Dropout(dropout)(x)
        
        # Dense layers
        x = layers.Dense(dense_units, activation='relu')(x)
        x = layers.Dropout(dropout)(x)
        
        # Output layer
        outputs = layers.Dense(1, activation='linear')(x)
        
        # Create model
        self.model = models.Model(inputs=inputs, outputs=outputs)
        
        # Select optimizer
        if optimizer_name == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = RMSprop(learning_rate=learning_rate)
        elif optimizer_name == 'nadam':
            optimizer = keras.optimizers.Nadam(learning_rate=learning_rate)
        else:
            optimizer = SGD(learning_rate=learning_rate)
        
        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        logger.info(f"✅ Model built successfully!")
        logger.info(f"   Layers: {n_layers}, Neurons: {neurons_list[:n_layers]}")
        logger.info(f"   Optimizer: {optimizer_name}, LR: {learning_rate}")
        logger.info(f"   Dropout: {dropout}, Bidirectional: {use_bidirectional}")
        
        return self.model
    
    def get_model_summary(self) -> str:
        """
        Get model summary as string
        
        Returns:
            Model summary string
        """
        if self.model is None:
            return "Model not built yet"
        
        stringlist = []
        self.model.summary(print_fn=lambda x: stringlist.append(x))
        return "\n".join(stringlist)
    
    def count_parameters(self) -> int:
        """
        Count total trainable parameters
        
        Returns:
            Number of trainable parameters
        """
        if self.model is None:
            return 0
        
        return self.model.count_params()
    
    def get_callbacks(self, monitor: str = 'val_loss') -> list:
        """
        Get training callbacks
        
        Args:
            monitor: Metric to monitor
            
        Returns:
            List of callbacks
        """
        callback_list = [
            callbacks.EarlyStopping(
                monitor=monitor,
                patience=config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor=monitor,
                factor=config.REDUCE_LR_FACTOR,
                patience=config.REDUCE_LR_PATIENCE,
                min_lr=config.MIN_LR,
                verbose=1
            )
        ]
        
        return callback_list
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = None, batch_size: int = None) -> keras.callbacks.History:
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of epochs
            batch_size: Batch size
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        epochs = epochs or config.EPOCHS
        batch_size = batch_size or self.hyperparameters.get('batch_size', 32)
        
        logger.info(f"🚀 Starting training for {epochs} epochs...")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.get_callbacks(),
            verbose=config.VERBOSE
        )
        
        logger.info("✅ Training complete!")
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not built or trained yet")
        
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluate model
        
        Args:
            X: Input features
            y: True targets
            
        Returns:
            Dictionary of metrics
        """
        if self.model is None:
            raise ValueError("Model not built or trained yet")
        
        results = self.model.evaluate(X, y, verbose=0)
        
        return {
            'loss': results[0],
            'mae': results[1],
            'mse': results[2]
        }
    
    def save_model(self, filepath: str):
        """
        Save model to file
        
        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("Model not built yet")
        
        self.model.save(filepath)
        logger.info(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load model from file
        
        Args:
            filepath: Path to model file
        """
        self.model = keras.models.load_model(filepath)
        logger.info(f"📂 Model loaded from {filepath}")


if __name__ == "__main__":
    # Test the LSTM model
    input_shape = (60, 50)  # 60 timesteps, 50 features
    
    hyperparameters = {
        'n_lstm_layers': 2,
        'neurons_layer1': 128,
        'neurons_layer2': 64,
        'neurons_layer3': 32,
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'dense_units': 32,
        'use_bidirectional': False,
        'batch_size': 32
    }
    
    lstm_model = LSTMModel(input_shape, hyperparameters)
    model = lstm_model.build_model()
    
    print("\n📊 Model Summary:")
    print(lstm_model.get_model_summary())
    print(f"\n🔢 Total parameters: {lstm_model.count_parameters():,}")