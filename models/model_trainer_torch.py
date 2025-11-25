"""
PyTorch-based Model Trainer with CUDA support
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
import os
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    """
    LSTM Model for Stock Prediction
    """
    
    def __init__(self, input_size, hyperparameters):
        """
        Initialize LSTM Model
        
        Args:
            input_size: Number of input features
            hyperparameters: Dictionary of hyperparameters
        """
        super(LSTMModel, self).__init__()
        
        self.n_lstm_layers = hyperparameters.get('n_lstm_layers', 2)
        self.neurons_layer1 = hyperparameters.get('neurons_layer1', 128)
        self.neurons_layer2 = hyperparameters.get('neurons_layer2', 64)
        self.neurons_layer3 = hyperparameters.get('neurons_layer3', 32)
        self.dropout_rate = hyperparameters.get('dropout_rate', 0.2)
        self.dense_units = hyperparameters.get('dense_units', 32)
        self.use_bidirectional = hyperparameters.get('use_bidirectional', False)
        
        # LSTM layers
        if self.use_bidirectional:
            self.lstm1 = nn.LSTM(
                input_size, 
                self.neurons_layer1, 
                batch_first=True, 
                bidirectional=True
            )
            lstm1_output_size = self.neurons_layer1 * 2
        else:
            self.lstm1 = nn.LSTM(
                input_size, 
                self.neurons_layer1, 
                batch_first=True
            )
            lstm1_output_size = self.neurons_layer1
        
        self.dropout1 = nn.Dropout(self.dropout_rate)
        
        if self.n_lstm_layers >= 2:
            if self.use_bidirectional:
                self.lstm2 = nn.LSTM(
                    lstm1_output_size,
                    self.neurons_layer2,
                    batch_first=True,
                    bidirectional=True
                )
                lstm2_output_size = self.neurons_layer2 * 2
            else:
                self.lstm2 = nn.LSTM(
                    lstm1_output_size,
                    self.neurons_layer2,
                    batch_first=True
                )
                lstm2_output_size = self.neurons_layer2
            
            self.dropout2 = nn.Dropout(self.dropout_rate)
        else:
            lstm2_output_size = lstm1_output_size
        
        if self.n_lstm_layers >= 3:
            self.lstm3 = nn.LSTM(
                lstm2_output_size,
                self.neurons_layer3,
                batch_first=True
            )
            lstm3_output_size = self.neurons_layer3
            self.dropout3 = nn.Dropout(self.dropout_rate)
        else:
            lstm3_output_size = lstm2_output_size
        
        # Dense layers
        self.fc1 = nn.Linear(lstm3_output_size, self.dense_units)
        self.relu = nn.ReLU()
        self.dropout_dense = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(self.dense_units, 1)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            
        Returns:
            Output tensor
        """
        # LSTM layer 1
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        
        # LSTM layer 2
        if self.n_lstm_layers >= 2:
            out, _ = self.lstm2(out)
            out = self.dropout2(out)
        
        # LSTM layer 3
        if self.n_lstm_layers >= 3:
            out, _ = self.lstm3(out)
            out = self.dropout3(out)
        
        # Take the last time step
        out = out[:, -1, :]
        
        # Dense layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout_dense(out)
        out = self.fc2(out)
        
        return out


class ModelTrainer:
    """
    PyTorch-based Model Trainer with CUDA support
    """
    
    def __init__(self, hyperparameters, force_gpu=False):
        """
        Initialize ModelTrainer
        
        Args:
            hyperparameters: Dictionary of model hyperparameters
            force_gpu: If True, will raise error if GPU not available
        """
        self.hyperparameters = hyperparameters
        self.model = None
        self.history = {'train_loss': [], 'val_loss': []}
        
        # Check CUDA availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if force_gpu and not torch.cuda.is_available():
            raise RuntimeError("GPU not available but force_gpu=True")
        
        if torch.cuda.is_available():
            logger.info(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("⚠️  Using CPU")
        
        # Enable cuDNN benchmarking for better performance
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
    
    def create_sequences(self, X, y, lookback):
        """
        Create sequences for LSTM input
        
        Args:
            X: Feature array
            y: Target array
            lookback: Number of timesteps to look back
            
        Returns:
            X_seq, y_seq arrays
        """
        X_seq, y_seq = [], []
        
        for i in range(len(X) - lookback):
            X_seq.append(X[i:i+lookback])
            y_seq.append(y[i+lookback])
        
        return np.array(X_seq), np.array(y_seq)
    
    def prepare_data(self, train_data, val_data, test_data, lookback):
        """
        Prepare data for training
        
        Args:
            train_data: Tuple of (X_train, y_train)
            val_data: Tuple of (X_val, y_val)
            test_data: Tuple of (X_test, y_test)
            lookback: Lookback window
            
        Returns:
            Dictionary with prepared data
        """
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data
        
        # Create sequences
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train, lookback)
        X_val_seq, y_val_seq = self.create_sequences(X_val, y_val, lookback)
        X_test_seq, y_test_seq = self.create_sequences(X_test, y_test, lookback)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_seq)
        y_train_tensor = torch.FloatTensor(y_train_seq).reshape(-1, 1)
        
        X_val_tensor = torch.FloatTensor(X_val_seq)
        y_val_tensor = torch.FloatTensor(y_val_seq).reshape(-1, 1)
        
        X_test_tensor = torch.FloatTensor(X_test_seq)
        y_test_tensor = torch.FloatTensor(y_test_seq).reshape(-1, 1)
        
        return {
            'X_train': X_train_tensor,
            'y_train': y_train_tensor,
            'X_val': X_val_tensor,
            'y_val': y_val_tensor,
            'X_test': X_test_tensor,
            'y_test': y_test_tensor
        }
    
    def train_model(self, prepared_data, epochs=None, verbose=1):
        """
        Train model with GPU acceleration
        
        Args:
            prepared_data: Dictionary with prepared data
            epochs: Number of epochs (optional)
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        X_train = prepared_data['X_train']
        y_train = prepared_data['y_train']
        X_val = prepared_data['X_val']
        y_val = prepared_data['y_val']
        
        # Build model
        input_size = X_train.shape[2]
        self.model = LSTMModel(input_size, self.hyperparameters)
        self.model = self.model.to(self.device)
        
        # Training parameters
        if epochs is None:
            epochs = self.hyperparameters.get('epochs', 100)
        
        batch_size = self.hyperparameters.get('batch_size', 32)
        learning_rate = self.hyperparameters.get('learning_rate', 0.001)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            pin_memory=True if torch.cuda.is_available() else False,
            num_workers=0
        )
        
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True if torch.cuda.is_available() else False,
            num_workers=0
        )
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        
        optimizer_name = self.hyperparameters.get('optimizer', 'adam')
        if optimizer_name == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,
            verbose=False
        )
        
        # Early stopping
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        best_model_state = None
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                # Move data to device
                batch_X = batch_X.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)
                    
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Print progress
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                if torch.cuda.is_available():
                    logger.info(f"   Epoch {epoch+1}/{epochs} - "
                              f"Train Loss: {train_loss:.6f}, "
                              f"Val Loss: {val_loss:.6f}, "
                              f"GPU Memory: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
                else:
                    logger.info(f"   Epoch {epoch+1}/{epochs} - "
                              f"Train Loss: {train_loss:.6f}, "
                              f"Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                
                if patience_counter >= patience:
                    if verbose:
                        logger.info(f"   Early stopping at epoch {epoch+1}")
                    # Restore best model
                    if best_model_state is not None:
                        self.model.load_state_dict(best_model_state)
                    break
        
        return self.history
    
    def evaluate_model(self, prepared_data):
        """
        Evaluate model on all datasets
        
        Args:
            prepared_data: Dictionary with prepared data
            
        Returns:
            Dictionary with metrics for each dataset
        """
        self.model.eval()
        metrics = {}
        
        with torch.no_grad():
            for dataset_name in ['train', 'val', 'test']:
                X = prepared_data[f'X_{dataset_name}'].to(self.device)
                y_true = prepared_data[f'y_{dataset_name}'].cpu().numpy().flatten()
                
                # Predict
                y_pred = self.model(X).cpu().numpy().flatten()
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                
                # MAPE
                mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
                
                # Directional accuracy
                if len(y_true) > 1:
                    true_direction = np.diff(y_true) > 0
                    pred_direction = np.diff(y_pred) > 0
                    directional_accuracy = np.mean(true_direction == pred_direction) * 100
                else:
                    directional_accuracy = 0
                
                metrics[dataset_name] = {
                    'rmse': rmse,
                    'mae': mae,
                    'r2_score': r2,
                    'mape': mape,
                    'directional_accuracy': directional_accuracy,
                    'explained_variance': r2
                }
        
        return metrics
    
    def get_predictions(self, prepared_data):
        """
        Get predictions for all datasets
        
        Args:
            prepared_data: Dictionary with prepared data
            
        Returns:
            Dictionary with predictions
        """
        self.model.eval()
        predictions = {}
        
        with torch.no_grad():
            for dataset_name in ['train', 'val', 'test']:
                X = prepared_data[f'X_{dataset_name}'].to(self.device)
                y_pred = self.model(X).cpu().numpy()
                predictions[dataset_name] = y_pred
        
        return predictions
    
    def save_model(self, filename):
        """
        Save model to file
        
        Args:
            filename: Name of file to save model
        """
        import config
        
        filepath = os.path.join(config.MODELS_DIR, filename)
        
        # Save model state dict and hyperparameters
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'hyperparameters': self.hyperparameters,
            'history': self.history
        }, filepath)
        
        logger.info(f"💾 Model saved to: {filepath}")
    
    def load_model(self, filename, input_size):
        """
        Load model from file
        
        Args:
            filename: Name of file to load model from
            input_size: Input feature size
        """
        import config
        
        filepath = os.path.join(config.MODELS_DIR, filename)
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.hyperparameters = checkpoint['hyperparameters']
        self.history = checkpoint['history']
        
        self.model = LSTMModel(input_size, self.hyperparameters)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"✅ Model loaded from: {filepath}")