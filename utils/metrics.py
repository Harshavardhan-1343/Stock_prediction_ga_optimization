"""
Evaluation Metrics
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAE value
    """
    return mean_absolute_error(y_true, y_pred)


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAPE value (as percentage)
    """
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate directional accuracy (percentage of correct direction predictions)
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Directional accuracy (as percentage)
    """
    # Calculate direction of change
    true_direction = np.diff(y_true.flatten()) > 0
    pred_direction = np.diff(y_pred.flatten()) > 0
    
    # Calculate accuracy
    correct = np.sum(true_direction == pred_direction)
    total = len(true_direction)
    
    return (correct / total) * 100


def calculate_max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate maximum absolute error
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Maximum error
    """
    return np.max(np.abs(y_true - y_pred))


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate all evaluation metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with all metrics
    """
    # Flatten arrays if needed
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    metrics = {
        'rmse': calculate_rmse(y_true, y_pred),
        'mae': calculate_mae(y_true, y_pred),
        'mape': calculate_mape(y_true, y_pred),
        'r2_score': r2_score(y_true, y_pred),
        'directional_accuracy': calculate_directional_accuracy(y_true, y_pred),
        'max_error': calculate_max_error(y_true, y_pred),
        'explained_variance': explained_variance_score(y_true, y_pred)
    }
    
    return metrics


def print_metrics(metrics: Dict[str, float], dataset_name: str = "Dataset"):
    """
    Print metrics in a formatted way
    
    Args:
        metrics: Dictionary of metrics
        dataset_name: Name of the dataset
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"{dataset_name} Metrics:")
    logger.info(f"{'='*60}")
    logger.info(f"  RMSE:                  {metrics['rmse']:.6f}")
    logger.info(f"  MAE:                   {metrics['mae']:.6f}")
    logger.info(f"  MAPE:                  {metrics['mape']:.2f}%")
    logger.info(f"  R² Score:              {metrics['r2_score']:.6f}")
    logger.info(f"  Directional Accuracy:  {metrics['directional_accuracy']:.2f}%")
    logger.info(f"  Max Error:             {metrics['max_error']:.6f}")
    logger.info(f"  Explained Variance:    {metrics['explained_variance']:.6f}")
    logger.info(f"{'='*60}\n")


def compare_models(metrics_dict: Dict[str, Dict[str, float]]):
    """
    Compare metrics across multiple models
    
    Args:
        metrics_dict: Dictionary with model names as keys and metrics as values
    """
    import pandas as pd
    
    # Create comparison DataFrame
    comparison = pd.DataFrame(metrics_dict).T
    
    logger.info("\n📊 Model Comparison:")
    logger.info(f"\n{comparison.to_string()}")
    
    # Find best model for each metric
    logger.info("\n🏆 Best Models per Metric:")
    for metric in comparison.columns:
        if metric in ['rmse', 'mae', 'mape', 'max_error']:
            best_model = comparison[metric].idxmin()
            best_value = comparison[metric].min()
        else:
            best_model = comparison[metric].idxmax()
            best_value = comparison[metric].max()
        
        logger.info(f"  {metric}: {best_model} ({best_value:.4f})")


if __name__ == "__main__":
    # Test metrics
    print("📊 Testing Metrics...")
    
    # Generate sample data
    np.random.seed(42)
    y_true = np.random.rand(100, 1)
    y_pred = y_true + np.random.normal(0, 0.1, (100, 1))
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    
    # Print metrics
    print_metrics(metrics, "Test Dataset")