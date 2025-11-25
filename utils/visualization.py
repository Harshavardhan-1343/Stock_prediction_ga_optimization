"""
Visualization utilities
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import os
import config

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                    title: str = "Predictions vs Actual", 
                    save_path: str = None):
    """
    Plot predictions vs actual values
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Time series plot
    axes[0].plot(y_true.flatten(), label='Actual', color=config.COLORS['actual'], linewidth=2)
    axes[0].plot(y_pred.flatten(), label='Predicted', color=config.COLORS['prediction'], 
                 linewidth=2, alpha=0.7)
    axes[0].set_xlabel('Time Steps', fontsize=12)
    axes[0].set_ylabel('Price (Normalized)', fontsize=12)
    axes[0].set_title(title, fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Scatter plot
    axes[1].scatter(y_true.flatten(), y_pred.flatten(), alpha=0.5, s=30)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    axes[1].set_xlabel('Actual Values', fontsize=12)
    axes[1].set_ylabel('Predicted Values', fontsize=12)
    axes[1].set_title('Scatter Plot: Predicted vs Actual', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        print(f"💾 Plot saved to {save_path}")
    
    plt.show()


def plot_multiple_predictions(predictions_dict: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                             save_path: str = None):
    """
    Plot predictions for multiple datasets (train, val, test)
    
    Args:
        predictions_dict: Dictionary with dataset names and (y_true, y_pred) tuples
        save_path: Path to save plot
    """
    n_datasets = len(predictions_dict)
    fig, axes = plt.subplots(n_datasets, 1, figsize=(14, 5*n_datasets))
    
    if n_datasets == 1:
        axes = [axes]
    
    for ax, (dataset_name, (y_true, y_pred)) in zip(axes, predictions_dict.items()):
        ax.plot(y_true.flatten(), label='Actual', linewidth=2)
        ax.plot(y_pred.flatten(), label='Predicted', linewidth=2, alpha=0.7)
        ax.set_xlabel('Time Steps', fontsize=12)
        ax.set_ylabel('Price (Normalized)', fontsize=12)
        ax.set_title(f'{dataset_name.capitalize()} Set Predictions', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        print(f"💾 Plot saved to {save_path}")
    
    plt.show()


def plot_ga_evolution(history_df: pd.DataFrame, save_path: str = None):
    """
    Plot genetic algorithm evolution
    
    Args:
        history_df: DataFrame with GA history
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Fitness evolution
    axes[0].plot(history_df['generation'], history_df['best_fitness'], 
                marker='o', label='Best Fitness', linewidth=2, markersize=6)
    axes[0].plot(history_df['generation'], history_df['avg_fitness'], 
                marker='s', label='Average Fitness', linewidth=2, markersize=6)
    axes[0].plot(history_df['generation'], history_df['worst_fitness'], 
                marker='^', label='Worst Fitness', linewidth=2, markersize=6)
    
    axes[0].set_xlabel('Generation', fontsize=12)
    axes[0].set_ylabel('Fitness Score', fontsize=12)
    axes[0].set_title('Genetic Algorithm - Fitness Evolution', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Fitness improvement
    fitness_improvement = history_df['best_fitness'].diff()
    axes[1].bar(history_df['generation'], fitness_improvement, alpha=0.7)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Generation', fontsize=12)
    axes[1].set_ylabel('Fitness Improvement', fontsize=12)
    axes[1].set_title('Generation-to-Generation Fitness Improvement', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        print(f"💾 Plot saved to {save_path}")
    
    plt.show()


def plot_training_history(history, save_path: str = None):
    """
    Plot model training history
    
    Args:
        history: Keras training history object
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # MAE
    axes[1].plot(history.history['mae'], label='Training MAE', linewidth=2)
    axes[1].plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        print(f"💾 Plot saved to {save_path}")
    
    plt.show()


def plot_feature_importance(feature_names: List[str], importance_scores: np.ndarray, 
                           top_n: int = 20, save_path: str = None):
    """
    Plot feature importance
    
    Args:
        feature_names: List of feature names
        importance_scores: Array of importance scores
        top_n: Number of top features to show
        save_path: Path to save plot
    """
    # Create DataFrame and sort
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(df)), df['importance'], alpha=0.8)
    plt.yticks(range(len(df)), df['feature'])
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        print(f"💾 Plot saved to {save_path}")
    
    plt.show()


def plot_error_distribution(y_true: np.ndarray, y_pred: np.ndarray, save_path: str = None):
    """
    Plot error distribution
    
    Args:
        y_true: True values
        y_pred: Predicted values
        save_path: Path to save plot
    """
    errors = y_true.flatten() - y_pred.flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(errors, bins=50, alpha=0.7, edgecolor='black')
    axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Prediction Error', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Error Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        print(f"💾 Plot saved to {save_path}")
    
    plt.show()


def plot_correlation_matrix(data: pd.DataFrame, save_path: str = None):
    """
    Plot correlation matrix heatmap
    
    Args:
        data: DataFrame with features
        save_path: Path to save plot
    """
    # Calculate correlation matrix
    corr = data.corr()
    
    # Plot
    plt.figure(figsize=(16, 14))
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        print(f"💾 Plot saved to {save_path}")
    
    plt.show()


def plot_stock_prices(data_dict: Dict[str, pd.DataFrame], save_path: str = None):
    """
    Plot stock prices for all banks
    
    Args:
        data_dict: Dictionary with ticker as key and DataFrame as value
        save_path: Path to save plot
    """
    plt.figure(figsize=(14, 8))
    
    for ticker, df in data_dict.items():
        plt.plot(df['date'], df['close'], label=ticker.replace('.NS', ''), linewidth=2)
    
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Close Price', fontsize=12)
    plt.title('Stock Price History - Indian Banks', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        print(f"💾 Plot saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Test visualizations
    print("📊 Testing Visualizations...")
    
    # Generate sample data
    np.random.seed(42)
    y_true = np.random.rand(100, 1)
    y_pred = y_true + np.random.normal(0, 0.1, (100, 1))
    
    # Test predictions plot
    plot_predictions(y_true, y_pred, title="Test Predictions")