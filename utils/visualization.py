"""
Visualization utilities for model predictions and GA evolution
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_predictions(y_true, y_pred, title="Model Predictions", save_path=None):
    """
    Plot true vs predicted values
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save plot (optional)
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Time series comparison
    axes[0].plot(y_true, label='Actual', color='blue', alpha=0.7, linewidth=2)
    axes[0].plot(y_pred, label='Predicted', color='red', alpha=0.7, linewidth=2)
    axes[0].set_title(f"{title} - Time Series", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Time Steps", fontsize=12)
    axes[0].set_ylabel("Price", fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot
    axes[1].scatter(y_true, y_pred, alpha=0.5, color='purple')
    axes[1].plot([y_true.min(), y_true.max()], 
                 [y_true.min(), y_true.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[1].set_title(f"{title} - Scatter Plot", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Actual Price", fontsize=12)
    axes[1].set_ylabel("Predicted Price", fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt. savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Plot saved to: {save_path}")
    
    plt.close()


def plot_multiple_predictions(predictions_dict, save_path=None):
    """
    Plot predictions for multiple datasets
    
    Args:
        predictions_dict: Dictionary with dataset names as keys and (y_true, y_pred) tuples as values
        save_path: Path to save plot (optional)
    """
    n_datasets = len(predictions_dict)
    fig, axes = plt.subplots(n_datasets, 1, figsize=(14, 5*n_datasets))
    
    if n_datasets == 1:
        axes = [axes]
    
    for idx, (dataset_name, (y_true, y_pred)) in enumerate(predictions_dict.items()):
        axes[idx].plot(y_true, label='Actual', color='blue', alpha=0.7, linewidth=2)
        axes[idx].plot(y_pred, label='Predicted', color='red', alpha=0.7, linewidth=2)
        axes[idx].set_title(f"{dataset_name. title()} Set Predictions", 
                           fontsize=14, fontweight='bold')
        axes[idx].set_xlabel("Time Steps", fontsize=12)
        axes[idx].set_ylabel("Price", fontsize=12)
        axes[idx].legend(fontsize=11)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Plot saved to: {save_path}")
    
    plt.close()


def plot_ga_evolution(history_df, save_path=None):
    """
    Plot genetic algorithm evolution
    
    Args:
        history_df: DataFrame with GA evolution history
        save_path: Path to save plot (optional)
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Normalize column names (handle both avg_fitness and average_fitness)
    col_mapping = {}
    for col in history_df.columns:
        if 'average' in col.lower() and 'fitness' in col.lower():
            col_mapping['average_fitness'] = col
        elif 'best' in col.lower() and 'fitness' in col.lower():
            col_mapping['best_fitness'] = col
        elif 'worst' in col.lower() and 'fitness' in col.lower():
            col_mapping['worst_fitness'] = col
        elif 'std' in col.lower() and 'fitness' in col.lower():
            col_mapping['std_fitness'] = col
        elif 'generation' in col.lower():
            col_mapping['generation'] = col
    
    # Use actual column names from the dataframe
    gen_col = col_mapping.get('generation', 'generation')
    best_col = col_mapping.get('best_fitness', 'best_fitness')
    avg_col = col_mapping.get('average_fitness', 'average_fitness')
    worst_col = col_mapping.get('worst_fitness', 'worst_fitness')
    std_col = col_mapping.get('std_fitness', 'std_fitness')
    
    # Plot 1: Fitness evolution
    if avg_col in history_df.columns:
        axes[0].plot(history_df[gen_col], history_df[avg_col],
                    label='Average Fitness', color='blue', linewidth=2, marker='o')
    if best_col in history_df.columns:
        axes[0].plot(history_df[gen_col], history_df[best_col],
                    label='Best Fitness', color='green', linewidth=2, marker='s')
    if worst_col in history_df.columns:
        axes[0].plot(history_df[gen_col], history_df[worst_col],
                    label='Worst Fitness', color='red', linewidth=2, marker='^')
    
    axes[0].set_title("Fitness Evolution Across Generations", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Generation", fontsize=12)
    axes[0].set_ylabel("Fitness (Lower is Better)", fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Standard deviation (population diversity)
    if std_col in history_df.columns:
        axes[1].plot(history_df[gen_col], history_df[std_col],label='Fitness Std Dev', color='purple', linewidth=2, marker='d')
        axes[1].set_title("Population Diversity (Fitness Std Dev)", fontsize=14, fontweight='bold')
        axes[1].set_xlabel("Generation", fontsize=12)
        axes[1].set_ylabel("Standard Deviation", fontsize=12)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"📊 GA Evolution plot saved to: {save_path}")
    
    plt.close()


def plot_stock_prices(data_dict, save_path=None):
    """
    Plot stock prices for multiple tickers
    
    Args:
        data_dict: Dictionary with ticker symbols as keys and DataFrames as values
        save_path: Path to save plot (optional)
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_dict)))
    
    for idx, (ticker, df) in enumerate(data_dict. items()):
        if 'date' in df.columns and 'close' in df.columns:
            ax.plot(df['date'], df['close'], 
                   label=ticker, color=colors[idx], linewidth=2)
    
    ax.set_title("Stock Prices Over Time", fontsize=14, fontweight='bold')
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Close Price", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt. xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Stock prices plot saved to: {save_path}")
    
    plt.close()


def plot_training_history(history, save_path=None):
    """
    Plot training history (loss curves)
    
    Args:
        history: Training history dictionary
        save_path: Path to save plot (optional)
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if 'train_loss' in history and 'val_loss' in history:
        epochs = range(1, len(history['train_loss']) + 1)
        
        ax.plot(epochs, history['train_loss'], 
               label='Training Loss', color='blue', linewidth=2)
        ax.plot(epochs, history['val_loss'], 
               label='Validation Loss', color='red', linewidth=2)
        
        ax.set_title("Training and Validation Loss", fontsize=14, fontweight='bold')
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Training history plot saved to: {save_path}")
        
        plt.close()


def plot_feature_importance(importance_dict, top_n=20, save_path=None):
    """
    Plot feature importance
    
    Args:
        importance_dict: Dictionary with feature names as keys and importance scores as values
        top_n: Number of top features to display
        save_path: Path to save plot (optional)
    """
    # Sort by importance
    sorted_features = sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    top_features = sorted_features[:top_n]
    
    features, importances = zip(*top_features)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['green' if x > 0 else 'red' for x in importances]
    ax.barh(range(len(features)), importances, color=colors, alpha=0.7)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Feature importance plot saved to: {save_path}")
    
    plt.close()