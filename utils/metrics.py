"""
Utility functions for calculating and displaying model metrics
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import logging

logger = logging.getLogger(__name__)


def calculate_metrics(y_true, y_pred):
    """
    Calculate comprehensive metrics for predictions
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    # Flatten arrays if needed
    y_true = y_true.flatten() if len(y_true.shape) > 1 else y_true
    y_pred = y_pred.flatten() if len(y_pred.shape) > 1 else y_pred
    
    # Basic metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    # Explained variance
    explained_var = explained_variance_score(y_true, y_pred)
    
    # Directional accuracy
    if len(y_true) > 1:
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        directional_accuracy = np.mean(true_direction == pred_direction) * 100
    else:
        directional_accuracy = 0
    
    # Max error
    max_error = np.max(np.abs(y_true - y_pred))
    
    # Mean error (bias)
    mean_error = np.mean(y_pred - y_true)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2_score': r2,
        'explained_variance': explained_var,
        'directional_accuracy': directional_accuracy,
        'max_error': max_error,
        'mean_error': mean_error
    }


def print_metrics(metrics, dataset_name="Dataset"):
    """
    Pretty print metrics
    
    Args:
        metrics: Dictionary of metrics
        dataset_name: Name of the dataset
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"{dataset_name} Metrics:")
    logger.info("=" * 60)
    
    # Print each metric with proper formatting
    if 'rmse' in metrics:
        logger.info(f"  RMSE:                  {metrics['rmse']:.6f}")
    
    if 'mae' in metrics:
        logger.info(f"  MAE:                   {metrics['mae']:.6f}")
    
    if 'mape' in metrics:
        # Cap MAPE display at reasonable values
        mape_val = min(metrics['mape'], 999999)
        logger.info(f"  MAPE:                  {mape_val:.2f}%")
    
    if 'r2_score' in metrics:
        logger.info(f"  R² Score:              {metrics['r2_score']:.6f}")
    
    if 'explained_variance' in metrics:
        logger.info(f"  Explained Variance:    {metrics['explained_variance']:.6f}")
    
    if 'directional_accuracy' in metrics:
        logger.info(f"  Directional Accuracy:  {metrics['directional_accuracy']:.2f}%")
    
    if 'max_error' in metrics:
        logger.info(f"  Max Error:             {metrics['max_error']:.6f}")
    
    if 'mean_error' in metrics:
        logger.info(f"  Mean Error (Bias):     {metrics['mean_error']:.6f}")
    
    logger.info("=" * 60)


def compare_models(models_metrics):
    """
    Compare metrics from multiple models
    
    Args:
        models_metrics: Dictionary with model names as keys and metrics as values
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 80)
    
    # Get all metric names
    all_metrics = set()
    for metrics in models_metrics.values():
        all_metrics.update(metrics.keys())
    
    # Sort metrics for consistent display
    metric_order = ['rmse', 'mae', 'mape', 'r2_score', 'explained_variance', 
                   'directional_accuracy', 'max_error', 'mean_error']
    sorted_metrics = [m for m in metric_order if m in all_metrics]
    
    # Add any remaining metrics not in the order list
    sorted_metrics.extend([m for m in all_metrics if m not in metric_order])
    
    # Create comparison table
    comparison_data = []
    for metric_name in sorted_metrics:
        row = {'Metric': metric_name.replace('_', ' ').title()}
        for model_name, metrics in models_metrics.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                if metric_name in ['mape', 'directional_accuracy']:
                    row[model_name] = f"{value:.2f}%"
                else:
                    row[model_name] = f"{value:.6f}"
            else:
                row[model_name] = "N/A"
        comparison_data.append(row)
    
    # Print as formatted table
    df = pd.DataFrame(comparison_data)
    logger.info("\n" + df.to_string(index=False))
    logger.info("\n" + "=" * 80)


def calculate_percentage_improvement(baseline_metrics, optimized_metrics):
    """
    Calculate percentage improvement from baseline to optimized
    
    Args:
        baseline_metrics: Baseline model metrics
        optimized_metrics: Optimized model metrics
        
    Returns:
        Dictionary of percentage improvements
    """
    improvements = {}
    
    # Metrics where lower is better
    lower_better = ['rmse', 'mae', 'mape', 'max_error', 'mean_error']
    
    # Metrics where higher is better
    higher_better = ['r2_score', 'explained_variance', 'directional_accuracy']
    
    for metric in baseline_metrics.keys():
        if metric not in optimized_metrics:
            continue
        
        baseline_val = baseline_metrics[metric]
        optimized_val = optimized_metrics[metric]
        
        if metric in lower_better:
            # For metrics where lower is better
            if baseline_val != 0:
                improvement = ((baseline_val - optimized_val) / baseline_val) * 100
            else:
                improvement = 0
        elif metric in higher_better:
            # For metrics where higher is better
            if baseline_val != 0:
                improvement = ((optimized_val - baseline_val) / abs(baseline_val)) * 100
            else:
                improvement = 0
        else:
            improvement = 0
        
        improvements[metric] = improvement
    
    return improvements


def print_improvement_analysis(baseline_metrics, optimized_metrics):
    """
    Print detailed improvement analysis
    
    Args:
        baseline_metrics: Baseline model metrics
        optimized_metrics: Optimized model metrics
    """
    improvements = calculate_percentage_improvement(baseline_metrics, optimized_metrics)
    
    logger.info("\n" + "=" * 80)
    logger.info("IMPROVEMENT ANALYSIS")
    logger.info("=" * 80)
    
    # Separate improvements and degradations
    positive_improvements = {k: v for k, v in improvements.items() if v > 0}
    negative_improvements = {k: v for k, v in improvements.items() if v <= 0}
    
    if positive_improvements:
        logger.info("\n✅ IMPROVEMENTS:")
        for metric, improvement in sorted(positive_improvements.items(), 
                                         key=lambda x: x[1], reverse=True):
            metric_name = metric.replace('_', ' ').title()
            logger.info(f"   {metric_name}: +{improvement:.2f}%")
    
    if negative_improvements:
        logger.info("\n❌ DEGRADATIONS:")
        for metric, improvement in sorted(negative_improvements.items(), 
                                         key=lambda x: x[1]):
            metric_name = metric.replace('_', ' ').title()
            logger.info(f"   {metric_name}: {improvement:.2f}%")
    
    # Overall assessment
    if improvements:
        avg_improvement = np.mean(list(improvements.values()))
        logger.info("\n" + "=" * 80)
        logger.info(f"AVERAGE IMPROVEMENT: {avg_improvement:.2f}%")
        
        if avg_improvement > 5:
            logger.info("🎉 OPTIMIZATION WAS HIGHLY SUCCESSFUL!")
        elif avg_improvement > 0:
            logger.info("✅ OPTIMIZATION SHOWED POSITIVE RESULTS")
        else:
            logger.info("⚠️  OPTIMIZATION DID NOT IMPROVE PERFORMANCE")
        
        logger.info("=" * 80)


def create_metrics_dataframe(all_metrics):
    """
    Create a pandas DataFrame from metrics dictionary
    
    Args:
        all_metrics: Dictionary with dataset names as keys and metrics as values
        
    Returns:
        pandas DataFrame
    """
    data = []
    
    for dataset_name, metrics in all_metrics.items():
        row = {'Dataset': dataset_name}
        row.update(metrics)
        data.append(row)
    
    return pd.DataFrame(data)


def save_metrics_to_csv(metrics_dict, filepath):
    """
    Save metrics to CSV file
    
    Args:
        metrics_dict: Dictionary of metrics
        filepath: Path to save CSV file
    """
    df = create_metrics_dataframe(metrics_dict)
    df.to_csv(filepath, index=False)
    logger.info(f"💾 Metrics saved to: {filepath}")


def get_best_model(models_metrics, metric='rmse'):
    """
    Get the best model based on a specific metric
    
    Args:
        models_metrics: Dictionary with model names as keys and metrics as values
        metric: Metric to use for comparison (default: 'rmse')
        
    Returns:
        Name of the best model
    """
    # Metrics where lower is better
    lower_better = ['rmse', 'mae', 'mape', 'max_error', 'mean_error']
    
    if metric in lower_better:
        best_model = min(models_metrics.items(), 
                        key=lambda x: x[1].get(metric, float('inf')))
    else:
        best_model = max(models_metrics.items(), 
                        key=lambda x: x[1].get(metric, float('-inf')))
    
    return best_model[0]


def format_metric_value(metric_name, value):
    """
    Format metric value for display
    
    Args:
        metric_name: Name of the metric
        value: Value of the metric
        
    Returns:
        Formatted string
    """
    if metric_name in ['mape', 'directional_accuracy']:
        return f"{value:.2f}%"
    elif isinstance(value, float):
        return f"{value:.6f}"
    else:
        return str(value)