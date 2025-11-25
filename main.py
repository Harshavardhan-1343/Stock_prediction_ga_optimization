"""
Main execution script for Stock Prediction with GA Optimization
"""

import numpy as np
import pandas as pd
import logging
import sys
import os
from datetime import datetime
import argparse
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
import config
np.random.seed(config.RANDOM_SEED)
import tensorflow as tf
tf.random.set_seed(config.RANDOM_SEED)

from data.data_collector import StockDataCollector
from data.data_preprocessor import DataPreprocessor
from optimization.genetic_algorithm import GeneticAlgorithm
from models.model_trainer import ModelTrainer
from utils.visualization import (plot_predictions, plot_multiple_predictions, 
                                 plot_ga_evolution, plot_stock_prices)
from utils.metrics import print_metrics, compare_models

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Stock Prediction with GA Optimization')
    
    parser.add_argument('--population_size', type=int, default=25,
                       help='Population size for GA')
    parser.add_argument('--generations', type=int, default=20,
                       help='Number of generations for GA')
    parser.add_argument('--banks', nargs='+', default=config.BANK_TICKERS,
                       help='Bank tickers to analyze')
    parser.add_argument('--skip_data_collection', action='store_true',
                       help='Skip data collection and use existing data')
    parser.add_argument('--run_baseline', action='store_true', default=True,
                       help='Run baseline model for comparison (default: True)')
    
    return parser.parse_args()


def collect_data(tickers):
    """
    Collect stock data
    
    Args:
        tickers: List of stock tickers
        
    Returns:
        Dictionary of DataFrames
    """
    logger.info("="*80)
    logger.info("STEP 1: DATA COLLECTION")
    logger.info("="*80)
    
    collector = StockDataCollector(tickers)
    
    # Try to load existing data
    try:
        data = collector.load_raw_data()
        if data and len(data) > 0:
            logger.info("✅ Loaded existing data from files")
            summary = collector.get_data_summary()
            if summary is not None:
                logger.info(f"\n{summary.to_string()}\n")
            return data
    except Exception as e:
        logger.warning(f"Could not load existing data: {str(e)}")
    
    # Fetch new data
    logger.info("📡 Fetching fresh data from Yahoo Finance...")
    data = collector.fetch_all_data()
    
    # Check if data collection was successful
    if not data or len(data) == 0:
        logger.error("❌ Data collection failed! Cannot proceed.")
        logger.info("\n💡 Solutions:")
        logger.info("   1. Check your internet connection")
        logger.info("   2. Wait a few minutes and try again")
        logger.info("   3. Try with a VPN if you're having regional issues")
        logger.info("   4. Verify the ticker symbols are correct")
        raise Exception("Data collection failed - no data retrieved")
    
    # Save data
    collector.save_raw_data()
    
    # Display summary
    summary = collector.get_data_summary()
    if summary is not None:
        logger.info(f"\n{summary.to_string()}\n")
    
    # Plot stock prices
    try:
        plot_stock_prices(data, save_path=os.path.join(config.RESULTS_DIR, 'stock_prices.png'))
    except Exception as e:
        logger.warning(f"Could not plot stock prices: {str(e)}")
    
    return data


def preprocess_data(raw_data, ticker):
    """
    Preprocess data for a specific ticker
    
    Args:
        raw_data: Dictionary of raw data
        ticker: Ticker to preprocess
        
    Returns:
        Tuple of train, val, test DataFrames
    """
    logger.info("="*80)
    logger.info(f"STEP 2: DATA PREPROCESSING - {ticker}")
    logger.info("="*80)
    
    preprocessor = DataPreprocessor()
    
    # Get data for this ticker
    df = raw_data[ticker].copy()
    
    # Preprocess
    processed_df = preprocessor.preprocess_data(df)
    
    logger.info(f"Processed data shape: {processed_df.shape}")
    logger.info(f"Features: {processed_df.shape[1] - 3}")  # -3 for date, ticker, target
    
    # Split data
    train_df, val_df, test_df = preprocessor.split_data(processed_df)
    
    # Scale features
    train_df, val_df, test_df, scalers = preprocessor.scale_features(
        train_df, val_df, test_df, target_col='close'
    )
    
    logger.info(f"✅ Data preprocessing complete!")
    
    return train_df, val_df, test_df, preprocessor, scalers


def prepare_data_for_ga(train_df, val_df, test_df, preprocessor):
    """
    Prepare data for genetic algorithm
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        preprocessor: DataPreprocessor instance
        
    Returns:
        Dictionary with prepared data
    """
    logger.info("="*80)
    logger.info("STEP 3: PREPARING DATA FOR GA")
    logger.info("="*80)
    
    # Get feature columns
    feature_cols = [col for col in train_df.columns 
                   if col not in ['date', 'ticker', 'close']]
    
    # Extract features and target
    X_train = train_df[feature_cols].values
    y_train = train_df['close'].values
    
    X_val = val_df[feature_cols].values
    y_val = val_df['close'].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df['close'].values
    
    data_dict = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }
    
    logger.info(f"✅ Data prepared for GA!")
    logger.info(f"   Features: {X_train.shape[1]}")
    logger.info(f"   Train samples: {X_train.shape[0]}")
    logger.info(f"   Val samples: {X_val.shape[0]}")
    logger.info(f"   Test samples: {X_test.shape[0]}")
    
    return data_dict


def run_baseline_model(data_dict, scalers):
    """
    Run baseline model for comparison
    
    Args:
        data_dict: Dictionary with prepared data
        scalers: Dictionary of scalers
        
    Returns:
        Dictionary with predictions and metrics
    """
    logger.info("\n" + "="*80)
    logger.info("BASELINE MODEL (Before Optimization)")
    logger.info("="*80)
    
    # Simple baseline hyperparameters (commonly used defaults)
    baseline_hyperparameters = {
        'n_lstm_layers': 2,
        'neurons_layer1': 128,
        'neurons_layer2': 64,
        'neurons_layer3': 32,
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'batch_size': 32,
        'lookback_window': 60,
        'dense_units': 32,
        'use_bidirectional': False
    }
    
    logger.info("\n📋 Baseline Hyperparameters:")
    for key, value in baseline_hyperparameters.items():
        logger.info(f"   {key}: {value}")
    
    trainer = ModelTrainer(baseline_hyperparameters)
    
    prepared_data = trainer.prepare_data(
        (data_dict['X_train'], data_dict['y_train']),
        (data_dict['X_val'], data_dict['y_val']),
        (data_dict['X_test'], data_dict['y_test']),
        60
    )
    
    trainer.train_model(prepared_data)
    metrics = trainer.evaluate_model(prepared_data)
    
    logger.info("\n📊 Baseline Model Results:")
    print_metrics(metrics['train'], "Training Set")
    print_metrics(metrics['val'], "Validation Set")
    print_metrics(metrics['test'], "Test Set")
    
    # Get predictions for visualization
    predictions = trainer.get_predictions(prepared_data)
    
    # Inverse transform predictions
    target_scaler = scalers['target']
    
    y_test_true = target_scaler.inverse_transform(prepared_data['y_test'].reshape(-1, 1))
    y_test_pred = target_scaler.inverse_transform(predictions['test'])
    
    # Plot baseline predictions
    plot_predictions(y_test_true, y_test_pred, 
                    title="Baseline Model Predictions (Before Optimization)",
                    save_path=os.path.join(config.RESULTS_DIR, 'baseline_predictions.png'))
    
    # Save baseline model
    trainer.save_model(f"baseline_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5")
    
    return metrics, predictions


def run_genetic_algorithm(data_dict, population_size, generations):
    """
    Run genetic algorithm optimization
    
    Args:
        data_dict: Dictionary with prepared data
        population_size: GA population size
        generations: Number of generations
        
    Returns:
        Genetic Algorithm instance
    """
    logger.info("\n" + "="*80)
    logger.info("GENETIC ALGORITHM OPTIMIZATION")
    logger.info("="*80)
    
    ga = GeneticAlgorithm(data_dict, population_size, generations)
    ga.evolve()
    
    # Save results
    ga.save_results()
    
    # Plot evolution
    history_df = ga.get_history_dataframe()
    plot_ga_evolution(history_df, save_path=os.path.join(config.RESULTS_DIR, 'ga_evolution.png'))
    
    return ga


def evaluate_best_model(ga, data_dict, scalers):
    """
    Evaluate best model from GA
    
    Args:
        ga: Genetic Algorithm instance
        data_dict: Dictionary with prepared data
        scalers: Dictionary of scalers
        
    Returns:
        Dictionary with predictions and metrics
    """
    logger.info("\n" + "="*80)
    logger.info("OPTIMIZED MODEL (After GA Optimization)")
    logger.info("="*80)
    
    # Get best chromosome
    best_chromosome = ga.get_best_chromosome()
    
    logger.info("\n🏆 Best Hyperparameters Found by GA:")
    for key, value in best_chromosome.genes.items():
        logger.info(f"   {key}: {value}")
    
    # Train final model with best hyperparameters
    trainer = ModelTrainer(best_chromosome.get_hyperparameters())
    
    lookback = best_chromosome.genes['lookback_window']
    prepared_data = trainer.prepare_data(
        (data_dict['X_train'], data_dict['y_train']),
        (data_dict['X_val'], data_dict['y_val']),
        (data_dict['X_test'], data_dict['y_test']),
        lookback
    )
    
    trainer.train_model(prepared_data)
    metrics = trainer.evaluate_model(prepared_data)
    
    # Print metrics
    logger.info("\n📊 Optimized Model Results:")
    print_metrics(metrics['train'], "Training Set")
    print_metrics(metrics['val'], "Validation Set")
    print_metrics(metrics['test'], "Test Set")
    
    # Get predictions
    predictions = trainer.get_predictions(prepared_data)
    
    # Inverse transform predictions
    target_scaler = scalers['target']
    
    y_train_true = target_scaler.inverse_transform(prepared_data['y_train'].reshape(-1, 1))
    y_train_pred = target_scaler.inverse_transform(predictions['train'])
    
    y_val_true = target_scaler.inverse_transform(prepared_data['y_val'].reshape(-1, 1))
    y_val_pred = target_scaler.inverse_transform(predictions['val'])
    
    y_test_true = target_scaler.inverse_transform(prepared_data['y_test'].reshape(-1, 1))
    y_test_pred = target_scaler.inverse_transform(predictions['test'])
    
    # Plot predictions
    predictions_dict = {
        'train': (y_train_true, y_train_pred),
        'validation': (y_val_true, y_val_pred),
        'test': (y_test_true, y_test_pred)
    }
    
    plot_multiple_predictions(predictions_dict, 
                             save_path=os.path.join(config.RESULTS_DIR, 'optimized_predictions.png'))
    
    # Plot test set comparison
    plot_predictions(y_test_true, y_test_pred, 
                    title="Optimized Model Predictions (After GA Optimization)",
                    save_path=os.path.join(config.RESULTS_DIR, 'optimized_test_predictions.png'))
    
    # Save best model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trainer.save_model(f"optimized_model_{timestamp}.h5")
    
    return predictions_dict, metrics


def compare_baseline_vs_optimized(baseline_metrics, optimized_metrics):
    """
    Compare baseline vs optimized model performance
    
    Args:
        baseline_metrics: Metrics from baseline model
        optimized_metrics: Metrics from optimized model
    """
    logger.info("\n" + "="*80)
    logger.info("📊 BASELINE vs OPTIMIZED MODEL COMPARISON")
    logger.info("="*80)
    
    # Extract test metrics for comparison
    baseline_test = baseline_metrics['test']
    optimized_test = optimized_metrics['test']
    
    # Create comparison dictionary
    comparison = {
        'Baseline (Before)': baseline_test,
        'Optimized (After)': optimized_test
    }
    
    # Print comparison
    compare_models(comparison)
    
    # Calculate improvements
    logger.info("\n" + "="*80)
    logger.info("📈 IMPROVEMENT ANALYSIS")
    logger.info("="*80)
    
    improvements = {}
    for metric in ['rmse', 'mae', 'mape']:
        baseline_val = baseline_test[metric]
        optimized_val = optimized_test[metric]
        improvement = ((baseline_val - optimized_val) / baseline_val) * 100
        improvements[metric] = improvement
        
        if improvement > 0:
            logger.info(f"✅ {metric.upper()}: Improved by {improvement:.2f}%")
        else:
            logger.info(f"❌ {metric.upper()}: Degraded by {abs(improvement):.2f}%")
    
    for metric in ['r2_score', 'directional_accuracy', 'explained_variance']:
        baseline_val = baseline_test[metric]
        optimized_val = optimized_test[metric]
        improvement = ((optimized_val - baseline_val) / abs(baseline_val)) * 100 if baseline_val != 0 else 0
        improvements[metric] = improvement
        
        if improvement > 0:
            logger.info(f"✅ {metric.upper()}: Improved by {improvement:.2f}%")
        else:
            logger.info(f"❌ {metric.upper()}: Degraded by {abs(improvement):.2f}%")
    
    # Overall verdict
    avg_improvement = np.mean([improvements['rmse'], improvements['mae'], 
                               improvements['r2_score'], improvements['directional_accuracy']])
    
    logger.info("\n" + "="*80)
    if avg_improvement > 5:
        logger.info("🎉 GA OPTIMIZATION WAS HIGHLY SUCCESSFUL!")
        logger.info(f"   Average Improvement: {avg_improvement:.2f}%")
    elif avg_improvement > 0:
        logger.info("✅ GA OPTIMIZATION SHOWED POSITIVE RESULTS")
        logger.info(f"   Average Improvement: {avg_improvement:.2f}%")
    else:
        logger.info("⚠️  GA OPTIMIZATION DID NOT IMPROVE PERFORMANCE")
        logger.info(f"   Average Change: {avg_improvement:.2f}%")
    logger.info("="*80)
    
    # Save comparison to file
    comparison_df = pd.DataFrame({
        'Metric': list(baseline_test.keys()),
        'Baseline': list(baseline_test.values()),
        'Optimized': list(optimized_test.values()),
        'Improvement (%)': [improvements.get(k, 0) for k in baseline_test.keys()]
    })
    
    comparison_file = os.path.join(config.RESULTS_DIR, 
                                   f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    comparison_df.to_csv(comparison_file, index=False)
    logger.info(f"\n💾 Comparison saved to: {comparison_file}")


def main():
    """
    Main execution function
    """
    logger.info("\n" + "="*80)
    logger.info("🚀 STOCK PREDICTION WITH GENETIC ALGORITHM OPTIMIZATION")
    logger.info("="*80 + "\n")
    
    # Parse arguments
    args = parse_arguments()
    
    try:
        # Step 1: Collect Data
        if args.skip_data_collection:
            collector = StockDataCollector(args.banks)
            raw_data = collector.load_raw_data()
        else:
            raw_data = collect_data(args.banks)
        
        # Select first ticker for demonstration
        ticker = args.banks[0]
        logger.info(f"\n🎯 Processing ticker: {ticker}\n")
        
        # Step 2: Preprocess Data
        train_df, val_df, test_df, preprocessor, scalers = preprocess_data(raw_data, ticker)
        
        # Step 3: Prepare Data for GA
        data_dict = prepare_data_for_ga(train_df, val_df, test_df, preprocessor)
        
        # Step 4: Run Baseline Model
        baseline_metrics = None
        if args.run_baseline:
            baseline_metrics, _ = run_baseline_model(data_dict, scalers)
        
        # Step 5: Run Genetic Algorithm
        ga = run_genetic_algorithm(data_dict, args.population_size, args.generations)
        
        # Step 6: Evaluate Best Model
        predictions, optimized_metrics = evaluate_best_model(ga, data_dict, scalers)
        
        # Step 7: Compare Results
        if baseline_metrics:
            compare_baseline_vs_optimized(baseline_metrics, optimized_metrics)
        
        logger.info("\n" + "="*80)
        logger.info("✅ EXECUTION COMPLETED SUCCESSFULLY!")
        logger.info("="*80 + "\n")
        
        logger.info(f"📁 Results saved in: {config.RESULTS_DIR}")
        logger.info(f"💾 Models saved in: {config.MODELS_DIR}")
        logger.info(f"📝 Logs saved in: {config.LOG_FILE}")
        
    except Exception as e:
        logger.error(f"\n❌ ERROR: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()