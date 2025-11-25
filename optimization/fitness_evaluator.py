"""
Fitness Evaluator for Genetic Algorithm
Evaluates the performance of LSTM models with different hyperparameters
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)


class FitnessEvaluator:
    """
    Evaluates fitness of chromosomes by training and evaluating LSTM models
    """
    
    def __init__(self, data_dict):
        """
        Initialize Fitness Evaluator
        
        Args:
            data_dict: Dictionary containing train, val, test data
        """
        self.data_dict = data_dict
        self.evaluation_count = 0
    
    def evaluate(self, chromosome):
        """
        Evaluate a chromosome by training a model and calculating fitness
        
        Args:
            chromosome: Chromosome object to evaluate
            
        Returns:
            float: Fitness score (lower is better)
        """
        self.evaluation_count += 1
        
        try:
            # Import here to avoid circular imports
            from models.model_trainer_torch import ModelTrainer
            
            # Get hyperparameters from chromosome
            hyperparameters = chromosome.get_hyperparameters()
            
            # Log evaluation
            logger.info("=" * 80)
            logger.info(f"Evaluating Individual {self.evaluation_count}")
            logger.info("=" * 80)
            logger.info(f"🧬 Evaluating chromosome: {chromosome.genes}")
            
            # Create and train model
            trainer = ModelTrainer(hyperparameters)
            
            # Prepare data with lookback window
            lookback = chromosome.genes['lookback_window']
            prepared_data = trainer.prepare_data(
                (self.data_dict['X_train'], self.data_dict['y_train']),
                (self.data_dict['X_val'], self.data_dict['y_val']),
                (self.data_dict['X_test'], self.data_dict['y_test']),
                lookback
            )
            
            # Train model (reduced epochs for GA speed)
            trainer.train_model(prepared_data, epochs=100, verbose=1)
            
            # Evaluate on validation set
            metrics = trainer.evaluate_model(prepared_data)
            val_metrics = metrics['val']
            
            # Calculate fitness score (weighted combination of metrics)
            # Lower is better, so we want to minimize this
            fitness = (
                0.4 * val_metrics['rmse'] +           # 40% weight on RMSE
                0.3 * val_metrics['mae'] +            # 30% weight on MAE
                0.2 * (1 - val_metrics['r2_score']) + # 20% weight on R2 (inverted)
                0.1 * (1 - val_metrics['directional_accuracy'] / 100)  # 10% weight on direction
            )
            
            # Get model complexity (for logging)
            model_params = self._count_model_parameters(trainer.model)
            
            # Log results
            logger.info(f"📊 Evaluation Results:")
            logger.info(f"   Validation RMSE: {val_metrics['rmse']:.6f}")
            logger.info(f"   Validation MAE: {val_metrics['mae']:.6f}")
            logger.info(f"   Validation R²: {val_metrics['r2_score']:.6f}")
            logger.info(f"   Directional Accuracy: {val_metrics['directional_accuracy']:.2f}%")
            logger.info(f"   Model Parameters: {model_params:,}")
            logger.info(f"   ⭐ Fitness Score: {fitness:.6f}")
            logger.info("=" * 80)
            
            # Store metrics in chromosome
            chromosome.fitness = fitness
            chromosome.metrics = val_metrics
            chromosome.model_params = model_params
            
            # Clean up GPU memory if using CUDA
            import torch
            if torch.cuda.is_available():
                del trainer
                torch.cuda.empty_cache()
            
            return fitness
            
        except Exception as e:
            logger.error(f"❌ Error evaluating chromosome: {e}")
            logger.error(f"   Chromosome genes: {chromosome.genes}")
            
            # Return a very high fitness score (bad fitness) on error
            # This ensures failed chromosomes are not selected
            chromosome.fitness = float('inf')
            chromosome.metrics = None
            chromosome.model_params = 0
            
            return float('inf')
    
    def _count_model_parameters(self, model):
        """
        Count the number of parameters in a PyTorch model
        
        Args:
            model: PyTorch model
            
        Returns:
            int: Number of parameters
        """
        try:
            import torch
            if isinstance(model, torch.nn.Module):
                return sum(p.numel() for p in model.parameters())
            else:
                return 0
        except Exception as e:
            logger.warning(f"Could not count model parameters: {e}")
            return 0
    
    def evaluate_final(self, chromosome):
        """
        Final evaluation on test set with full training
        
        Args:
            chromosome: Chromosome to evaluate
            
        Returns:
            dict: Complete metrics for train, val, and test sets
        """
        try:
            from models.model_trainer_torch import ModelTrainer
            
            hyperparameters = chromosome.get_hyperparameters()
            
            logger.info("=" * 80)
            logger.info("FINAL EVALUATION")
            logger.info("=" * 80)
            logger.info(f"🏆 Evaluating best chromosome: {chromosome.genes}")
            
            # Create and train model
            trainer = ModelTrainer(hyperparameters)
            
            # Prepare data
            lookback = chromosome.genes['lookback_window']
            prepared_data = trainer.prepare_data(
                (self.data_dict['X_train'], self.data_dict['y_train']),
                (self.data_dict['X_val'], self.data_dict['y_val']),
                (self.data_dict['X_test'], self.data_dict['y_test']),
                lookback
            )
            
            # Train with full epochs
            trainer.train_model(prepared_data, verbose=1)
            
            # Get complete metrics
            all_metrics = trainer.evaluate_model(prepared_data)
            
            # Log results
            logger.info("\n📊 Final Evaluation Results:")
            for dataset in ['train', 'val', 'test']:
                logger.info(f"\n{dataset.upper()} SET:")
                metrics = all_metrics[dataset]
                logger.info(f"   RMSE: {metrics['rmse']:.6f}")
                logger.info(f"   MAE: {metrics['mae']:.6f}")
                logger.info(f"   R²: {metrics['r2_score']:.6f}")
                logger.info(f"   Directional Accuracy: {metrics['directional_accuracy']:.2f}%")
            
            logger.info("=" * 80)
            
            return all_metrics
            
        except Exception as e:
            logger.error(f"❌ Error in final evaluation: {e}")
            return None
    
    def batch_evaluate(self, chromosomes):
        """
        Evaluate multiple chromosomes
        
        Args:
            chromosomes: List of chromosomes to evaluate
            
        Returns:
            list: List of fitness scores
        """
        fitness_scores = []
        
        logger.info(f"\n🔬 Batch Evaluating {len(chromosomes)} chromosomes...")
        
        for i, chromosome in enumerate(chromosomes, 1):
            logger.info(f"\n📊 Evaluating chromosome {i}/{len(chromosomes)}")
            fitness = self.evaluate(chromosome)
            fitness_scores.append(fitness)
        
        return fitness_scores
    
    def get_evaluation_count(self):
        """
        Get the total number of evaluations performed
        
        Returns:
            int: Number of evaluations
        """
        return self.evaluation_count
    
    def reset_count(self):
        """
        Reset the evaluation counter
        """
        self.evaluation_count = 0