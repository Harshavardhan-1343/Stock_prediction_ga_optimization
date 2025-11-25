"""
Fitness Evaluator for Genetic Algorithm
"""

import numpy as np
from typing import Dict, Tuple
import logging
from models.model_trainer import ModelTrainer
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FitnessEvaluator:
    """
    Evaluates fitness of chromosomes based on model performance
    """
    
    def __init__(self, data_dict: Dict):
        """
        Initialize fitness evaluator
        
        Args:
            data_dict: Dictionary containing train, val, test data
        """
        self.data_dict = data_dict
        
    def calculate_fitness(self, chromosome: 'Chromosome') -> float:
        """
        Calculate fitness score for a chromosome
        
        Args:
            chromosome: Chromosome to evaluate
            
        Returns:
            Fitness score (higher is better)
        """
        try:
            # Get hyperparameters from chromosome
            hyperparameters = chromosome.get_hyperparameters()
            
            logger.info(f"🧬 Evaluating chromosome: {hyperparameters}")
            
            # Create and train model
            trainer = ModelTrainer(hyperparameters)
            
            # Prepare data with lookback window
            lookback = hyperparameters.get('lookback_window', 60)
            prepared_data = trainer.prepare_data(
                (self.data_dict['X_train'], self.data_dict['y_train']),
                (self.data_dict['X_val'], self.data_dict['y_val']),
                (self.data_dict['X_test'], self.data_dict['y_test']),
                lookback
            )
            
            # Train model
            trainer.train_model(prepared_data)
            
            # Evaluate model
            metrics = trainer.evaluate_model(prepared_data)
            
            # Store metrics in chromosome
            chromosome.metrics = metrics
            
            # Calculate fitness
            fitness = self._compute_fitness_score(metrics)
            
            logger.info(f"✅ Fitness: {fitness:.4f} | Val RMSE: {metrics['val']['rmse']:.4f} | "
                       f"Val R²: {metrics['val']['r2_score']:.4f}")
            
            return fitness
            
        except Exception as e:
            logger.error(f"❌ Error evaluating chromosome: {str(e)}")
            return 0.0
    
    def _compute_fitness_score(self, metrics: Dict) -> float:
        """
        Compute fitness score from metrics
        
        Args:
            metrics: Dictionary of evaluation metrics
            
        Returns:
            Fitness score
        """
        # Extract validation metrics
        val_metrics = metrics['val']
        
        rmse = val_metrics['rmse']
        mae = val_metrics['mae']
        r2 = val_metrics['r2_score']
        
        # Model complexity penalty
        total_params = metrics['total_params']
        training_time = metrics['training_time']
        
        # Normalize complexity penalty (0 to 1)
        # Penalize models with > 1M parameters and training time > 1 hour
        complexity_penalty = (total_params / 1_000_000) + (training_time / 3600)
        complexity_penalty = min(complexity_penalty, 1.0)
        
        # Calculate fitness using weighted combination
        # Higher fitness is better
        weights = config.FITNESS_WEIGHTS
        
        # For RMSE and MAE, lower is better, so we use inverse
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        
        fitness = (
            weights['rmse_weight'] * (1.0 / (rmse + epsilon)) +
            weights['mae_weight'] * (1.0 / (mae + epsilon)) +
            weights['r2_weight'] * max(r2, 0) -  # R² can be negative, so clamp to 0
            weights['complexity_weight'] * complexity_penalty
        )
        
        # Normalize fitness to reasonable range
        fitness = fitness * 100
        
        return fitness
    
    def evaluate_population(self, population: list) -> list:
        """
        Evaluate fitness for entire population
        
        Args:
            population: List of chromosomes
            
        Returns:
            List of chromosomes with fitness scores
        """
        logger.info(f"🔬 Evaluating population of {len(population)} individuals...")
        
        for i, chromosome in enumerate(population):
            logger.info(f"\n{'='*80}")
            logger.info(f"Evaluating Individual {i+1}/{len(population)}")
            logger.info(f"{'='*80}")
            
            fitness = self.calculate_fitness(chromosome)
            chromosome.fitness = fitness
        
        # Sort population by fitness (descending)
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        logger.info(f"\n✅ Population evaluation complete!")
        logger.info(f"   Best fitness: {population[0].fitness:.4f}")
        logger.info(f"   Worst fitness: {population[-1].fitness:.4f}")
        logger.info(f"   Average fitness: {np.mean([c.fitness for c in population]):.4f}")
        
        return population


if __name__ == "__main__":
    print("🔬 Testing Fitness Evaluator...")
    print("Note: Requires actual data to test properly")