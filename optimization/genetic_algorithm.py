"""
Genetic Algorithm Implementation
"""

import random
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import logging
import json
import os
from datetime import datetime
from .chromosome import Chromosome
from .fitness_evaluator import FitnessEvaluator
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeneticAlgorithm:
    """
    Genetic Algorithm for hyperparameter optimization
    """
    
    def __init__(self, data_dict: Dict, population_size: int = None, 
                 generations: int = None):
        """
        Initialize Genetic Algorithm
        
        Args:
            data_dict: Dictionary with prepared data
            population_size: Size of population
            generations: Number of generations
        """
        self.data_dict = data_dict
        self.population_size = population_size or config.GA_CONFIG['population_size']
        self.generations = generations or config.GA_CONFIG['generations']
        
        self.crossover_rate = config.GA_CONFIG['crossover_rate']
        self.mutation_rate = config.GA_CONFIG['mutation_rate']
        self.tournament_size = config.GA_CONFIG['tournament_size']
        self.elitism_count = config.GA_CONFIG['elitism_count']
        
        self.population = []
        self.best_chromosome = None
        self.history = {
            'generation': [],
            'best_fitness': [],
            'avg_fitness': [],
            'worst_fitness': [],
            'best_chromosome': []
        }
        
        self.evaluator = FitnessEvaluator(data_dict)
        
    def initialize_population(self):
        """
        Initialize population with random chromosomes
        """
        logger.info(f"🧬 Initializing population of {self.population_size} individuals...")
        
        self.population = [Chromosome() for _ in range(self.population_size)]
        
        logger.info("✅ Population initialized!")
        
    def tournament_selection(self) -> Chromosome:
        """
        Select a chromosome using tournament selection
        
        Returns:
            Selected chromosome
        """
        tournament = random.sample(self.population, self.tournament_size)
        winner = max(tournament, key=lambda x: x.fitness)
        return winner
    
    def create_next_generation(self) -> List[Chromosome]:
        """
        Create next generation using selection, crossover, and mutation
        
        Returns:
            List of new chromosomes
        """
        next_generation = []
        
        # Elitism: Keep best individuals
        elite = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:self.elitism_count]
        next_generation.extend([chrom.clone() for chrom in elite])
        
        # Generate rest of population
        while len(next_generation) < self.population_size:
            # Selection
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            # Crossover
            child1, child2 = Chromosome.crossover(parent1, parent2, self.crossover_rate)
            
            # Mutation
            child1.mutate(self.mutation_rate)
            child2.mutate(self.mutation_rate)
            
            next_generation.append(child1)
            if len(next_generation) < self.population_size:
                next_generation.append(child2)
        
        return next_generation[:self.population_size]
    
    def evolve(self):
        """
        Main evolution loop
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 Starting Genetic Algorithm")
        logger.info(f"   Population: {self.population_size}")
        logger.info(f"   Generations: {self.generations}")
        logger.info(f"   Crossover Rate: {self.crossover_rate}")
        logger.info(f"   Mutation Rate: {self.mutation_rate}")
        logger.info(f"{'='*80}\n")
        
        # Initialize population
        self.initialize_population()
        
        # Evolution loop
        for generation in range(self.generations):
            logger.info(f"\n{'#'*80}")
            logger.info(f"GENERATION {generation + 1}/{self.generations}")
            logger.info(f"{'#'*80}\n")
            
            # Evaluate population
            self.population = self.evaluator.evaluate_population(self.population)
            
            # Update best chromosome
            current_best = self.population[0]
            if self.best_chromosome is None or current_best.fitness > self.best_chromosome.fitness:
                self.best_chromosome = current_best.clone()
                logger.info(f"\n🎉 New best chromosome found!")
                logger.info(f"   Fitness: {self.best_chromosome.fitness:.4f}")
            
            # Record history
            fitness_scores = [c.fitness for c in self.population]
            self.history['generation'].append(generation + 1)
            self.history['best_fitness'].append(max(fitness_scores))
            self.history['avg_fitness'].append(np.mean(fitness_scores))
            self.history['worst_fitness'].append(min(fitness_scores))
            self.history['best_chromosome'].append(current_best.to_dict())
            
            # Display generation summary
            logger.info(f"\n📊 Generation {generation + 1} Summary:")
            logger.info(f"   Best Fitness: {max(fitness_scores):.4f}")
            logger.info(f"   Average Fitness: {np.mean(fitness_scores):.4f}")
            logger.info(f"   Worst Fitness: {min(fitness_scores):.4f}")
            logger.info(f"   Std Dev: {np.std(fitness_scores):.4f}")
            
            # Create next generation (skip for last generation)
            if generation < self.generations - 1:
                logger.info(f"\n🔄 Creating next generation...")
                self.population = self.create_next_generation()
            
            # Save checkpoint
            self.save_checkpoint(generation)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ Evolution Complete!")
        logger.info(f"   Best Fitness: {self.best_chromosome.fitness:.4f}")
        logger.info(f"   Best Hyperparameters:")
        for key, value in self.best_chromosome.genes.items():
            logger.info(f"      {key}: {value}")
        logger.info(f"{'='*80}\n")
        
    def get_best_chromosome(self) -> Chromosome:
        """
        Get best chromosome from evolution
        
        Returns:
            Best chromosome
        """
        return self.best_chromosome
    
    def get_history_dataframe(self) -> pd.DataFrame:
        """
        Get evolution history as DataFrame
        
        Returns:
            DataFrame with history
        """
        return pd.DataFrame({
            'generation': self.history['generation'],
            'best_fitness': self.history['best_fitness'],
            'avg_fitness': self.history['avg_fitness'],
            'worst_fitness': self.history['worst_fitness']
        })
    
    def save_checkpoint(self, generation: int):
        """
        Save checkpoint of current state
        
        Args:
            generation: Current generation number
        """
        checkpoint = {
            'generation': generation,
            'population_size': self.population_size,
            'best_chromosome': self.best_chromosome.to_dict() if self.best_chromosome else None,
            'history': self.history
        }
        
        filename = f"ga_checkpoint_gen_{generation+1}.json"
        filepath = os.path.join(config.RESULTS_DIR, filename)
        
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)
        
        logger.info(f"💾 Checkpoint saved: {filename}")
    
    def save_results(self):
        """
        Save final results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save best chromosome
        best_chrom_file = os.path.join(config.RESULTS_DIR, f"best_chromosome_{timestamp}.json")
        with open(best_chrom_file, 'w') as f:
            json.dump(self.best_chromosome.to_dict(), f, indent=2, default=str)
        
        # Save history
        history_df = self.get_history_dataframe()
        history_file = os.path.join(config.RESULTS_DIR, f"ga_history_{timestamp}.csv")
        history_df.to_csv(history_file, index=False)
        
        # Save full history with chromosomes
        full_history_file = os.path.join(config.RESULTS_DIR, f"ga_full_history_{timestamp}.json")
        with open(full_history_file, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)
        
        logger.info(f"\n💾 Results saved:")
        logger.info(f"   Best chromosome: {best_chrom_file}")
        logger.info(f"   History: {history_file}")
        logger.info(f"   Full history: {full_history_file}")


if __name__ == "__main__":
    print("🧬 Testing Genetic Algorithm...")
    print("Note: Requires actual data to test properly")