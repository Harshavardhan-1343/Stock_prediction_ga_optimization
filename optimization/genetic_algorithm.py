"""
Genetic Algorithm for LSTM Hyperparameter Optimization
"""

import numpy as np
import pandas as pd
import random
import logging
import json
import os
from datetime import datetime
from optimization.chromosome import Chromosome
from optimization.fitness_evaluator import FitnessEvaluator

logger = logging.getLogger(__name__)


class GeneticAlgorithm:
    """
    Genetic Algorithm for optimizing LSTM hyperparameters
    """
    
    def __init__(self, data_dict, population_size=20, generations=10, 
                 mutation_rate=0.1, crossover_rate=0.8, elitism_rate=0.1):
        """
        Initialize Genetic Algorithm
        
        Args:
            data_dict: Dictionary with training data
            population_size: Size of population
            generations: Number of generations
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elitism_rate: Percentage of best individuals to keep
        """
        self.data_dict = data_dict
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = max(1, int(population_size * elitism_rate))
        
        self.population = []
        self.best_chromosome = None
        self.history = {
            'generation': [],
            'best_fitness': [],
            'average_fitness': [],
            'worst_fitness': [],
            'std_fitness': []
        }
        
        # Initialize fitness evaluator
        self.evaluator = FitnessEvaluator(data_dict)
        
        logger.info("="*80)
        logger.info("🧬 GENETIC ALGORITHM INITIALIZED")
        logger.info("="*80)
        logger.info(f"   Population Size: {population_size}")
        logger.info(f"   Generations: {generations}")
        logger.info(f"   Mutation Rate: {mutation_rate}")
        logger.info(f"   Crossover Rate: {crossover_rate}")
        logger.info(f"   Elitism Count: {self.elitism_count}")
        logger.info("="*80)
    
    def initialize_population(self):
        """
        Create initial population with random chromosomes
        """
        logger.info("\n🌱 Initializing population...")
        
        self.population = []
        for i in range(self.population_size):
            chromosome = Chromosome()
            self.population.append(chromosome)
            logger.info(f"   Individual {i+1}/{self.population_size} created")
        
        logger.info(f"✅ Population initialized with {len(self.population)} individuals")
    
    def evaluate_population(self):
        """
        Evaluate fitness of all individuals in population
        """
        logger.info("\n🔬 Evaluating population...")
        
        unevaluated = [ind for ind in self.population if ind.fitness is None]
        
        if not unevaluated:
            logger.info("   All individuals already evaluated")
            return
        
        logger.info(f"   {len(unevaluated)} individuals to evaluate")
        
        for i, individual in enumerate(unevaluated, 1):
            logger.info(f"\n📊 Evaluating individual {i}/{len(unevaluated)}")
            fitness = self.evaluator.evaluate(individual)
            
            # CRITICAL: Check if fitness was properly assigned
            if individual.fitness is None:
                logger.error(f"❌ Individual {i} fitness is still None after evaluation!")
                logger.error(f"   Returned fitness: {fitness}")
                logger.error(f"   Genes: {individual.genes}")
                individual.fitness = float('inf')  # Assign bad fitness
            else:
                logger.info(f"✅ Individual {i} fitness: {individual.fitness:.6f}")
        
        # Sort population by fitness (lower is better)
        self.population.sort(key=lambda x: x.fitness if x.fitness is not None else float('inf'))
        
        # Update best chromosome
        if self.population[0].fitness is not None and self.population[0].fitness != float('inf'):
            if self.best_chromosome is None or self.population[0].fitness < self.best_chromosome.fitness:
                self.best_chromosome = self.population[0].copy()
                logger.info(f"\n🏆 NEW BEST CHROMOSOME FOUND!")
                logger.info(f"   Fitness: {self.best_chromosome.fitness:.6f}")
        
        # Log statistics
        fitness_values = [ind.fitness for ind in self.population if ind.fitness is not None and ind.fitness != float('inf')]
        
        if fitness_values:
            logger.info(f"\n✅ Population evaluation complete!")
            logger.info(f"   Best fitness: {min(fitness_values):.6f}")
            logger.info(f"   Worst fitness: {max(fitness_values):.6f}")
            logger.info(f"   Average fitness: {np.mean(fitness_values):.6f}")
        else:
            logger.error("❌ NO VALID FITNESS VALUES IN POPULATION!")
    
    def selection(self):
        """
        Select parents for reproduction using tournament selection
        
        Returns:
            List of selected parent chromosomes
        """
        parents = []
        tournament_size = 3
        
        # Keep elite individuals
        elite = self.population[:self.elitism_count]
        
        # Select remaining parents through tournament
        for _ in range(self.population_size - self.elitism_count):
            # Tournament selection
            tournament = random.sample(self.population, min(tournament_size, len(self.population)))
            winner = min(tournament, key=lambda x: x.fitness if x.fitness is not None else float('inf'))
            parents.append(winner.copy())
        
        return elite + parents
    
    def crossover(self, parents):
        """
        Create offspring through crossover
        
        Args:
            parents: List of parent chromosomes
            
        Returns:
            List of offspring chromosomes
        """
        offspring = []
        
        # Keep elite individuals unchanged
        offspring.extend([p.copy() for p in parents[:self.elitism_count]])
        
        # Create offspring from remaining parents
        remaining = parents[self.elitism_count:]
        random.shuffle(remaining)
        
        for i in range(0, len(remaining) - 1, 2):
            parent1 = remaining[i]
            parent2 = remaining[i + 1]
            
            if random.random() < self.crossover_rate:
                child1, child2 = parent1.crossover(parent2)
                offspring.append(child1)
                offspring.append(child2)
            else:
                offspring.append(parent1.copy())
                offspring.append(parent2.copy())
        
        # Handle odd number of parents
        if len(remaining) % 2 == 1:
            offspring.append(remaining[-1].copy())
        
        return offspring[:self.population_size]
    
    def mutation(self, offspring):
        """
        Apply mutation to offspring
        
        Args:
            offspring: List of offspring chromosomes
            
        Returns:
            List of mutated offspring
        """
        mutated = []
        
        # Don't mutate elite individuals
        mutated.extend(offspring[:self.elitism_count])
        
        # Mutate remaining offspring
        for individual in offspring[self.elitism_count:]:
            mutated_individual = individual.mutate(self.mutation_rate)
            mutated.append(mutated_individual)
        
        return mutated
    
    def evolve(self):
        """
        Run the genetic algorithm evolution process
        """
        logger.info("\n" + "="*80)
        logger.info("🚀 STARTING EVOLUTION")
        logger.info("="*80)
        
        # Initialize population
        self.initialize_population()
        
        # Evaluate initial population
        self.evaluate_population()
        
        # Evolution loop
        for generation in range(self.generations):
            logger.info("\n" + "="*80)
            logger.info(f"🧬 GENERATION {generation + 1}/{self.generations}")
            logger.info("="*80)
            
            # Selection
            logger.info("\n🎯 Selection phase...")
            parents = self.selection()
            logger.info(f"   Selected {len(parents)} parents")
            
            # Crossover
            logger.info("\n🔀 Crossover phase...")
            offspring = self.crossover(parents)
            logger.info(f"   Created {len(offspring)} offspring")
            
            # Mutation
            logger.info("\n🧪 Mutation phase...")
            offspring = self.mutation(offspring)
            logger.info(f"   Mutated {len(offspring) - self.elitism_count} individuals")
            
            # Replace population
            self.population = offspring
            
            # Evaluate new population
            self.evaluate_population()
            
            # Record statistics
            fitness_values = [ind.fitness for ind in self.population if ind.fitness is not None and ind.fitness != float('inf')]
            
            if fitness_values:
                self.history['generation'].append(generation + 1)
                self.history['best_fitness'].append(min(fitness_values))
                self.history['average_fitness'].append(np.mean(fitness_values))
                self.history['worst_fitness'].append(max(fitness_values))
                self.history['std_fitness'].append(np.std(fitness_values))
                
                logger.info(f"\n📊 Generation {generation + 1} Summary:")
                logger.info(f"   Best Fitness: {min(fitness_values):.6f}")
                logger.info(f"   Average Fitness: {np.mean(fitness_values):.6f}")
                logger.info(f"   Worst Fitness: {max(fitness_values):.6f}")
                logger.info(f"   Std Dev: {np.std(fitness_values):.6f}")
            else:
                logger.error(f"❌ Generation {generation + 1}: NO VALID FITNESS VALUES!")
            
            # Save checkpoint
            if (generation + 1) % 5 == 0:
                self.save_checkpoint(f"ga_checkpoint_gen_{generation + 1}.json")
        
        logger.info("\n" + "="*80)
        logger.info("✅ EVOLUTION COMPLETE")
        logger.info("="*80)
        
        if self.best_chromosome:
            logger.info(f"\n🏆 BEST CHROMOSOME:")
            logger.info(f"   Fitness: {self.best_chromosome.fitness:.6f}")
            logger.info(f"   Genes: {self.best_chromosome.genes}")
        else:
            logger.error("❌ NO BEST CHROMOSOME FOUND!")
    
    def get_best_chromosome(self):
        """
        Get the best chromosome found
        
        Returns:
            Best chromosome
        """
        return self.best_chromosome
    
    def get_history_dataframe(self):
        """
        Get evolution history as pandas DataFrame
        
        Returns:
            DataFrame with evolution history
        """
        return pd.DataFrame(self.history)
    
    def save_results(self, filename=None):
        """
        Save GA results to file
        
        Args:
            filename: Name of file to save results
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ga_results_{timestamp}.json"
        
        import config
        filepath = os.path.join(config.RESULTS_DIR, filename)
        
        results = {
            'best_chromosome': self.best_chromosome.to_dict() if self.best_chromosome else None,
            'history': self.history,
            'parameters': {
                'population_size': self.population_size,
                'generations': self.generations,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'elitism_count': self.elitism_count
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n💾 Results saved to: {filepath}")
    
    def save_checkpoint(self, filename):
        """
        Save GA checkpoint
    
        Args:
            filename: Name of checkpoint file
        """
        import config
        filepath = os.path.join(config.RESULTS_DIR, filename)
    
        try:
            checkpoint = {
                'population': [ind.to_dict() for ind in self.population],
                'best_chromosome': self.best_chromosome.to_dict() if self.best_chromosome else None,
                'history': {
                    'generation': [int(x) for x in self.history['generation']],
                'best_fitness': [float(x) for x in self.history['best_fitness']],
                'average_fitness': [float(x) for x in self.history['average_fitness']],
                'worst_fitness': [float(x) for x in self.history['worst_fitness']],
                'std_fitness': [float(x) for x in self.history['std_fitness']]
            },
            'evaluator_count': int(self.evaluator.get_evaluation_count())
            }
        
            with open(filepath, 'w') as f:
                json.dump(checkpoint, f, indent=2)
        
            logger.info(f"💾 Checkpoint saved: {filename}")
        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, filename):
        """
        Load GA checkpoint
        
        Args:
            filename: Name of checkpoint file
        """
        import config
        filepath = os.path.join(config.RESULTS_DIR, filename)
        
        with open(filepath, 'r') as f:
            checkpoint = json.load(f)
        
        self.population = [Chromosome.from_dict(ind) for ind in checkpoint['population']]
        self.best_chromosome = Chromosome.from_dict(checkpoint['best_chromosome']) if checkpoint['best_chromosome'] else None
        self.history = checkpoint['history']
        
        logger.info(f"✅ Checkpoint loaded from: {filepath}")