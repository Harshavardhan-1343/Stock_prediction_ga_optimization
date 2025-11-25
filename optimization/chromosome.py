"""
Chromosome representation for Genetic Algorithm
Encodes LSTM model hyperparameters
"""

import numpy as np
import random
import logging

logger = logging.getLogger(__name__)


class Chromosome:
    """
    Represents a set of hyperparameters as a chromosome
    """
    
    # Define the hyperparameter search space
    GENE_SPACE = {
        'n_lstm_layers': [1, 2, 3],
        'neurons_layer1': [64, 128, 256],
        'neurons_layer2': [32, 64, 128, 256],
        'neurons_layer3': [16, 32, 64, 128],
        'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01],
        'batch_size': [16, 32, 64, 128],
        'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
        'lookback_window': [30, 60, 90, 120],
        'optimizer': ['adam', 'rmsprop', 'sgd'],
        'dense_units': [16, 32, 64, 128],
        'use_bidirectional': [True, False]
    }
    
    def __init__(self, genes=None):
        """
        Initialize chromosome with genes
        
        Args:
            genes: Dictionary of hyperparameters (optional)
        """
        if genes is None:
            # Create random genes
            self.genes = self._create_random_genes()
        else:
            self.genes = genes.copy()
        
        self.fitness = None
        self.metrics = None
        self.model_params = 0
    
    def _create_random_genes(self):
        """
        Create random genes from gene space
        
        Returns:
            Dictionary of random hyperparameters
        """
        genes = {}
        for gene_name, gene_values in self.GENE_SPACE.items():
            genes[gene_name] = random.choice(gene_values)
        return genes
    
    def get_hyperparameters(self):
        """
        Get hyperparameters in format suitable for model training
        
        Returns:
            Dictionary of hyperparameters
        """
        # Add default values for additional hyperparameters
        hyperparameters = self.genes.copy()
        
        # Add epochs (not evolved, fixed)
        hyperparameters['epochs'] = 100
        
        return hyperparameters
    
    def mutate(self, mutation_rate=0.1):
        """
        Mutate genes with given probability
        
        Args:
            mutation_rate: Probability of mutation for each gene
            
        Returns:
            New mutated chromosome
        """
        new_genes = self.genes.copy()
        
        for gene_name in new_genes.keys():
            if random.random() < mutation_rate:
                # Mutate this gene
                new_genes[gene_name] = random.choice(self.GENE_SPACE[gene_name])
        
        return Chromosome(new_genes)
    
    def crossover(self, other):
        """
        Perform crossover with another chromosome
        
        Args:
            other: Another chromosome
            
        Returns:
            Two new offspring chromosomes
        """
        # Single-point crossover
        genes1 = {}
        genes2 = {}
        
        gene_names = list(self.genes.keys())
        crossover_point = random.randint(0, len(gene_names))
        
        for i, gene_name in enumerate(gene_names):
            if i < crossover_point:
                genes1[gene_name] = self.genes[gene_name]
                genes2[gene_name] = other.genes[gene_name]
            else:
                genes1[gene_name] = other.genes[gene_name]
                genes2[gene_name] = self.genes[gene_name]
        
        return Chromosome(genes1), Chromosome(genes2)
    
    def copy(self):
        """
        Create a copy of this chromosome
        
        Returns:
            New chromosome with same genes
        """
        new_chromosome = Chromosome(self.genes.copy())
        new_chromosome.fitness = self.fitness
        new_chromosome.metrics = self.metrics
        new_chromosome.model_params = self.model_params
        return new_chromosome
    
    def __str__(self):
        """
        String representation
        
        Returns:
            String description of chromosome
        """
        return f"Chromosome(fitness={self.fitness:.6f if self.fitness else 'None'}, genes={self.genes})"
    
    def __repr__(self):
        """
        Representation for debugging
        
        Returns:
            String representation
        """
        return self.__str__()
    
    def __lt__(self, other):
        """
        Less than comparison (for sorting)
        Lower fitness is better
        
        Args:
            other: Another chromosome
            
        Returns:
            bool: True if this chromosome is better (lower fitness)
        """
        if self.fitness is None:
            return False
        if other.fitness is None:
            return True
        return self.fitness < other.fitness
    
    def __eq__(self, other):
        """
        Equality comparison
        
        Args:
            other: Another chromosome
            
        Returns:
            bool: True if genes are identical
        """
        if not isinstance(other, Chromosome):
            return False
        return self.genes == other.genes
    
    def to_dict(self):
        """
        Convert chromosome to dictionary
        
        Returns:
            Dictionary representation
        """
        return {
            'genes': self.genes,
            'fitness': self.fitness,
            'metrics': self.metrics,
            'model_params': self.model_params
        }
    
    @classmethod
    def from_dict(cls, data):
        """
        Create chromosome from dictionary
        
        Args:
            data: Dictionary with chromosome data
            
        Returns:
            New chromosome
        """
        chromosome = cls(data['genes'])
        chromosome.fitness = data.get('fitness')
        chromosome.metrics = data.get('metrics')
        chromosome.model_params = data.get('model_params', 0)
        return chromosome