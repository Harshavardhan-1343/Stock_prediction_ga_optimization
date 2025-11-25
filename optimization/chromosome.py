"""
Chromosome representation for Genetic Algorithm
"""

import random
import numpy as np
from typing import Dict, Any
import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Chromosome:
    """
    Represents a chromosome (individual) in the genetic algorithm
    Each chromosome encodes hyperparameters for the LSTM model
    """
    
    def __init__(self, genes: Dict[str, Any] = None):
        """
        Initialize chromosome
        
        Args:
            genes: Dictionary of genes (hyperparameters)
        """
        if genes is None:
            self.genes = self._initialize_random_genes()
        else:
            self.genes = genes
        
        self.fitness = 0.0
        self.metrics = {}
        
    def _initialize_random_genes(self) -> Dict[str, Any]:
        """
        Initialize chromosome with random genes
        
        Returns:
            Dictionary of random genes
        """
        genes = {}
        
        for gene_name, gene_space in config.GENE_SPACE.items():
            genes[gene_name] = random.choice(gene_space)
        
        return genes
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        Get hyperparameters from genes
        
        Returns:
            Dictionary of hyperparameters
        """
        return self.genes.copy()
    
    def mutate(self, mutation_rate: float = 0.15):
        """
        Mutate the chromosome
        
        Args:
            mutation_rate: Probability of mutation for each gene
        """
        for gene_name, gene_space in config.GENE_SPACE.items():
            if random.random() < mutation_rate:
                self.genes[gene_name] = random.choice(gene_space)
                logger.debug(f"Mutated {gene_name} to {self.genes[gene_name]}")
    
    @staticmethod
    def crossover(parent1: 'Chromosome', parent2: 'Chromosome', 
                  crossover_rate: float = 0.8) -> tuple:
        """
        Perform crossover between two parent chromosomes
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            crossover_rate: Probability of crossover
            
        Returns:
            Tuple of two offspring chromosomes
        """
        if random.random() > crossover_rate:
            # No crossover, return copies of parents
            return Chromosome(parent1.genes.copy()), Chromosome(parent2.genes.copy())
        
        # Uniform crossover
        child1_genes = {}
        child2_genes = {}
        
        for gene_name in parent1.genes.keys():
            if random.random() < 0.5:
                child1_genes[gene_name] = parent1.genes[gene_name]
                child2_genes[gene_name] = parent2.genes[gene_name]
            else:
                child1_genes[gene_name] = parent2.genes[gene_name]
                child2_genes[gene_name] = parent1.genes[gene_name]
        
        return Chromosome(child1_genes), Chromosome(child2_genes)
    
    def __str__(self) -> str:
        """
        String representation of chromosome
        
        Returns:
            String representation
        """
        genes_str = ", ".join([f"{k}={v}" for k, v in self.genes.items()])
        return f"Chromosome(fitness={self.fitness:.4f}, genes=[{genes_str}])"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def to_dict(self) -> Dict:
        """
        Convert chromosome to dictionary
        
        Returns:
            Dictionary representation
        """
        return {
            'genes': self.genes,
            'fitness': self.fitness,
            'metrics': self.metrics
        }
    
    def from_dict(self, data: Dict):
        """
        Load chromosome from dictionary
        
        Args:
            data: Dictionary with chromosome data
        """
        self.genes = data['genes']
        self.fitness = data.get('fitness', 0.0)
        self.metrics = data.get('metrics', {})
    
    def clone(self) -> 'Chromosome':
        """
        Create a clone of this chromosome
        
        Returns:
            Cloned chromosome
        """
        cloned = Chromosome(self.genes.copy())
        cloned.fitness = self.fitness
        cloned.metrics = self.metrics.copy()
        return cloned


if __name__ == "__main__":
    # Test chromosome
    print("🧬 Testing Chromosome...")
    
    # Create random chromosome
    chrom1 = Chromosome()
    print(f"\nChromosome 1:\n{chrom1}")
    
    # Create another chromosome
    chrom2 = Chromosome()
    print(f"\nChromosome 2:\n{chrom2}")
    
    # Test crossover
    child1, child2 = Chromosome.crossover(chrom1, chrom2)
    print(f"\nChild 1 after crossover:\n{child1}")
    print(f"\nChild 2 after crossover:\n{child2}")
    
    # Test mutation
    child1.mutate(mutation_rate=0.3)
    print(f"\nChild 1 after mutation:\n{child1}")