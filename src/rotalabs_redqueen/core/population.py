"""Population management for evolutionary computation.

A Population maintains a collection of individuals across generations,
supporting selection, replacement, and statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

import numpy as np

from rotalabs_redqueen.core.fitness import FitnessResult, FitnessValue
from rotalabs_redqueen.core.genome import BehaviorDescriptor, Genome

G = TypeVar("G", bound=Genome)


@dataclass
class Individual(Generic[G]):
    """An individual in the population.

    Combines a genome with its evaluation results.
    """

    genome: G
    fitness: FitnessValue = field(default_factory=lambda: FitnessValue(0.0))
    behavior: BehaviorDescriptor = field(default_factory=lambda: BehaviorDescriptor(()))
    birth_generation: int = 0

    @classmethod
    def from_result(cls, genome: G, result: FitnessResult, generation: int = 0) -> Individual[G]:
        """Create individual from fitness result.

        Args:
            genome: The genome
            result: Fitness evaluation result
            generation: Birth generation

        Returns:
            New individual
        """
        return cls(
            genome=genome,
            fitness=result.fitness,
            behavior=result.behavior,
            birth_generation=generation,
        )

    def __lt__(self, other: Individual[G]) -> bool:
        return self.fitness < other.fitness

    def __le__(self, other: Individual[G]) -> bool:
        return self.fitness <= other.fitness

    def __gt__(self, other: Individual[G]) -> bool:
        return self.fitness > other.fitness

    def __ge__(self, other: Individual[G]) -> bool:
        return self.fitness >= other.fitness


@dataclass
class PopulationConfig:
    """Configuration for population management."""

    size: int = 100
    elitism: int = 1  # Number of best individuals to preserve


class Population(Generic[G]):
    """A population of individuals.

    Manages individuals across generations with selection and replacement.
    """

    def __init__(
        self,
        individuals: list[Individual[G]] | None = None,
        config: PopulationConfig | None = None,
        generation: int = 0,
    ):
        """Initialize population.

        Args:
            individuals: Initial individuals (or empty)
            config: Population configuration
            generation: Current generation number
        """
        self.individuals: list[Individual[G]] = individuals or []
        self.config = config or PopulationConfig()
        self.generation = generation

    @classmethod
    def random(
        cls,
        genome_class: type[G],
        size: int,
        rng: np.random.Generator | None = None,
    ) -> Population[G]:
        """Create a population of random individuals.

        Args:
            genome_class: Genome class with random() method
            size: Population size
            rng: Random number generator

        Returns:
            New population with random genomes (unevaluated)
        """
        individuals = [
            Individual(genome=genome_class.random(rng)) for _ in range(size)
        ]
        return cls(individuals=individuals, config=PopulationConfig(size=size))

    def __len__(self) -> int:
        return len(self.individuals)

    def __getitem__(self, idx: int) -> Individual[G]:
        return self.individuals[idx]

    def __iter__(self):
        return iter(self.individuals)

    def add(self, individual: Individual[G]) -> None:
        """Add an individual to the population."""
        self.individuals.append(individual)

    def best(self) -> Individual[G] | None:
        """Get the best individual by fitness.

        Returns:
            Best individual or None if empty
        """
        if not self.individuals:
            return None
        return max(self.individuals, key=lambda i: i.fitness.value)

    def top_n(self, n: int) -> list[Individual[G]]:
        """Get the top N individuals by fitness.

        Args:
            n: Number of individuals to return

        Returns:
            List of top N individuals, sorted by fitness (best first)
        """
        return sorted(self.individuals, key=lambda i: i.fitness.value, reverse=True)[:n]

    def tournament_select(
        self,
        tournament_size: int = 3,
        rng: np.random.Generator | None = None,
    ) -> Individual[G]:
        """Select an individual via tournament selection.

        Args:
            tournament_size: Number of individuals in tournament
            rng: Random number generator

        Returns:
            Winner of tournament (best fitness)
        """
        if rng is None:
            rng = np.random.default_rng()

        indices = rng.choice(len(self.individuals), size=tournament_size, replace=False)
        contestants = [self.individuals[i] for i in indices]
        return max(contestants, key=lambda i: i.fitness.value)

    def select_parents(
        self,
        n: int,
        tournament_size: int = 3,
        rng: np.random.Generator | None = None,
    ) -> list[Individual[G]]:
        """Select N parents via tournament selection.

        Args:
            n: Number of parents to select
            tournament_size: Tournament size
            rng: Random number generator

        Returns:
            List of selected parents
        """
        return [self.tournament_select(tournament_size, rng) for _ in range(n)]

    def average_fitness(self) -> float:
        """Get average fitness of population."""
        if not self.individuals:
            return 0.0
        return sum(i.fitness.value for i in self.individuals) / len(self.individuals)

    def fitness_std(self) -> float:
        """Get standard deviation of fitness."""
        if len(self.individuals) < 2:
            return 0.0
        values = [i.fitness.value for i in self.individuals]
        return float(np.std(values))

    def diversity(self) -> float:
        """Measure population diversity using average pairwise distance.

        Returns:
            Average behavior distance between individuals
        """
        if len(self.individuals) < 2:
            return 0.0

        total = 0.0
        count = 0
        for i, ind1 in enumerate(self.individuals):
            for ind2 in self.individuals[i + 1:]:
                total += ind1.behavior.distance(ind2.behavior)
                count += 1

        return total / count if count > 0 else 0.0
