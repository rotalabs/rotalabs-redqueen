"""Genome abstract base class for evolutionary computation.

A Genome represents an evolvable solution in the search space. Implementations
define how to create, mutate, and combine solutions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

T = TypeVar("T", bound="Genome")


@dataclass(frozen=True)
class BehaviorDescriptor:
    """Characterizes behavior for quality-diversity algorithms.

    The behavior descriptor places an individual in a multi-dimensional
    behavior space, enabling MAP-Elites and novelty search.
    """

    values: tuple[float, ...]

    def distance(self, other: BehaviorDescriptor) -> float:
        """Euclidean distance between behavior descriptors."""
        if len(self.values) != len(other.values):
            raise ValueError("Behavior descriptors must have same dimensions")
        return float(np.sqrt(sum((a - b) ** 2 for a, b in zip(self.values, other.values))))

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, idx: int) -> float:
        return self.values[idx]


class Genome(ABC, Generic[T]):
    """Abstract base class for evolvable genomes.

    Implementations must define:
    - random(): Create a random genome
    - mutate(): Create a mutated copy
    - crossover(): Combine with another genome
    - to_phenotype(): Convert to evaluatable form
    - behavior(): Extract behavior descriptor for QD
    """

    @classmethod
    @abstractmethod
    def random(cls, rng: np.random.Generator | None = None) -> T:
        """Create a random genome instance.

        Args:
            rng: Random number generator for reproducibility

        Returns:
            New random genome
        """
        pass

    @abstractmethod
    def mutate(self, rng: np.random.Generator | None = None) -> T:
        """Create a mutated copy of this genome.

        Args:
            rng: Random number generator for reproducibility

        Returns:
            New mutated genome (original unchanged)
        """
        pass

    @abstractmethod
    def crossover(self, other: T, rng: np.random.Generator | None = None) -> T:
        """Create offspring by combining with another genome.

        Args:
            other: Another genome to combine with
            rng: Random number generator for reproducibility

        Returns:
            New offspring genome
        """
        pass

    @abstractmethod
    def to_phenotype(self) -> str:
        """Convert genome to evaluatable phenotype.

        For LLM attacks, this produces the actual prompt text.

        Returns:
            String representation for evaluation
        """
        pass

    @abstractmethod
    def behavior(self) -> BehaviorDescriptor:
        """Extract behavior descriptor for quality-diversity.

        Returns:
            Behavior descriptor characterizing this genome
        """
        pass

    def distance(self, other: T) -> float:
        """Genetic distance to another genome.

        Default implementation uses behavior distance. Override for
        custom distance metrics.

        Args:
            other: Another genome

        Returns:
            Distance value (higher = more different)
        """
        return self.behavior().distance(other.behavior())
