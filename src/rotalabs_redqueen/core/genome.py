"""Genome abstract base class for evolutionary computation.

A Genome represents an evolvable solution in the search space. Implementations
define how to create, mutate, and combine solutions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

from rotalabs_redqueen.core.rng import Rng
from rotalabs_redqueen.core.stimulus import Stimulus

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
    - to_stimulus(): Convert to the Stimulus a Target executes
    - behavior(): Extract behavior descriptor for QD
    """

    @classmethod
    @abstractmethod
    def random(cls, rng: Rng | None = None) -> T:
        """Create a random genome instance.

        Args:
            rng: Random number generator for reproducibility

        Returns:
            New random genome
        """
        pass

    @abstractmethod
    def mutate(self, rng: Rng | None = None) -> T:
        """Create a mutated copy of this genome.

        Args:
            rng: Random number generator for reproducibility

        Returns:
            New mutated genome (original unchanged)
        """
        pass

    @abstractmethod
    def crossover(self, other: T, rng: Rng | None = None) -> T:
        """Create offspring by combining with another genome.

        Args:
            other: Another genome to combine with
            rng: Random number generator for reproducibility

        Returns:
            New offspring genome
        """
        pass

    @abstractmethod
    def to_stimulus(self) -> Stimulus:
        """Convert genome to the Stimulus a Target executes.

        May be a single prompt, a multi-turn conversation, or an agentic
        action plan (see :mod:`rotalabs_redqueen.core.stimulus`).

        Returns:
            The Stimulus phenotype for evaluation.
        """
        pass

    @abstractmethod
    def behavior(self) -> BehaviorDescriptor:
        """Extract behavior descriptor for quality-diversity.

        Returns:
            Behavior descriptor characterizing this genome
        """
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize the genome to a JSON-compatible dict.

        Must round-trip via :meth:`from_dict`. This is how archives persist
        genomes for cross-run continuity.
        """
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict) -> T:
        """Reconstruct a genome from :meth:`to_dict` output."""
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
