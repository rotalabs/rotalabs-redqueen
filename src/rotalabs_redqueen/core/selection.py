"""Selection operators for evolutionary computation.

Selection operators determine which individuals become parents
for the next generation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, TypeVar

import numpy as np

from rotalabs_redqueen.core.genome import BehaviorDescriptor, Genome
from rotalabs_redqueen.core.population import Individual, Population

G = TypeVar("G", bound=Genome)


class Selection(ABC, Generic[G]):
    """Abstract base class for selection operators."""

    @abstractmethod
    def select(
        self,
        population: Population[G],
        n: int,
        rng: np.random.Generator | None = None,
    ) -> list[Individual[G]]:
        """Select N individuals from the population.

        Args:
            population: Population to select from
            n: Number of individuals to select
            rng: Random number generator

        Returns:
            List of selected individuals
        """
        pass


class TournamentSelection(Selection[G], Generic[G]):
    """Tournament selection operator.

    Selects individuals by running tournaments where the best
    of a random subset wins.
    """

    def __init__(self, tournament_size: int = 3):
        """Initialize tournament selection.

        Args:
            tournament_size: Number of individuals per tournament
        """
        self.tournament_size = tournament_size

    def select(
        self,
        population: Population[G],
        n: int,
        rng: np.random.Generator | None = None,
    ) -> list[Individual[G]]:
        """Select N individuals via tournament."""
        return population.select_parents(n, self.tournament_size, rng)


@dataclass
class NoveltyArchive(Generic[G]):
    """Archive for novelty search.

    Tracks explored behaviors to compute novelty scores.
    """

    behaviors: list[BehaviorDescriptor] = field(default_factory=list)
    novelty_threshold: float = 0.1
    max_size: int = 1000
    k_nearest: int = 15

    def add(self, behavior: BehaviorDescriptor) -> None:
        """Add a behavior to the archive.

        Only adds if sufficiently novel.

        Args:
            behavior: Behavior descriptor to potentially add
        """
        if len(self.behaviors) == 0:
            self.behaviors.append(behavior)
            return

        novelty = self.compute_novelty(behavior)
        if novelty > self.novelty_threshold:
            self.behaviors.append(behavior)
            # Trim if too large
            if len(self.behaviors) > self.max_size:
                self.behaviors = self.behaviors[-self.max_size:]

    def compute_novelty(self, behavior: BehaviorDescriptor) -> float:
        """Compute novelty score for a behavior.

        Novelty is the average distance to k-nearest neighbors.

        Args:
            behavior: Behavior to score

        Returns:
            Novelty score (higher = more novel)
        """
        if len(self.behaviors) == 0:
            return float("inf")

        # Compute distances to all archived behaviors
        distances = sorted([behavior.distance(b) for b in self.behaviors])

        # Average of k-nearest
        k = min(self.k_nearest, len(distances))
        return sum(distances[:k]) / k


class NoveltySelection(Selection[G], Generic[G]):
    """Selection based purely on novelty.

    Selects individuals with the most novel behaviors.
    """

    def __init__(self, archive: NoveltyArchive[G] | None = None, k_nearest: int = 15):
        """Initialize novelty selection.

        Args:
            archive: Shared novelty archive (or creates new one)
            k_nearest: Number of neighbors for novelty computation
        """
        self.archive = archive or NoveltyArchive(k_nearest=k_nearest)
        self.k_nearest = k_nearest

    def select(
        self,
        population: Population[G],
        n: int,
        rng: np.random.Generator | None = None,
    ) -> list[Individual[G]]:
        """Select N most novel individuals."""
        # Score all individuals by novelty
        scored = []
        for ind in population:
            novelty = self.archive.compute_novelty(ind.behavior)
            scored.append((ind, novelty))

        # Sort by novelty (highest first)
        scored.sort(key=lambda x: x[1], reverse=True)

        # Update archive with selected behaviors
        selected = [ind for ind, _ in scored[:n]]
        for ind in selected:
            self.archive.add(ind.behavior)

        return selected


class NoveltyFitnessSelection(Selection[G], Generic[G]):
    """Combined novelty and fitness selection.

    Balances exploration (novelty) and exploitation (fitness).
    """

    def __init__(
        self,
        archive: NoveltyArchive[G] | None = None,
        novelty_weight: float = 0.5,
        k_nearest: int = 15,
    ):
        """Initialize combined selection.

        Args:
            archive: Shared novelty archive
            novelty_weight: Weight for novelty (0-1, rest goes to fitness)
            k_nearest: Number of neighbors for novelty
        """
        self.archive = archive or NoveltyArchive(k_nearest=k_nearest)
        self.novelty_weight = novelty_weight
        self.fitness_weight = 1.0 - novelty_weight

    def select(
        self,
        population: Population[G],
        n: int,
        rng: np.random.Generator | None = None,
    ) -> list[Individual[G]]:
        """Select based on combined novelty and fitness score."""
        # Score all individuals
        scored = []
        max_fitness = max(ind.fitness.value for ind in population) or 1.0
        max_novelty = 1.0  # Will normalize

        novelty_scores = []
        for ind in population:
            novelty = self.archive.compute_novelty(ind.behavior)
            novelty_scores.append(novelty)

        max_novelty = max(novelty_scores) or 1.0

        for ind, novelty in zip(population, novelty_scores):
            norm_fitness = ind.fitness.value / max_fitness
            norm_novelty = novelty / max_novelty
            combined = (
                self.fitness_weight * norm_fitness + self.novelty_weight * norm_novelty
            )
            scored.append((ind, combined))

        # Sort by combined score
        scored.sort(key=lambda x: x[1], reverse=True)

        # Update archive
        selected = [ind for ind, _ in scored[:n]]
        for ind in selected:
            self.archive.add(ind.behavior)

        return selected
