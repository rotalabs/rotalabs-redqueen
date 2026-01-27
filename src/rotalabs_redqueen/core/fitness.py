"""Fitness evaluation for evolutionary computation.

Fitness functions evaluate how well a genome solves the target problem.
Supports single-objective, multi-objective, and quality-diversity evaluation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from rotalabs_redqueen.core.genome import BehaviorDescriptor, Genome

G = TypeVar("G", bound=Genome)


@dataclass
class FitnessValue:
    """Fitness value supporting single and multi-objective optimization.

    For single-objective: use value field
    For multi-objective: use objectives field
    """

    value: float = 0.0
    objectives: tuple[float, ...] | None = None

    @property
    def is_multi_objective(self) -> bool:
        """Check if this is a multi-objective fitness."""
        return self.objectives is not None

    def dominates(self, other: FitnessValue) -> bool:
        """Check if this fitness dominates another (Pareto dominance).

        For single-objective: higher is better
        For multi-objective: dominates if >= in all objectives and > in at least one
        """
        if self.is_multi_objective and other.is_multi_objective:
            assert self.objectives is not None and other.objectives is not None
            if len(self.objectives) != len(other.objectives):
                raise ValueError("Objectives must have same dimensions")
            at_least_one_better = False
            for a, b in zip(self.objectives, other.objectives):
                if a < b:
                    return False
                if a > b:
                    at_least_one_better = True
            return at_least_one_better
        return self.value > other.value

    def __lt__(self, other: FitnessValue) -> bool:
        return self.value < other.value

    def __le__(self, other: FitnessValue) -> bool:
        return self.value <= other.value

    def __gt__(self, other: FitnessValue) -> bool:
        return self.value > other.value

    def __ge__(self, other: FitnessValue) -> bool:
        return self.value >= other.value


@dataclass
class EvaluationMetadata:
    """Metadata from fitness evaluation."""

    duration_ms: float = 0.0
    query_count: int = 0
    notes: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class FitnessResult:
    """Complete result of fitness evaluation."""

    fitness: FitnessValue
    behavior: BehaviorDescriptor
    metadata: EvaluationMetadata = field(default_factory=EvaluationMetadata)


class Fitness(ABC, Generic[G]):
    """Abstract base class for fitness evaluation.

    Implementations define how to score genomes for a specific problem.
    """

    @abstractmethod
    async def evaluate(self, genome: G) -> FitnessResult:
        """Evaluate a single genome.

        Args:
            genome: Genome to evaluate

        Returns:
            Fitness result with value, behavior, and metadata
        """
        pass

    async def evaluate_batch(self, genomes: list[G]) -> list[FitnessResult]:
        """Evaluate multiple genomes.

        Default implementation evaluates sequentially. Override for
        parallel evaluation.

        Args:
            genomes: List of genomes to evaluate

        Returns:
            List of fitness results in same order
        """
        import asyncio

        return await asyncio.gather(*[self.evaluate(g) for g in genomes])


class SyncFitness(Fitness[G], Generic[G]):
    """Base class for synchronous fitness functions.

    Use this when evaluation doesn't require async I/O.
    """

    @abstractmethod
    def evaluate_sync(self, genome: G) -> FitnessResult:
        """Synchronous evaluation.

        Args:
            genome: Genome to evaluate

        Returns:
            Fitness result
        """
        pass

    async def evaluate(self, genome: G) -> FitnessResult:
        """Async wrapper for sync evaluation."""
        return self.evaluate_sync(genome)
