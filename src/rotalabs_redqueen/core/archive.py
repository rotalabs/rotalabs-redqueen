"""Quality-diversity archives for evolutionary computation.

Archives maintain diverse, high-quality solutions across a behavior space.
MAP-Elites is the primary implementation.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

from rotalabs_redqueen.core.canonical import canonical_json
from rotalabs_redqueen.core.fitness import FitnessValue
from rotalabs_redqueen.core.genome import BehaviorDescriptor, Genome
from rotalabs_redqueen.core.population import Individual
from rotalabs_redqueen.core.stimulus import SPEC_VERSION

G = TypeVar("G", bound=Genome)


@dataclass
class BehaviorDimension:
    """Configuration for one dimension of the behavior space."""

    name: str
    min_value: float
    max_value: float
    bins: int = 10

    def get_bin(self, value: float) -> int:
        """Map a value to a bin index.

        Args:
            value: Value in this dimension

        Returns:
            Bin index (0 to bins-1)
        """
        # Clamp to range
        value = max(self.min_value, min(self.max_value, value))
        # Normalize to [0, 1]
        normalized = (value - self.min_value) / (self.max_value - self.min_value + 1e-10)
        # Map to bin
        bin_idx = int(normalized * self.bins)
        return min(bin_idx, self.bins - 1)


@dataclass
class ArchiveCoverage:
    """Statistics about archive coverage."""

    filled_cells: int
    total_cells: int

    @property
    def coverage_percent(self) -> float:
        """Percentage of cells filled."""
        if self.total_cells == 0:
            return 0.0
        return 100.0 * self.filled_cells / self.total_cells


class Archive(ABC, Generic[G]):
    """Abstract base class for quality-diversity archives."""

    @abstractmethod
    def add(self, individual: Individual[G]) -> bool:
        """Try to add an individual to the archive.

        Args:
            individual: Individual to add

        Returns:
            True if added (new cell or better than existing)
        """
        pass

    @abstractmethod
    def get_all(self) -> list[Individual[G]]:
        """Get all individuals in the archive."""
        pass

    @abstractmethod
    def coverage(self) -> ArchiveCoverage:
        """Get archive coverage statistics."""
        pass


class MapElitesArchive(Archive[G], Generic[G]):
    """MAP-Elites quality-diversity archive.

    Maintains a grid of cells in behavior space, keeping the best
    individual in each cell.
    """

    def __init__(self, dimensions: list[BehaviorDimension]):
        """Initialize MAP-Elites archive.

        Args:
            dimensions: Configuration for each behavior dimension
        """
        self.dimensions = dimensions
        self.cells: dict[tuple[int, ...], Individual[G]] = {}

    def _get_cell_key(self, behavior: BehaviorDescriptor) -> tuple[int, ...]:
        """Map behavior to cell coordinates.

        Args:
            behavior: Behavior descriptor

        Returns:
            Tuple of bin indices for each dimension
        """
        if len(behavior) != len(self.dimensions):
            raise ValueError(
                f"Behavior has {len(behavior)} dimensions, archive has {len(self.dimensions)}"
            )
        return tuple(dim.get_bin(behavior[i]) for i, dim in enumerate(self.dimensions))

    def add(self, individual: Individual[G]) -> bool:
        """Try to add an individual to the archive.

        Adds if:
        - Cell is empty, or
        - Individual has higher fitness than current occupant
        """
        key = self._get_cell_key(individual.behavior)

        if key not in self.cells:
            self.cells[key] = individual
            return True

        if individual.fitness > self.cells[key].fitness:
            self.cells[key] = individual
            return True

        return False

    def get(self, behavior: BehaviorDescriptor) -> Individual[G] | None:
        """Get individual at a behavior location.

        Args:
            behavior: Behavior descriptor

        Returns:
            Individual at that cell or None
        """
        key = self._get_cell_key(behavior)
        return self.cells.get(key)

    def get_all(self) -> list[Individual[G]]:
        """Get all individuals in the archive."""
        return list(self.cells.values())

    def best(self) -> Individual[G] | None:
        """Get the best individual in the archive."""
        if not self.cells:
            return None
        return max(self.cells.values(), key=lambda i: i.fitness.value)

    def coverage(self) -> ArchiveCoverage:
        """Get archive coverage statistics."""
        total = 1
        for dim in self.dimensions:
            total *= dim.bins
        return ArchiveCoverage(filled_cells=len(self.cells), total_cells=total)

    def sample_random(self, n: int, rng=None) -> list[Individual[G]]:
        """Sample N random individuals from the archive.

        Args:
            n: Number to sample
            rng: Random number generator

        Returns:
            List of sampled individuals
        """
        from rotalabs_redqueen.core.rng import Rng

        if rng is None:
            rng = Rng()

        individuals = self.get_all()
        if len(individuals) <= n:
            return individuals

        indices = rng.choice(len(individuals), size=n, replace=False)
        return [individuals[i] for i in indices]

    def seed(self, n: int, rng=None) -> list[G]:
        """Sample N genomes from current elites to warm-start a new run.

        Empty archive -> empty list (redqueen-spec interfaces.md §5).
        """
        return [ind.genome for ind in self.sample_random(n, rng)]

    def to_dict(self) -> dict:
        """Serialize to the Archive wire schema (redqueen-spec types.md)."""
        cov = self.coverage()
        cells = []
        for coords in sorted(self.cells.keys()):
            ind = self.cells[coords]
            cells.append(
                {
                    "coords": list(coords),
                    "elite": {
                        "genome": ind.genome.to_dict(),
                        "fitness": {
                            "value": ind.fitness.value,
                            "objectives": (
                                list(ind.fitness.objectives)
                                if ind.fitness.objectives is not None
                                else None
                            ),
                        },
                        "behavior": {"values": list(ind.behavior.values)},
                        "generation": ind.birth_generation,
                    },
                }
            )
        return {
            "spec_version": SPEC_VERSION,
            "dimensions": [
                {"name": d.name, "min": d.min_value, "max": d.max_value, "bins": d.bins}
                for d in self.dimensions
            ],
            "cells": cells,
            "coverage": {
                "filled_cells": cov.filled_cells,
                "total_cells": cov.total_cells,
                "coverage_percent": cov.coverage_percent,
            },
        }

    def save(self, uri: str) -> None:
        """Persist the archive to a ``file://`` path (or a plain path)."""
        path = uri[len("file://") :] if uri.startswith("file://") else uri
        Path(path).write_text(canonical_json(self.to_dict()) + "\n")

    @classmethod
    def load(cls, uri: str, genome_class: type[G]) -> MapElitesArchive[G]:
        """Load an archive written by :meth:`save`; ``genome_class`` rebuilds elites."""
        path = uri[len("file://") :] if uri.startswith("file://") else uri
        data = json.loads(Path(path).read_text())
        dims = [
            BehaviorDimension(d["name"], d["min"], d["max"], d["bins"]) for d in data["dimensions"]
        ]
        archive = cls(dims)
        for cell in data["cells"]:
            elite = cell["elite"]
            f = elite["fitness"]
            archive.cells[tuple(cell["coords"])] = Individual(
                genome=genome_class.from_dict(elite["genome"]),
                fitness=FitnessValue(
                    value=f["value"],
                    objectives=tuple(f["objectives"]) if f["objectives"] is not None else None,
                ),
                behavior=BehaviorDescriptor(tuple(elite["behavior"]["values"])),
                birth_generation=elite["generation"],
            )
        return archive
