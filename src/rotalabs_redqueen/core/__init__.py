"""Core evolutionary computation framework.

This module provides the generic evolutionary infrastructure that can be
used for any optimization or quality-diversity problem.
"""

from rotalabs_redqueen.core.archive import (
    Archive,
    ArchiveCoverage,
    BehaviorDimension,
    MapElitesArchive,
)
from rotalabs_redqueen.core.evolution import (
    Evolution,
    EvolutionConfig,
    EvolutionResult,
    evolve,
)
from rotalabs_redqueen.core.fitness import (
    EvaluationMetadata,
    Fitness,
    FitnessResult,
    FitnessValue,
    SyncFitness,
)
from rotalabs_redqueen.core.genome import BehaviorDescriptor, Genome
from rotalabs_redqueen.core.population import Individual, Population, PopulationConfig
from rotalabs_redqueen.core.selection import (
    NoveltyArchive,
    NoveltyFitnessSelection,
    NoveltySelection,
    Selection,
    TournamentSelection,
)

__all__ = [
    # Genome
    "Genome",
    "BehaviorDescriptor",
    # Fitness
    "Fitness",
    "SyncFitness",
    "FitnessValue",
    "FitnessResult",
    "EvaluationMetadata",
    # Population
    "Individual",
    "Population",
    "PopulationConfig",
    # Selection
    "Selection",
    "TournamentSelection",
    "NoveltyArchive",
    "NoveltySelection",
    "NoveltyFitnessSelection",
    # Archive
    "Archive",
    "MapElitesArchive",
    "BehaviorDimension",
    "ArchiveCoverage",
    # Evolution
    "Evolution",
    "EvolutionConfig",
    "EvolutionResult",
    "evolve",
]
