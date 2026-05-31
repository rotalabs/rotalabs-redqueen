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
from rotalabs_redqueen.core.coevolution import CoevolutionResult, coevolve
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
from rotalabs_redqueen.core.report import Report, ReportExporter
from rotalabs_redqueen.core.rng import Rng
from rotalabs_redqueen.core.selection import (
    LexicaseSelection,
    NoveltyArchive,
    NoveltyFitnessSelection,
    NoveltySelection,
    Selection,
    TournamentSelection,
)
from rotalabs_redqueen.core.stimulus import Message, Stimulus, ToolCall, Transcript
from rotalabs_redqueen.core.taxonomy import TaxonomyLabel

__all__ = [
    # RNG
    "Rng",
    # Stimulus
    "Stimulus",
    "Transcript",
    "Message",
    "ToolCall",
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
    "LexicaseSelection",
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
    # Co-evolution
    "coevolve",
    "CoevolutionResult",
    # Reporting
    "Report",
    "ReportExporter",
    "TaxonomyLabel",
]
