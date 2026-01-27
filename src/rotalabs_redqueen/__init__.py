"""rotalabs-redqueen: Evolutionary adversarial testing for LLMs.

A quality-diversity framework for automated red-teaming of language models.
Uses evolutionary algorithms to discover diverse, effective attack strategies.
"""

from rotalabs_redqueen._version import __version__

# Core evolutionary framework
from rotalabs_redqueen.core import (
    Archive,
    ArchiveCoverage,
    BehaviorDescriptor,
    BehaviorDimension,
    EvaluationMetadata,
    Evolution,
    EvolutionConfig,
    EvolutionResult,
    Fitness,
    FitnessResult,
    FitnessValue,
    Genome,
    Individual,
    MapElitesArchive,
    NoveltyArchive,
    NoveltyFitnessSelection,
    NoveltySelection,
    Population,
    PopulationConfig,
    Selection,
    SyncFitness,
    TournamentSelection,
    evolve,
)

# LLM adversarial testing domain
from rotalabs_redqueen.llm import (
    AnthropicTarget,
    AttackStrategy,
    Encoding,
    HarmCategory,
    HeuristicJudge,
    JailbreakFitness,
    JailbreakMetrics,
    Judge,
    JudgeResult,
    LLMAttackGenome,
    LLMJudge,
    LLMTarget,
    MockTarget,
    MultiTargetFitness,
    NetworkError,
    OllamaTarget,
    OpenAITarget,
    Persona,
    RateLimitError,
    TargetError,
    TargetResponse,
    create_target,
)

__all__ = [
    "__version__",
    # Core - Genome
    "Genome",
    "BehaviorDescriptor",
    # Core - Fitness
    "Fitness",
    "SyncFitness",
    "FitnessValue",
    "FitnessResult",
    "EvaluationMetadata",
    # Core - Population
    "Individual",
    "Population",
    "PopulationConfig",
    # Core - Selection
    "Selection",
    "TournamentSelection",
    "NoveltyArchive",
    "NoveltySelection",
    "NoveltyFitnessSelection",
    # Core - Archive
    "Archive",
    "MapElitesArchive",
    "BehaviorDimension",
    "ArchiveCoverage",
    # Core - Evolution
    "Evolution",
    "EvolutionConfig",
    "EvolutionResult",
    "evolve",
    # LLM - Genome
    "LLMAttackGenome",
    "AttackStrategy",
    "Encoding",
    "HarmCategory",
    "Persona",
    # LLM - Targets
    "LLMTarget",
    "OpenAITarget",
    "AnthropicTarget",
    "OllamaTarget",
    "MockTarget",
    "TargetResponse",
    "TargetError",
    "RateLimitError",
    "NetworkError",
    "create_target",
    # LLM - Judges
    "Judge",
    "JudgeResult",
    "HeuristicJudge",
    "LLMJudge",
    # LLM - Fitness
    "JailbreakFitness",
    "MultiTargetFitness",
    "JailbreakMetrics",
]
