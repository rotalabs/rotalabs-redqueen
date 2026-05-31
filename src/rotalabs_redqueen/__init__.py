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
    CoevolutionResult,
    EvaluationMetadata,
    Evolution,
    EvolutionConfig,
    EvolutionResult,
    Fitness,
    FitnessResult,
    FitnessValue,
    Genome,
    Individual,
    LexicaseSelection,
    MapElitesArchive,
    Message,
    NoveltyArchive,
    NoveltyFitnessSelection,
    NoveltySelection,
    Population,
    PopulationConfig,
    Report,
    ReportExporter,
    Rng,
    Selection,
    Stimulus,
    SyncFitness,
    TaxonomyLabel,
    ToolCall,
    TournamentSelection,
    Transcript,
    coevolve,
    evolve,
)

# LLM adversarial testing domain
from rotalabs_redqueen.llm import (
    AgenticGenome,
    AgenticStrategy,
    AnthropicTarget,
    AttackStrategy,
    DefendedTarget,
    DefenderBlockFitness,
    Encoding,
    Escalation,
    GeminiTarget,
    HarmCategory,
    HeuristicJudge,
    JailbreakFitness,
    JailbreakMetrics,
    Judge,
    JudgeResult,
    LLMAttackGenome,
    LLMJudge,
    LLMTarget,
    MCPTarget,
    MockTarget,
    MultiTargetFitness,
    MultiTurnGenome,
    NetworkError,
    OllamaTarget,
    OpenAITarget,
    Persona,
    RateLimitError,
    SystemPromptDefense,
    TargetError,
    TargetResponse,
    create_target,
)

__all__ = [
    "__version__",
    # Core - RNG
    "Rng",
    # Core - Stimulus
    "Stimulus",
    "Transcript",
    "Message",
    "ToolCall",
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
    "LexicaseSelection",
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
    # Core - Co-evolution
    "coevolve",
    "CoevolutionResult",
    # Core - Reporting
    "Report",
    "ReportExporter",
    "TaxonomyLabel",
    # LLM - Genome
    "LLMAttackGenome",
    "AttackStrategy",
    "Encoding",
    "HarmCategory",
    "Persona",
    "MultiTurnGenome",
    "Escalation",
    "AgenticGenome",
    "AgenticStrategy",
    # LLM - Targets
    "LLMTarget",
    "OpenAITarget",
    "AnthropicTarget",
    "GeminiTarget",
    "OllamaTarget",
    "MockTarget",
    "MCPTarget",
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
    # LLM - Co-evolution
    "SystemPromptDefense",
    "DefendedTarget",
    "DefenderBlockFitness",
]
