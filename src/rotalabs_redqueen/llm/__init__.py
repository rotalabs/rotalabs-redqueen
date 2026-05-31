"""LLM adversarial testing domain.

Provides attack genomes, LLM targets, judges, and fitness functions
for evolutionary red-teaming of language models.
"""

from rotalabs_redqueen.llm.defense import (
    DefendedTarget,
    DefenderBlockFitness,
    SystemPromptDefense,
)
from rotalabs_redqueen.llm.fitness import (
    JailbreakFitness,
    JailbreakMetrics,
    MultiTargetFitness,
)
from rotalabs_redqueen.llm.genome import (
    AgenticGenome,
    AgenticStrategy,
    AttackStrategy,
    Encoding,
    Escalation,
    HarmCategory,
    LLMAttackGenome,
    MultiTurnGenome,
    Persona,
)
from rotalabs_redqueen.llm.judge import (
    HeuristicJudge,
    Judge,
    JudgeResult,
    LLMJudge,
)
from rotalabs_redqueen.llm.targets import (
    AnthropicTarget,
    GeminiTarget,
    LLMTarget,
    MockTarget,
    NetworkError,
    OllamaTarget,
    OpenAITarget,
    RateLimitError,
    TargetError,
    TargetResponse,
    create_target,
)

__all__ = [
    # Genome
    "LLMAttackGenome",
    "AttackStrategy",
    "Encoding",
    "HarmCategory",
    "Persona",
    "MultiTurnGenome",
    "Escalation",
    "AgenticGenome",
    "AgenticStrategy",
    # Targets
    "LLMTarget",
    "OpenAITarget",
    "AnthropicTarget",
    "GeminiTarget",
    "OllamaTarget",
    "MockTarget",
    "TargetResponse",
    "TargetError",
    "RateLimitError",
    "NetworkError",
    "create_target",
    # Judges
    "Judge",
    "JudgeResult",
    "HeuristicJudge",
    "LLMJudge",
    # Fitness
    "JailbreakFitness",
    "MultiTargetFitness",
    "JailbreakMetrics",
    # Co-evolution
    "SystemPromptDefense",
    "DefendedTarget",
    "DefenderBlockFitness",
]
