"""LLM adversarial testing domain.

Provides attack genomes, LLM targets, judges, and fitness functions
for evolutionary red-teaming of language models.
"""

from rotalabs_redqueen.llm.fitness import (
    JailbreakFitness,
    JailbreakMetrics,
    MultiTargetFitness,
)
from rotalabs_redqueen.llm.genome import (
    AttackStrategy,
    Encoding,
    HarmCategory,
    LLMAttackGenome,
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
    # Targets
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
    # Judges
    "Judge",
    "JudgeResult",
    "HeuristicJudge",
    "LLMJudge",
    # Fitness
    "JailbreakFitness",
    "MultiTargetFitness",
    "JailbreakMetrics",
]
