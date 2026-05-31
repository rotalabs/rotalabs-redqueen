# Getting Started

## Installation

```bash
pip install rotalabs-redqueen           # core + mock target
pip install rotalabs-redqueen[openai]   # + OpenAI
pip install rotalabs-redqueen[anthropic]
pip install rotalabs-redqueen[llm]      # all providers
```

## Single-turn attacks

```python
import asyncio
from rotalabs_redqueen import (
    LLMAttackGenome, JailbreakFitness, MockTarget, HeuristicJudge, evolve,
)

async def main():
    target = MockTarget()  # or OpenAITarget / AnthropicTarget / GeminiTarget / OllamaTarget
    fitness = JailbreakFitness(target, HeuristicJudge())
    result = await evolve(
        genome_class=LLMAttackGenome,
        fitness=fitness,
        generations=50,
        population_size=20,
        seed=1234,
        progress=False,
    )
    if result.best:
        print("fitness:", result.best.fitness.value)
        print("prompt:", result.best.genome.to_prompt())

asyncio.run(main())
```

## Multi-turn and agentic attacks

The genome's phenotype is a `Stimulus`, so the same engine drives every surface — just swap the
genome class:

```python
from rotalabs_redqueen import MultiTurnGenome, AgenticGenome, JailbreakFitness, MockTarget, evolve

# Crescendo-style multi-turn escalation
await evolve(genome_class=MultiTurnGenome, fitness=JailbreakFitness(MockTarget()),
             generations=50, population_size=20, seed=1, progress=False)

# Multi-step tool-use / MCP exploit plans
await evolve(genome_class=AgenticGenome, fitness=JailbreakFitness(MockTarget()),
             generations=50, population_size=20, seed=1, progress=False)
```

## Co-evolution (attacker vs defender)

```python
from rotalabs_redqueen import (
    coevolve, LLMAttackGenome, SystemPromptDefense,
    JailbreakFitness, DefenderBlockFitness, MockTarget, HeuristicJudge,
)

base = MockTarget()
judge = HeuristicJudge()
result = await coevolve(
    attacker_class=LLMAttackGenome,
    defender_class=SystemPromptDefense,
    attacker_fitness_vs=lambda d: JailbreakFitness(d.as_defense(base), judge),
    defender_fitness_vs=lambda a: DefenderBlockFitness(a, base, judge),
    generations=20,
    population_size=24,
    seed=1,
)
print(result.best_defender.to_dict(), result.attacker_fitness, result.defender_fitness)
```

## Quality-diversity with MAP-Elites

```python
from rotalabs_redqueen import (
    MapElitesArchive, BehaviorDimension, AttackStrategy, Encoding,
    LLMAttackGenome, JailbreakFitness, MockTarget, evolve,
)

archive = MapElitesArchive(dimensions=[
    BehaviorDimension("strategy", 0.0, 1.0, len(AttackStrategy)),
    BehaviorDimension("encoding", 0.0, 1.0, len(Encoding)),
    BehaviorDimension("has_persona", 0.0, 1.0, 2),
])
result = await evolve(genome_class=LLMAttackGenome, fitness=JailbreakFitness(MockTarget()),
                      generations=100, archive=archive, seed=1, progress=False)

cov = result.archive.coverage()
print(f"coverage: {cov.coverage_percent:.1f}% ({cov.filled_cells} diverse attacks)")
```

## Compliance report

```python
from rotalabs_redqueen import ReportExporter

exporter = ReportExporter()
report = exporter.export(result.archive.get_all(),
                         campaign_id="run-1",
                         coverage=result.archive.coverage())
print(exporter.render(report, "markdown").decode())   # or "json"
```

Successful attacks are grouped by harm category and crosswalked to OWASP LLM/Agentic Top-10,
MITRE ATLAS, EU AI Act Article 55, and NIST AI RMF.

## Persistence and continuous red-teaming

```python
from rotalabs_redqueen import MapElitesArchive, LLMAttackGenome, Rng

result.archive.save("file://archive.json")
prior = MapElitesArchive.load("file://archive.json", LLMAttackGenome)
warm_start = prior.seed(10, Rng(0))   # sample elite genomes to seed the next run
```

## Custom genome

A genome's phenotype is a `Stimulus`; it must also serialize via `to_dict` / `from_dict`.

```python
from rotalabs_redqueen import Genome, BehaviorDescriptor, Stimulus

class MyGenome(Genome["MyGenome"]):
    @classmethod
    def random(cls, rng): ...
    def mutate(self, rng): ...
    def crossover(self, other, rng): ...
    def to_stimulus(self) -> Stimulus:
        return Stimulus.single_turn(prompt="...")
    def behavior(self) -> BehaviorDescriptor: ...
    def distance(self, other) -> float: ...
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, data): ...
```

Use the canonical `rng` (`rng.random()`, `rng.integers(n)`, `rng.choice(n, size, replace=False)`,
`rng.shuffle(list)`) so runs stay reproducible.

## Reproducibility

Seeded campaigns are deterministic and cross-language portable: the canonical PRNG is
cross-validated against an independent implementation, and an L1/L2/L3 conformance suite gates
engine, LLM-domain, and report behavior against golden fixtures (`pytest`).
