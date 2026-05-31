# rotalabs-redqueen

Quality-diversity evolutionary red-teaming for LLMs **and agents**, from [Rotalabs](https://rotalabs.ai).

Rather than hand-crafting jailbreaks, rotalabs-redqueen *evolves* diverse, effective attack
strategies and maps the vulnerability space with MAP-Elites. It operates at the **semantic level**
and spans the full 2026 attack surface:

- **Single-turn** prompt attacks (strategies, encodings, personas)
- **Multi-turn** Crescendo-style escalation
- **Agentic / tool-use / MCP** multi-step exploit plans

Seeded runs are **bit-reproducible** (and cross-language portable), and a campaign can be projected
into an **audit-ready compliance report** (OWASP, MITRE ATLAS, EU AI Act Art. 55, NIST AI RMF).

> **2.0 is a breaking release.** See [`CHANGELOG.md`](CHANGELOG.md) for migration from 1.x.

## Installation

```bash
pip install rotalabs-redqueen           # core + mock target
pip install rotalabs-redqueen[openai]   # + OpenAI
pip install rotalabs-redqueen[anthropic]
pip install rotalabs-redqueen[llm]      # all providers
pip install rotalabs-redqueen[dev]      # tests/lint
```

## Quick start

```python
import asyncio
from rotalabs_redqueen import (
    LLMAttackGenome, JailbreakFitness, MockTarget, HeuristicJudge, evolve,
)

async def main():
    target = MockTarget()  # swap for OpenAITarget / AnthropicTarget / GeminiTarget / OllamaTarget
    fitness = JailbreakFitness(target, HeuristicJudge())

    result = await evolve(
        genome_class=LLMAttackGenome,
        fitness=fitness,
        generations=50,
        population_size=20,
        seed=1234,            # same seed -> same result, every time
        progress=False,
    )

    if result.best:
        print("fitness:", result.best.fitness.value)
        print("prompt:", result.best.genome.to_prompt())

asyncio.run(main())
```

## Multi-turn and agentic attacks

The genome's phenotype is a `Stimulus` — a single prompt, a conversation, or an agentic action
plan — so the *same engine* drives every surface. Just swap the genome class:

```python
from rotalabs_redqueen import MultiTurnGenome, AgenticGenome, JailbreakFitness, MockTarget, evolve

# Crescendo-style multi-turn escalation
mt = await evolve(genome_class=MultiTurnGenome,
                  fitness=JailbreakFitness(MockTarget()),
                  generations=50, population_size=20, seed=1, progress=False)

# Multi-step tool-use / MCP exploit plans
ag = await evolve(genome_class=AgenticGenome,
                  fitness=JailbreakFitness(MockTarget()),
                  generations=50, population_size=20, seed=1, progress=False)
```

## Quality-diversity with MAP-Elites

```python
from rotalabs_redqueen import (
    LLMAttackGenome, JailbreakFitness, MockTarget,
    MapElitesArchive, BehaviorDimension, AttackStrategy, Encoding, evolve,
)

archive = MapElitesArchive(dimensions=[
    BehaviorDimension("strategy", 0.0, 1.0, len(AttackStrategy)),
    BehaviorDimension("encoding", 0.0, 1.0, len(Encoding)),
    BehaviorDimension("has_persona", 0.0, 1.0, 2),
])
result = await evolve(genome_class=LLMAttackGenome,
                      fitness=JailbreakFitness(MockTarget()),
                      generations=100, archive=archive, seed=1, progress=False)

cov = result.archive.coverage()
print(f"coverage: {cov.coverage_percent:.1f}% ({cov.filled_cells} diverse attacks)")
```

## Compliance report

Project the archive over the attack taxonomy into standards-aligned evidence:

```python
from rotalabs_redqueen import ReportExporter

exporter = ReportExporter()
report = exporter.export(result.archive.get_all(),
                         campaign_id="run-1",
                         coverage=result.archive.coverage())
print(exporter.render(report, "markdown").decode())   # or "json"
```

The report groups successful attacks by harm category and crosswalks them to OWASP LLM/Agentic
Top-10, MITRE ATLAS, EU AI Act Article 55, and NIST AI RMF GOVERN 1.7.

## Persistence and continuous red-teaming

Archives serialize, so attacks accumulate across runs (e.g. a CI gate that gets stronger over time):

```python
from rotalabs_redqueen import MapElitesArchive, LLMAttackGenome, Rng

result.archive.save("file://archive.json")

prior = MapElitesArchive.load("file://archive.json", LLMAttackGenome)
warm_start = prior.seed(10, Rng(0))   # sample elite genomes to seed the next run
```

## Command line

```bash
rotalabs-redqueen run --target mock:random --generations 20 --seed 1
rotalabs-redqueen run --target openai:gpt-4 --use-archive --output results.json
rotalabs-redqueen run --target mock:random --llm-judge anthropic:claude-sonnet-4-20250514
rotalabs-redqueen info --strategies | --encodings | --targets
```

## Architecture

**Core framework** (generic, reusable for any QD problem): `Genome`, `Fitness`, `Population`,
`Selection` (tournament / novelty / **lexicase**), `MapElitesArchive`, `Evolution`, and the
canonical `Rng`.

**LLM domain**: `LLMAttackGenome` / `MultiTurnGenome` / `AgenticGenome`; `LLMTarget`
(OpenAI, Anthropic, **Gemini**, Ollama, Mock); `Judge` (heuristic, LLM); `JailbreakFitness` /
`MultiTargetFitness`.

| Surface | Genome | Stimulus kind |
|---------|--------|---------------|
| Single-turn | `LLMAttackGenome` | `single_turn` |
| Multi-turn | `MultiTurnGenome` | `multi_turn` |
| Agentic / MCP | `AgenticGenome` | `agentic` |

## Extending

### Custom genome

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
    def from_dict(cls, data) -> "MyGenome": ...
```

`rng` is the canonical `Rng` — use `rng.random()`, `rng.integers(n)`, `rng.choice(n, size, replace=False)`,
`rng.shuffle(list)` so runs stay reproducible.

### Custom target

```python
from rotalabs_redqueen import LLMTarget, Message, TargetResponse

class MyTarget(LLMTarget):
    @property
    def name(self) -> str:
        return "my-target"

    async def _complete(self, messages: list[Message]) -> TargetResponse:
        text = await my_llm_api([{"role": m.role, "content": m.content} for m in messages])
        return TargetResponse(content=text, model="my-model")
```

`interact()` (single-turn + scripted multi-turn rollout) is provided by the base class.

### Custom judge

```python
from rotalabs_redqueen import Judge, JudgeResult

class MyJudge(Judge):
    async def judge(self, stimulus, transcript) -> JudgeResult:
        score = my_score(transcript.assistant_text)
        return JudgeResult(success=score >= 0.5, score=score)
```

## Reproducibility & conformance

Seeded campaigns are deterministic and cross-language portable: the canonical PRNG
(xoshiro256++ + SplitMix64) is cross-validated against an independent implementation, and an L1/L2/L3
conformance suite gates engine, LLM-domain, and report behavior against golden fixtures.

```bash
pytest                                            # full suite incl. conformance
python -m rotalabs_redqueen._gen_conformance      # regenerate golden fixtures (intentional changes only)
```

## Use cases

Red-teaming, guardrail/defense testing, robustness benchmarking, and documented adversarial-testing
evidence for compliance.

## Responsible use

For **defensive security research** — testing systems you own or are authorized to test. Do not use
it to attack systems without authorization or to circumvent the safety of production systems.

## Links

- Website: https://rotalabs.ai
- GitHub: https://github.com/rotalabs/rotalabs-redqueen
- Documentation: https://rotalabs.github.io/rotalabs-redqueen/
- Contact: research@rotalabs.ai
