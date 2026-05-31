# rotalabs-redqueen

Quality-diversity evolutionary red-teaming for LLMs **and agents**.

## Overview

rotalabs-redqueen *evolves* diverse, effective adversarial attacks against language models
and agents, and maps the vulnerability space with MAP-Elites. It operates at the semantic level
and spans the full attack surface:

- **Single-turn** prompt attacks (strategies, encodings, personas)
- **Multi-turn** Crescendo-style escalation
- **Agentic / tool-use / MCP** multi-step exploit plans

Seeded runs are **bit-reproducible** (and cross-language portable), and a campaign can be
projected into an **audit-ready compliance report** (OWASP, MITRE ATLAS, EU AI Act Art. 55,
NIST AI RMF).

## Key features

- **Quality-diversity evolution** — MAP-Elites + novelty search over a behavior space
- **Three attack surfaces** — one engine, swappable genome (`LLMAttackGenome`, `MultiTurnGenome`, `AgenticGenome`)
- **Reproducible** — canonical seedable PRNG; same seed → same archive, conformance-gated
- **Persistent** — archives save/load and seed the next run (continuous red-teaming)
- **Compliance** — project the archive over the attack taxonomy into standards-aligned evidence
- **Multi-provider** — OpenAI, Anthropic, Gemini, Ollama, Mock

## Installation

```bash
pip install rotalabs-redqueen           # core + mock target
pip install rotalabs-redqueen[llm]      # all providers
pip install rotalabs-redqueen[dev]      # tests/lint
```

## Quick start

```python
import asyncio
from rotalabs_redqueen import (
    LLMAttackGenome, JailbreakFitness, MockTarget, HeuristicJudge,
    MapElitesArchive, BehaviorDimension, AttackStrategy, Encoding, evolve,
)

async def main():
    fitness = JailbreakFitness(MockTarget(), HeuristicJudge())
    archive = MapElitesArchive(dimensions=[
        BehaviorDimension("strategy", 0.0, 1.0, len(AttackStrategy)),
        BehaviorDimension("encoding", 0.0, 1.0, len(Encoding)),
        BehaviorDimension("has_persona", 0.0, 1.0, 2),
    ])
    result = await evolve(
        genome_class=LLMAttackGenome,
        fitness=fitness,
        generations=50,
        population_size=20,
        seed=1234,            # reproducible
        archive=archive,
        progress=False,
    )
    cov = result.archive.coverage()
    print(f"coverage: {cov.coverage_percent:.1f}%  best: {result.best.fitness.value:.3f}")

asyncio.run(main())
```

Swap `genome_class` for `MultiTurnGenome` or `AgenticGenome` to evolve multi-turn or agentic
attacks with the same engine. See [Getting Started](getting-started.md).

## Core concepts

| Concept | Description |
|---------|-------------|
| **Genome** | An evolvable attack; its phenotype is a `Stimulus` (single-turn / multi-turn / agentic) |
| **Target** | Executes a `Stimulus`, returns a `Transcript` |
| **Judge** | Scores a `(Stimulus, Transcript)` — did the attack succeed? |
| **Fitness** | Composes target + judge into a score |
| **Archive** | MAP-Elites grid of diverse elite attacks; persists across runs |
| **Report** | Projects the archive over the taxonomy into compliance evidence |
