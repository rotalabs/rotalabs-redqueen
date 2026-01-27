# rotalabs-redqueen

Evolutionary adversarial testing framework - Quality-diversity evolution for AI safety research.

## Overview

rotalabs-redqueen is a quality-diversity framework for automated red-teaming of language models. It uses evolutionary algorithms (MAP-Elites, novelty search) to discover diverse, effective test cases for AI safety evaluation.

## Key Features

- **Quality-Diversity Evolution**: MAP-Elites and novelty search
- **LLM Domain Primitives**: Attack genomes, targets, judges, fitness
- **Multi-Target Testing**: Test across multiple LLM providers
- **Extensible Architecture**: Custom genomes, fitness functions, archives

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Evolution Engine                       │
├─────────────────┬─────────────────┬─────────────────────┤
│   Population    │    Selection    │     Archive         │
│   Management    │    Operators    │   (MAP-Elites)      │
└────────┬────────┴────────┬────────┴────────┬────────────┘
         │                 │                 │
         ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────┐
│                   Genome Layer                           │
├─────────────────────────────────────────────────────────┤
│  LLMAttackGenome: strategies, personas, encodings       │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                   Fitness Evaluation                     │
├─────────────────┬─────────────────┬─────────────────────┤
│    LLM Target   │     Judge       │   Jailbreak         │
│   (API call)    │   (Evaluate)    │   Metrics           │
└─────────────────┴─────────────────┴─────────────────────┘
```

## Installation

```bash
# Core framework
pip install rotalabs-redqueen

# With LLM targets
pip install rotalabs-redqueen[llm]

# Everything
pip install rotalabs-redqueen[all]
```

## Quick Start

```python
from rotalabs_redqueen import (
    evolve,
    EvolutionConfig,
    LLMAttackGenome,
    JailbreakFitness,
    OpenAITarget,
    HeuristicJudge,
    MapElitesArchive,
)

# Configure target
target = OpenAITarget(model="gpt-4o-mini")
judge = HeuristicJudge()
fitness = JailbreakFitness(target=target, judge=judge)

# Configure archive
archive = MapElitesArchive(
    dimensions=[
        BehaviorDimension("length", 0, 500, 10),
        BehaviorDimension("complexity", 0, 1, 10),
    ]
)

# Run evolution
config = EvolutionConfig(
    population_size=100,
    generations=50,
    mutation_rate=0.3,
)

result = evolve(
    genome_class=LLMAttackGenome,
    fitness=fitness,
    archive=archive,
    config=config,
)

print(f"Archive coverage: {result.coverage:.1%}")
print(f"Best fitness: {result.best_fitness:.3f}")
```

## Core Concepts

| Concept | Description |
|---------|-------------|
| **Genome** | Represents a test case (attack prompt) |
| **Fitness** | Evaluates how effective a test case is |
| **Archive** | Stores diverse, high-quality solutions |
| **Selection** | Chooses parents for reproduction |
| **Evolution** | Runs the evolutionary loop |
