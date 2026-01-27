# rotalabs-redqueen

Evolutionary adversarial testing framework for LLMs from [Rotalabs](https://rotalabs.ai).

Quality-diversity evolution for automated red-teaming and AI safety research.

## Overview

rotalabs-redqueen uses evolutionary algorithms to discover diverse, effective adversarial attacks against language models. Rather than manually crafting jailbreaks, it evolves attack strategies using:

- **Genetic Algorithms** - Standard evolutionary optimization
- **MAP-Elites** - Quality-diversity to find diverse successful attacks
- **Novelty Search** - Reward novel behaviors, not just fitness

The framework operates at the **semantic level** - evolving attack strategies, encodings, and personas rather than raw tokens.

## Installation

```bash
# Core package (includes mock target for testing)
pip install rotalabs-redqueen

# With OpenAI support
pip install rotalabs-redqueen[openai]

# With Anthropic support
pip install rotalabs-redqueen[anthropic]

# All LLM providers
pip install rotalabs-redqueen[llm]

# Development
pip install rotalabs-redqueen[dev]
```

## Quick Start

### Python API

```python
import asyncio
from rotalabs_redqueen import (
    LLMAttackGenome,
    JailbreakFitness,
    MockTarget,
    HeuristicJudge,
    evolve,
)

async def main():
    # Create target and fitness function
    target = MockTarget()  # Use OpenAITarget or AnthropicTarget for real tests
    fitness = JailbreakFitness(target, HeuristicJudge())

    # Run evolution
    result = await evolve(
        genome_class=LLMAttackGenome,
        fitness=fitness,
        generations=50,
        population_size=20,
    )

    # Examine results
    if result.best:
        print(f"Best fitness: {result.best.fitness.value}")
        print(f"Best prompt: {result.best.genome.to_prompt()}")

asyncio.run(main())
```

### Quality-Diversity with MAP-Elites

```python
from rotalabs_redqueen import (
    LLMAttackGenome,
    JailbreakFitness,
    MockTarget,
    MapElitesArchive,
    BehaviorDimension,
    AttackStrategy,
    Encoding,
    evolve,
)

async def main():
    target = MockTarget()
    fitness = JailbreakFitness(target)

    # Create archive to track diverse solutions
    archive = MapElitesArchive(
        dimensions=[
            BehaviorDimension("strategy", 0.0, 1.0, len(AttackStrategy)),
            BehaviorDimension("encoding", 0.0, 1.0, len(Encoding)),
            BehaviorDimension("has_persona", 0.0, 1.0, 2),
        ]
    )

    result = await evolve(
        genome_class=LLMAttackGenome,
        fitness=fitness,
        generations=100,
        archive=archive,
    )

    # Check archive coverage
    coverage = result.archive.coverage()
    print(f"Archive coverage: {coverage.coverage_percent:.1f}%")
    print(f"Diverse solutions: {coverage.filled_cells}")
```

### Command Line Interface

```bash
# Run a test campaign with mock target
rotalabs-redqueen run --target mock:random --generations 20

# Run against OpenAI (requires OPENAI_API_KEY)
rotalabs-redqueen run --target openai:gpt-4 --generations 50

# Use MAP-Elites for diverse attacks
rotalabs-redqueen run --target mock:random --use-archive

# Use LLM judge for more accurate evaluation
rotalabs-redqueen run --target mock:random --llm-judge anthropic:claude-sonnet-4-20250514

# Save results to file
rotalabs-redqueen run --target mock:random --output results.json

# Show available options
rotalabs-redqueen info --strategies
rotalabs-redqueen info --encodings
rotalabs-redqueen info --targets
```

## Architecture

### Core Framework

The core evolutionary framework is generic and can be used for any optimization problem:

- **Genome** - Abstract base for evolvable representations
- **Fitness** - Async fitness evaluation
- **Population** - Collection of individuals with selection
- **Selection** - Tournament, novelty, and hybrid selection
- **Archive** - MAP-Elites quality-diversity archive
- **Evolution** - Main evolutionary loop

### LLM Domain

The LLM domain provides specialized components for adversarial testing:

- **LLMAttackGenome** - Attack representation with strategies, encodings, personas
- **LLMTarget** - Unified interface for OpenAI, Anthropic, Ollama, etc.
- **Judge** - Evaluate attack success (heuristic or LLM-based)
- **JailbreakFitness** - Fitness function combining target and judge

## Attack Strategies

| Strategy | Description |
|----------|-------------|
| `ROLEPLAY` | Assume a character/persona (e.g., DAN) |
| `ENCODING` | Obfuscate the request (base64, rot13, etc.) |
| `AUTHORITY` | Claim special permissions |
| `HYPOTHETICAL` | Frame as fictional/educational |
| `MULTI_TURN` | Build up through conversation |
| `DIRECT` | Direct jailbreak attempt |

## Encodings

| Encoding | Description |
|----------|-------------|
| `NONE` | No encoding |
| `BASE64` | Base64 encoding |
| `ROT13` | ROT13 cipher |
| `LEETSPEAK` | L33t sp34k |
| `PIG_LATIN` | Pig Latin |
| `REVERSE` | Reversed text |

## Extending

### Custom Genomes

```python
from rotalabs_redqueen import Genome, BehaviorDescriptor

class MyGenome(Genome["MyGenome"]):
    @classmethod
    def random(cls, rng=None):
        # Create random genome
        ...

    def mutate(self, rng=None):
        # Return mutated copy
        ...

    def crossover(self, other, rng=None):
        # Return offspring
        ...

    def to_phenotype(self):
        # Convert to evaluable form
        ...

    def behavior(self):
        # Return behavior descriptor for QD
        return BehaviorDescriptor((dim1, dim2, ...))
```

### Custom Fitness Functions

```python
from rotalabs_redqueen import Fitness, FitnessResult, FitnessValue

class MyFitness(Fitness[MyGenome]):
    async def evaluate(self, genome):
        # Evaluate genome
        score = compute_score(genome.to_phenotype())
        return FitnessResult(
            fitness=FitnessValue(score),
            behavior=genome.behavior(),
        )
```

### Custom Targets

```python
from rotalabs_redqueen import LLMTarget, TargetResponse

class MyTarget(LLMTarget):
    @property
    def name(self):
        return "my-target"

    async def query(self, prompt):
        # Query your LLM
        response = await my_llm_api(prompt)
        return TargetResponse(
            content=response.text,
            model="my-model",
            tokens_used=response.tokens,
        )
```

## Use Cases

- **Red-teaming**: Discover vulnerabilities in LLM safety measures
- **Defense testing**: Validate content filters and guardrails
- **Research**: Study attack patterns and defenses systematically
- **Benchmarking**: Compare robustness across models

## Responsible Use

This tool is intended for **defensive security research** - testing and improving the safety of AI systems you own or have permission to test.

**Do not use this tool to:**
- Attack systems without authorization
- Generate harmful content for malicious purposes
- Circumvent safety measures of production systems

## Links

- Website: https://rotalabs.ai
- GitHub: https://github.com/rotalabs/rotalabs-redqueen
- Documentation: https://rotalabs.github.io/rotalabs-redqueen/
- Contact: research@rotalabs.ai
