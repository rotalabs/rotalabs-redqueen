"""Conformance runners (redqueen-spec conformance.md).

Each ``run_lN`` executes a fixed, seeded campaign and returns a JSON-serializable
dict. Canonicalized, these must reproduce the golden fixtures byte-for-byte on
every run and platform -- the regression gate for determinism, and the anchor a
cross-language port (TypeScript / Rust) must match.

- L1: generic engine determinism with a toy genome + synthetic fitness.
- L2: LLM domain with the deterministic MockTarget + HeuristicJudge.
- L3: the taxonomy/compliance report projected from an L2 archive.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from rotalabs_redqueen.core.archive import BehaviorDimension, MapElitesArchive
from rotalabs_redqueen.core.fitness import FitnessResult, FitnessValue, SyncFitness
from rotalabs_redqueen.core.genome import BehaviorDescriptor, Genome
from rotalabs_redqueen.core.report import ReportExporter
from rotalabs_redqueen.core.rng import Rng
from rotalabs_redqueen.core.stimulus import Stimulus
from rotalabs_redqueen.llm import (
    AgenticGenome,
    AgenticStrategy,
    AttackStrategy,
    Encoding,
    Escalation,
    HeuristicJudge,
    JailbreakFitness,
    LLMAttackGenome,
    MockTarget,
    MultiTurnGenome,
)

CONFORMANCE_SEED = 20260531


# --- L1: a minimal, dependency-free toy genome -------------------------------


@dataclass
class _ToyGenome(Genome["_ToyGenome"]):
    value: float = 0.0

    @classmethod
    def random(cls, rng: Rng | None = None) -> _ToyGenome:
        rng = rng or Rng()
        return cls(value=rng.next_double())

    def mutate(self, rng: Rng | None = None) -> _ToyGenome:
        rng = rng or Rng()
        return _ToyGenome(value=min(1.0, max(0.0, self.value + rng.uniform(-0.1, 0.1))))

    def crossover(self, other: _ToyGenome, rng: Rng | None = None) -> _ToyGenome:
        rng = rng or Rng()
        a = rng.next_double()
        return _ToyGenome(value=a * self.value + (1 - a) * other.value)

    def to_stimulus(self) -> Stimulus:
        return Stimulus.single_turn(prompt=f"{self.value:.6f}")

    def behavior(self) -> BehaviorDescriptor:
        return BehaviorDescriptor((self.value,))

    def distance(self, other: _ToyGenome) -> float:
        return abs(self.value - other.value)

    def to_dict(self) -> dict:
        return {"value": self.value}

    @classmethod
    def from_dict(cls, data: dict) -> _ToyGenome:
        return cls(value=data["value"])


class _ToyFitness(SyncFitness["_ToyGenome"]):
    def evaluate_sync(self, genome: _ToyGenome) -> FitnessResult:
        return FitnessResult(fitness=FitnessValue(genome.value), behavior=genome.behavior())


def _llm_archive() -> MapElitesArchive:
    return MapElitesArchive(
        dimensions=[
            BehaviorDimension("strategy", 0.0, 1.0, len(AttackStrategy)),
            BehaviorDimension("encoding", 0.0, 1.0, len(Encoding)),
            BehaviorDimension("has_persona", 0.0, 1.0, 2),
        ]
    )


async def _l1(seed: int) -> dict:
    from rotalabs_redqueen.core.evolution import evolve

    archive = MapElitesArchive([BehaviorDimension("value", 0.0, 1.0, 10)])
    await evolve(
        genome_class=_ToyGenome,
        fitness=_ToyFitness(),
        generations=20,
        population_size=24,
        seed=seed,
        archive=archive,
        progress=False,
    )
    return archive.to_dict()


async def _l2(seed: int) -> MapElitesArchive:
    from rotalabs_redqueen.core.evolution import evolve

    archive = _llm_archive()
    await evolve(
        genome_class=LLMAttackGenome,
        fitness=JailbreakFitness(MockTarget(), HeuristicJudge()),
        generations=20,
        population_size=24,
        seed=seed,
        archive=archive,
        progress=False,
    )
    return archive


def run_l1(seed: int = CONFORMANCE_SEED) -> dict:
    """L1: engine determinism (toy genome). Returns the archive wire dict."""
    return asyncio.run(_l1(seed))


def run_l2(seed: int = CONFORMANCE_SEED) -> dict:
    """L2: LLM domain determinism. Returns the archive wire dict."""
    return asyncio.run(_l2(seed)).to_dict()


def run_l3(seed: int = CONFORMANCE_SEED) -> dict:
    """L3: compliance report projected from the L2 archive."""

    async def _go() -> dict:
        archive = await _l2(seed)
        report = ReportExporter().export(
            archive.get_all(), campaign_id="conformance-l2", coverage=archive.coverage()
        )
        return report.to_dict()

    return asyncio.run(_go())


async def _l4_multiturn(seed: int) -> MapElitesArchive:
    from rotalabs_redqueen.core.evolution import evolve

    archive = MapElitesArchive(
        [
            BehaviorDimension("turns", 0.0, 1.0, 5),
            BehaviorDimension("escalation", 0.0, 1.0, len(Escalation)),
            BehaviorDimension("has_persona", 0.0, 1.0, 2),
        ]
    )
    await evolve(
        genome_class=MultiTurnGenome,
        fitness=JailbreakFitness(MockTarget(), HeuristicJudge()),
        generations=20,
        population_size=24,
        seed=seed,
        archive=archive,
        progress=False,
    )
    return archive


async def _l4_agentic(seed: int) -> MapElitesArchive:
    from rotalabs_redqueen.core.evolution import evolve

    archive = MapElitesArchive(
        [
            BehaviorDimension("strategy", 0.0, 1.0, len(AgenticStrategy)),
            BehaviorDimension("steps", 0.0, 1.0, 5),
            BehaviorDimension("tool", 0.0, 1.0, 5),
        ]
    )
    await evolve(
        genome_class=AgenticGenome,
        fitness=JailbreakFitness(MockTarget(), HeuristicJudge()),
        generations=20,
        population_size=24,
        seed=seed,
        archive=archive,
        progress=False,
    )
    return archive


def run_l4_multiturn(seed: int = CONFORMANCE_SEED) -> dict:
    """L4: multi-turn (Crescendo-style) determinism. Returns the archive wire dict."""
    return asyncio.run(_l4_multiturn(seed)).to_dict()


def run_l4_agentic(seed: int = CONFORMANCE_SEED) -> dict:
    """L4: agentic / tool-use determinism. Returns the archive wire dict."""
    return asyncio.run(_l4_agentic(seed)).to_dict()


async def _l5(seed: int) -> dict:
    from rotalabs_redqueen.core.coevolution import coevolve
    from rotalabs_redqueen.llm.defense import DefenderBlockFitness, SystemPromptDefense

    base_target = MockTarget()
    judge = HeuristicJudge()
    result = await coevolve(
        attacker_class=LLMAttackGenome,
        defender_class=SystemPromptDefense,
        attacker_fitness_vs=lambda d: JailbreakFitness(d.as_defense(base_target), judge),
        defender_fitness_vs=lambda a: DefenderBlockFitness(a, base_target, judge),
        generations=15,
        population_size=24,
        seed=seed,
    )
    return {
        "best_attacker": result.best_attacker.to_dict(),
        "best_defender": result.best_defender.to_dict(),
        "attacker_fitness": result.attacker_fitness,
        "defender_fitness": result.defender_fitness,
        "generations": result.generations,
        "history": result.history,
    }


def run_l5(seed: int = CONFORMANCE_SEED) -> dict:
    """L5: competitive co-evolution determinism (attacker vs defender)."""
    return asyncio.run(_l5(seed))
