"""A seeded run must be bit-reproducible.

This is the practical payoff of the canonical PRNG (`core/rng.py`) plus a pure
mock domain: same seed -> same archive, every time, on every machine. It is the
foundation the cross-language conformance corpus builds on.
"""

import asyncio

from rotalabs_redqueen import (
    AttackStrategy,
    BehaviorDimension,
    Encoding,
    HeuristicJudge,
    JailbreakFitness,
    LLMAttackGenome,
    MapElitesArchive,
    MockTarget,
    evolve,
)


def _run(seed: int):
    fitness = JailbreakFitness(MockTarget(), HeuristicJudge())
    archive = MapElitesArchive(
        dimensions=[
            BehaviorDimension("strategy", 0.0, 1.0, len(AttackStrategy)),
            BehaviorDimension("encoding", 0.0, 1.0, len(Encoding)),
            BehaviorDimension("has_persona", 0.0, 1.0, 2),
        ]
    )
    r = asyncio.run(
        evolve(
            genome_class=LLMAttackGenome,
            fitness=fitness,
            generations=12,
            population_size=16,
            seed=seed,
            archive=archive,
            progress=False,
        )
    )
    cov = r.archive.coverage()
    return (
        r.best.genome.to_prompt(),
        r.best.fitness.value,
        cov.filled_cells,
        cov.coverage_percent,
    )


def test_seeded_run_is_reproducible():
    assert _run(1234) == _run(1234)


def test_mock_target_is_a_pure_function_of_the_prompt():
    target = MockTarget()
    a = asyncio.run(target.query("a representative attack prompt"))
    b = asyncio.run(target.query("a representative attack prompt"))
    assert a.content == b.content
