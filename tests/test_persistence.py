"""Archive persistence + CI-continuity (save / load / seed)."""

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
    Rng,
    evolve,
)


def _campaign(seed: int) -> MapElitesArchive:
    archive = MapElitesArchive(
        dimensions=[
            BehaviorDimension("strategy", 0.0, 1.0, len(AttackStrategy)),
            BehaviorDimension("encoding", 0.0, 1.0, len(Encoding)),
            BehaviorDimension("has_persona", 0.0, 1.0, 2),
        ]
    )
    asyncio.run(
        evolve(
            genome_class=LLMAttackGenome,
            fitness=JailbreakFitness(MockTarget(), HeuristicJudge()),
            generations=10,
            population_size=16,
            seed=seed,
            archive=archive,
            progress=False,
        )
    )
    return archive


def test_save_load_roundtrip(tmp_path):
    original = _campaign(1)
    path = tmp_path / "archive.json"
    original.save(str(path))

    restored = MapElitesArchive.load(str(path), LLMAttackGenome)
    assert restored.to_dict() == original.to_dict()
    assert restored.coverage().filled_cells == original.coverage().filled_cells
    # restored elites are real genomes that still produce a stimulus
    for ind in restored.get_all():
        assert ind.genome.to_stimulus().kind == "single_turn"


def test_seed_warmstart():
    archive = _campaign(2)
    genomes = archive.seed(3, Rng(0))
    assert len(genomes) == min(3, archive.coverage().filled_cells)
    assert all(isinstance(g, LLMAttackGenome) for g in genomes)


def test_seed_empty_archive_is_empty():
    empty = MapElitesArchive([BehaviorDimension("x", 0.0, 1.0, 4)])
    assert empty.seed(5, Rng(0)) == []
