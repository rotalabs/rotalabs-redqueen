"""Tests for core evolutionary framework."""

import numpy as np
import pytest

from rotalabs_redqueen.core import (
    BehaviorDescriptor,
    BehaviorDimension,
    Evolution,
    EvolutionConfig,
    Fitness,
    FitnessResult,
    FitnessValue,
    Genome,
    Individual,
    MapElitesArchive,
    Population,
    TournamentSelection,
)


# Simple test genome for testing the framework
class SimpleGenome(Genome["SimpleGenome"]):
    """Simple genome with a single float value."""

    def __init__(self, value: float = 0.0):
        self.value = value

    @classmethod
    def random(cls, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        return cls(rng.random())

    def mutate(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        delta = rng.normal(0, 0.1)
        return SimpleGenome(max(0, min(1, self.value + delta)))

    def crossover(self, other: "SimpleGenome", rng=None):
        if rng is None:
            rng = np.random.default_rng()
        alpha = rng.random()
        return SimpleGenome(alpha * self.value + (1 - alpha) * other.value)

    def to_phenotype(self):
        return str(self.value)

    def behavior(self):
        return BehaviorDescriptor((self.value,))

    def distance(self, other: "SimpleGenome") -> float:
        return abs(self.value - other.value)


class SimpleFitness(Fitness[SimpleGenome]):
    """Fitness function that maximizes genome value."""

    async def evaluate(self, genome: SimpleGenome) -> FitnessResult:
        return FitnessResult(
            fitness=FitnessValue(genome.value),
            behavior=genome.behavior(),
        )


class TestBehaviorDescriptor:
    """Tests for BehaviorDescriptor."""

    def test_creation(self):
        bd = BehaviorDescriptor((0.5, 0.3, 0.8))
        assert bd.values == (0.5, 0.3, 0.8)

    def test_distance(self):
        bd1 = BehaviorDescriptor((0.0, 0.0))
        bd2 = BehaviorDescriptor((1.0, 0.0))
        assert bd1.distance(bd2) == pytest.approx(1.0)

        bd3 = BehaviorDescriptor((1.0, 1.0))
        assert bd1.distance(bd3) == pytest.approx(np.sqrt(2))


class TestFitnessValue:
    """Tests for FitnessValue."""

    def test_creation(self):
        fv = FitnessValue(0.75)
        assert fv.value == 0.75

    def test_comparison(self):
        fv1 = FitnessValue(0.5)
        fv2 = FitnessValue(0.7)
        assert fv2 > fv1
        assert fv1 < fv2


class TestPopulation:
    """Tests for Population."""

    def test_random_creation(self):
        rng = np.random.default_rng(42)
        pop = Population.random(SimpleGenome, 10, rng)
        assert len(pop) == 10

    def test_best(self):
        rng = np.random.default_rng(42)
        pop = Population.random(SimpleGenome, 10, rng)
        # Manually set fitness
        for i, ind in enumerate(pop):
            ind.fitness = FitnessValue(float(i) / 10)
        best = pop.best()
        assert best is not None
        assert best.fitness.value == 0.9


class TestTournamentSelection:
    """Tests for TournamentSelection."""

    def test_selection(self):
        rng = np.random.default_rng(42)
        pop = Population.random(SimpleGenome, 20, rng)
        for i, ind in enumerate(pop):
            ind.fitness = FitnessValue(float(i) / 20)

        selection = TournamentSelection(tournament_size=3)
        selected = selection.select(pop, n=5, rng=rng)
        assert len(selected) == 5
        # Higher fitness individuals should be more likely selected
        # (statistical test would be flaky, just check it runs)


class TestMapElitesArchive:
    """Tests for MapElitesArchive."""

    def test_creation(self):
        archive = MapElitesArchive(
            dimensions=[
                BehaviorDimension("x", 0.0, 1.0, 10),
            ]
        )
        coverage = archive.coverage()
        assert coverage.total_cells == 10
        assert coverage.filled_cells == 0

    def test_add(self):
        archive = MapElitesArchive(
            dimensions=[
                BehaviorDimension("x", 0.0, 1.0, 10),
            ]
        )
        genome = SimpleGenome(0.5)
        individual = Individual(
            genome=genome,
            fitness=FitnessValue(0.8),
            behavior=genome.behavior(),
        )
        added = archive.add(individual)
        assert added
        coverage = archive.coverage()
        assert coverage.filled_cells == 1


@pytest.mark.asyncio
class TestEvolution:
    """Tests for Evolution engine."""

    async def test_basic_evolution(self):
        config = EvolutionConfig(
            population_size=10,
            seed=42,
        )
        engine = Evolution(
            genome_class=SimpleGenome,
            fitness=SimpleFitness(),
            config=config,
        )
        result = await engine.run(generations=5, progress=False)

        assert result.best is not None
        assert result.generations == 5
        assert len(result.history) == 6  # Initial + 5 generations

    async def test_evolution_with_archive(self):
        config = EvolutionConfig(
            population_size=10,
            seed=42,
        )
        archive = MapElitesArchive(
            dimensions=[
                BehaviorDimension("value", 0.0, 1.0, 10),
            ]
        )
        engine = Evolution(
            genome_class=SimpleGenome,
            fitness=SimpleFitness(),
            config=config,
            archive=archive,
        )
        result = await engine.run(generations=10, progress=False)

        assert result.archive is not None
        coverage = result.archive.coverage()
        assert coverage.filled_cells > 0
