"""Main evolution engine for evolutionary computation.

Orchestrates the evolutionary loop: evaluation, selection, variation, replacement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

import numpy as np
from tqdm import tqdm

from rotalabs_redqueen.core.archive import Archive
from rotalabs_redqueen.core.fitness import Fitness, FitnessResult
from rotalabs_redqueen.core.genome import Genome
from rotalabs_redqueen.core.population import Individual, Population
from rotalabs_redqueen.core.selection import Selection, TournamentSelection

G = TypeVar("G", bound=Genome)


@dataclass
class EvolutionConfig:
    """Configuration for evolution."""

    population_size: int = 100
    elitism: int = 1
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    tournament_size: int = 3
    seed: int | None = None


@dataclass
class EvolutionResult(Generic[G]):
    """Result of an evolution run."""

    best: Individual[G] | None
    population: Population[G]
    generations: int
    archive: Archive[G] | None = None
    history: list[dict] = field(default_factory=list)


class Evolution(Generic[G]):
    """Main evolution engine.

    Supports both standard GA and MAP-Elites quality-diversity.
    """

    def __init__(
        self,
        genome_class: type[G],
        fitness: Fitness[G],
        config: EvolutionConfig | None = None,
        selection: Selection[G] | None = None,
        archive: Archive[G] | None = None,
    ):
        """Initialize evolution engine.

        Args:
            genome_class: Class of genome to evolve
            fitness: Fitness function
            config: Evolution configuration
            selection: Selection operator (default: tournament)
            archive: QD archive (optional, enables MAP-Elites mode)
        """
        self.genome_class = genome_class
        self.fitness = fitness
        self.config = config or EvolutionConfig()
        self.selection = selection or TournamentSelection(self.config.tournament_size)
        self.archive = archive
        self.rng = np.random.default_rng(self.config.seed)

    async def run(
        self,
        generations: int,
        progress: bool = True,
    ) -> EvolutionResult[G]:
        """Run evolution for specified generations.

        Args:
            generations: Number of generations to run
            progress: Show progress bar

        Returns:
            Evolution result with best individual and final population
        """
        # Initialize population
        population = Population.random(
            self.genome_class,
            self.config.population_size,
            self.rng,
        )

        # Evaluate initial population
        results = await self._evaluate_population(population)
        self._update_population(population, results, generation=0)

        # Update archive if using QD
        if self.archive:
            for ind in population:
                self.archive.add(ind)

        history = []
        iterator = range(generations)
        if progress:
            iterator = tqdm(iterator, desc="Evolution")

        for gen in iterator:
            # Record history
            history.append(self._record_stats(population, gen))

            # Create offspring
            offspring = self._create_offspring(population)

            # Evaluate offspring
            results = await self._evaluate_population_from_genomes(offspring)

            # Create offspring individuals
            offspring_inds = [
                Individual.from_result(g, r, generation=gen + 1)
                for g, r in zip(offspring, results)
            ]

            # Update archive if using QD
            if self.archive:
                for ind in offspring_inds:
                    self.archive.add(ind)

            # Survivor selection
            population = self._select_survivors(population, offspring_inds)
            population.generation = gen + 1

            # Update progress bar
            if progress:
                best = population.best()
                if best:
                    iterator.set_postfix(
                        best=f"{best.fitness.value:.3f}",
                        avg=f"{population.average_fitness():.3f}",
                    )

        # Final history entry
        history.append(self._record_stats(population, generations))

        return EvolutionResult(
            best=population.best(),
            population=population,
            generations=generations,
            archive=self.archive,
            history=history,
        )

    async def _evaluate_population(
        self, population: Population[G]
    ) -> list[FitnessResult]:
        """Evaluate all individuals in population."""
        genomes = [ind.genome for ind in population]
        return await self.fitness.evaluate_batch(genomes)

    async def _evaluate_population_from_genomes(
        self, genomes: list[G]
    ) -> list[FitnessResult]:
        """Evaluate a list of genomes."""
        return await self.fitness.evaluate_batch(genomes)

    def _update_population(
        self,
        population: Population[G],
        results: list[FitnessResult],
        generation: int,
    ) -> None:
        """Update population with fitness results."""
        for ind, result in zip(population, results):
            ind.fitness = result.fitness
            ind.behavior = result.behavior
            ind.birth_generation = generation

    def _create_offspring(self, population: Population[G]) -> list[G]:
        """Create offspring through selection and variation."""
        offspring = []
        n_offspring = self.config.population_size - self.config.elitism

        while len(offspring) < n_offspring:
            # Select parents
            parents = self.selection.select(population, 2, self.rng)
            parent1, parent2 = parents[0].genome, parents[1].genome

            # Crossover
            if self.rng.random() < self.config.crossover_rate:
                child = parent1.crossover(parent2, self.rng)
            else:
                child = parent1

            # Mutation
            if self.rng.random() < self.config.mutation_rate:
                child = child.mutate(self.rng)

            offspring.append(child)

        return offspring[:n_offspring]

    def _select_survivors(
        self,
        population: Population[G],
        offspring: list[Individual[G]],
    ) -> Population[G]:
        """Select survivors for next generation."""
        # Elitism: keep best from current population
        elite = population.top_n(self.config.elitism)

        # Combine elite with offspring
        new_individuals = elite + offspring

        return Population(
            individuals=new_individuals[: self.config.population_size],
            config=population.config,
            generation=population.generation + 1,
        )

    def _record_stats(self, population: Population[G], generation: int) -> dict:
        """Record statistics for history."""
        best = population.best()
        stats = {
            "generation": generation,
            "best_fitness": best.fitness.value if best else 0.0,
            "average_fitness": population.average_fitness(),
            "fitness_std": population.fitness_std(),
            "diversity": population.diversity(),
        }
        if self.archive:
            coverage = self.archive.coverage()
            stats["archive_coverage"] = coverage.coverage_percent
            stats["archive_size"] = coverage.filled_cells
        return stats


async def evolve(
    genome_class: type[G],
    fitness: Fitness[G],
    generations: int = 100,
    population_size: int = 100,
    mutation_rate: float = 0.3,
    crossover_rate: float = 0.7,
    seed: int | None = None,
    use_archive: bool = False,
    archive: Archive[G] | None = None,
    progress: bool = True,
) -> EvolutionResult[G]:
    """Convenience function for running evolution.

    Args:
        genome_class: Class of genome to evolve
        fitness: Fitness function
        generations: Number of generations
        population_size: Population size
        mutation_rate: Probability of mutation
        crossover_rate: Probability of crossover
        seed: Random seed for reproducibility
        use_archive: Enable MAP-Elites mode
        archive: Custom archive (optional)
        progress: Show progress bar

    Returns:
        Evolution result
    """
    config = EvolutionConfig(
        population_size=population_size,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        seed=seed,
    )

    engine = Evolution(
        genome_class=genome_class,
        fitness=fitness,
        config=config,
        archive=archive,
    )

    return await engine.run(generations, progress=progress)
