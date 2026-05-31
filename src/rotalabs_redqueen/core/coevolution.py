"""Competitive co-evolution: evolve an attacker population against a defender population.

Generic and deterministic (canonical RNG). Each generation, attackers are scored
against the current champion defender and defenders against the current champion
attacker; both populations then breed. The domain supplies the two fitness
factories, so this stays free of LLM specifics (redqueen-spec interfaces.md §7-8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar

from rotalabs_redqueen.core.fitness import Fitness
from rotalabs_redqueen.core.genome import Genome
from rotalabs_redqueen.core.population import Individual, Population
from rotalabs_redqueen.core.rng import Rng
from rotalabs_redqueen.core.selection import TournamentSelection

A = TypeVar("A", bound=Genome)
D = TypeVar("D", bound=Genome)


@dataclass
class CoevolutionResult(Generic[A, D]):
    """Outcome of a co-evolution run."""

    best_attacker: A
    best_defender: D
    attacker_fitness: float
    defender_fitness: float
    generations: int
    history: list[dict] = field(default_factory=list)


def _best(individuals: list[Individual]) -> Individual:
    best = individuals[0]
    for ind in individuals[1:]:
        if ind.fitness.value > best.fitness.value:
            best = ind
    return best


def _breed(
    individuals: list[Individual],
    rng: Rng,
    population_size: int,
    elitism: int,
    mutation_rate: float,
    crossover_rate: float,
    tournament_size: int,
) -> list[Genome]:
    """Selection + variation -> next-generation genomes (engine draw order)."""
    pop = Population(individuals)
    selection = TournamentSelection(tournament_size)
    offspring: list[Genome] = []
    n_offspring = population_size - elitism
    while len(offspring) < n_offspring:
        parents = selection.select(pop, 2, rng)
        p1, p2 = parents[0].genome, parents[1].genome
        child = p1.crossover(p2, rng) if rng.random() < crossover_rate else p1
        if rng.random() < mutation_rate:
            child = child.mutate(rng)
        offspring.append(child)
    elite = [ind.genome for ind in pop.top_n(elitism)]
    return (elite + offspring)[:population_size]


async def coevolve(
    attacker_class: type[A],
    defender_class: type[D],
    attacker_fitness_vs: Callable[[D], Fitness],
    defender_fitness_vs: Callable[[A], Fitness],
    generations: int,
    population_size: int = 20,
    elitism: int = 1,
    mutation_rate: float = 0.3,
    crossover_rate: float = 0.7,
    tournament_size: int = 3,
    seed: int | None = None,
) -> CoevolutionResult[A, D]:
    """Co-evolve attacker and defender populations.

    Args:
        attacker_class / defender_class: genome classes to evolve.
        attacker_fitness_vs(defender): builds the attacker Fitness against a defender.
        defender_fitness_vs(attacker): builds the defender Fitness against an attacker.
    """
    rng = Rng(seed)
    attackers: list[Genome] = [attacker_class.random(rng) for _ in range(population_size)]
    defenders: list[Genome] = [defender_class.random(rng) for _ in range(population_size)]
    champ_attacker: Genome = attackers[0]
    champ_defender: Genome = defenders[0]
    best_a = 0.0
    best_d = 0.0
    history: list[dict] = []

    for gen in range(generations):
        a_results = await attacker_fitness_vs(champ_defender).evaluate_batch(attackers)
        d_results = await defender_fitness_vs(champ_attacker).evaluate_batch(defenders)
        a_inds = [Individual.from_result(g, r, gen) for g, r in zip(attackers, a_results)]
        d_inds = [Individual.from_result(g, r, gen) for g, r in zip(defenders, d_results)]

        best_attacker_ind = _best(a_inds)
        best_defender_ind = _best(d_inds)
        champ_attacker = best_attacker_ind.genome
        champ_defender = best_defender_ind.genome
        best_a = best_attacker_ind.fitness.value
        best_d = best_defender_ind.fitness.value
        history.append(
            {
                "generation": gen,
                "best_attacker_fitness": best_a,
                "best_defender_fitness": best_d,
            }
        )

        attackers = _breed(
            a_inds, rng, population_size, elitism, mutation_rate, crossover_rate, tournament_size
        )
        defenders = _breed(
            d_inds, rng, population_size, elitism, mutation_rate, crossover_rate, tournament_size
        )

    return CoevolutionResult(
        best_attacker=champ_attacker,
        best_defender=champ_defender,
        attacker_fitness=best_a,
        defender_fitness=best_d,
        generations=generations,
        history=history,
    )
