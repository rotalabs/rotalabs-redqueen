"""Command-line interface for rotalabs-redqueen.

Run evolutionary adversarial testing campaigns against LLMs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from rotalabs_redqueen._version import __version__
from rotalabs_redqueen.core import (
    BehaviorDimension,
    Evolution,
    EvolutionConfig,
    MapElitesArchive,
)
from rotalabs_redqueen.llm import (
    AttackStrategy,
    Encoding,
    HeuristicJudge,
    JailbreakFitness,
    LLMAttackGenome,
    LLMJudge,
    create_target,
)


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="redqueen",
        description="Evolutionary adversarial testing for LLMs",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Run campaign
    run_parser = subparsers.add_parser("run", help="Run adversarial testing campaign")
    run_parser.add_argument(
        "--target",
        type=str,
        default="mock:random",
        help="Target specification (e.g., openai:gpt-4, anthropic:claude-sonnet-4-20250514, mock:random)",
    )
    run_parser.add_argument(
        "--generations",
        type=int,
        default=50,
        help="Number of generations to evolve",
    )
    run_parser.add_argument(
        "--population-size",
        type=int,
        default=20,
        help="Population size",
    )
    run_parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.3,
        help="Mutation rate",
    )
    run_parser.add_argument(
        "--crossover-rate",
        type=float,
        default=0.7,
        help="Crossover rate",
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    run_parser.add_argument(
        "--use-archive",
        action="store_true",
        help="Enable MAP-Elites quality-diversity mode",
    )
    run_parser.add_argument(
        "--llm-judge",
        type=str,
        default=None,
        help="Use LLM judge instead of heuristic (e.g., anthropic:claude-sonnet-4-20250514)",
    )
    run_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)",
    )
    run_parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )

    # Info command
    info_parser = subparsers.add_parser("info", help="Show information")
    info_parser.add_argument(
        "--strategies",
        action="store_true",
        help="List available attack strategies",
    )
    info_parser.add_argument(
        "--encodings",
        action="store_true",
        help="List available encodings",
    )
    info_parser.add_argument(
        "--targets",
        action="store_true",
        help="List supported target providers",
    )

    return parser


async def run_campaign(args: argparse.Namespace) -> int:
    """Run an adversarial testing campaign."""
    # Create target
    try:
        target = create_target(args.target)
    except ValueError as e:
        print(f"Error creating target: {e}", file=sys.stderr)
        return 1

    print(f"Target: {target.name}")

    # Create judge
    if args.llm_judge:
        try:
            judge_target = create_target(args.llm_judge)
            judge = LLMJudge(judge_target)
            print(f"Judge: LLM ({judge_target.name})")
        except ValueError as e:
            print(f"Error creating judge: {e}", file=sys.stderr)
            return 1
    else:
        judge = HeuristicJudge()
        print("Judge: Heuristic")

    # Create fitness function
    fitness = JailbreakFitness(target, judge)

    # Create config
    config = EvolutionConfig(
        population_size=args.population_size,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        seed=args.seed,
    )

    # Create archive if using QD
    archive = None
    if args.use_archive:
        archive = MapElitesArchive(
            dimensions=[
                BehaviorDimension("strategy", 0.0, 1.0, len(AttackStrategy)),
                BehaviorDimension("encoding", 0.0, 1.0, len(Encoding)),
                BehaviorDimension("has_persona", 0.0, 1.0, 2),
            ]
        )
        print("Mode: MAP-Elites (Quality-Diversity)")
    else:
        print("Mode: Standard Genetic Algorithm")

    print(f"Population: {args.population_size}")
    print(f"Generations: {args.generations}")
    print()

    # Run evolution
    engine = Evolution(
        genome_class=LLMAttackGenome,
        fitness=fitness,
        config=config,
        archive=archive,
    )

    result = await engine.run(
        generations=args.generations,
        progress=not args.no_progress,
    )

    # Print results
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    if result.best:
        print(f"\nBest fitness: {result.best.fitness.value:.3f}")
        print(f"Best strategy: {result.best.genome.primary_strategy.value}")
        print(f"Best encoding: {result.best.genome.encoding.value}")
        if result.best.genome.persona:
            print(f"Best persona: {result.best.genome.persona.name}")

        print("\nBest prompt:")
        print("-" * 40)
        print(result.best.genome.to_prompt()[:500])
        if len(result.best.genome.to_prompt()) > 500:
            print("...")
        print("-" * 40)

    if result.archive:
        coverage = result.archive.coverage()
        print(f"\nArchive coverage: {coverage.coverage_percent:.1f}%")
        print(f"Filled cells: {coverage.filled_cells}/{coverage.total_cells}")

    # Save results if requested
    if args.output:
        output_data = {
            "generations": result.generations,
            "best_fitness": result.best.fitness.value if result.best else None,
            "best_prompt": result.best.genome.to_prompt() if result.best else None,
            "history": result.history,
            "archive_coverage": (
                result.archive.coverage().coverage_percent if result.archive else None
            ),
        }
        output_path = Path(args.output)
        output_path.write_text(json.dumps(output_data, indent=2))
        print(f"\nResults saved to: {args.output}")

    return 0


def show_info(args: argparse.Namespace) -> int:
    """Show information about available options."""
    if args.strategies:
        print("Available attack strategies:")
        for strategy in AttackStrategy:
            print(f"  - {strategy.value}")
        return 0

    if args.encodings:
        print("Available encodings:")
        for encoding in Encoding:
            print(f"  - {encoding.value}")
        return 0

    if args.targets:
        print("Supported target providers:")
        print("  - openai:<model>    (e.g., openai:gpt-4)")
        print("  - anthropic:<model> (e.g., anthropic:claude-sonnet-4-20250514)")
        print("  - ollama:<model>    (e.g., ollama:llama2)")
        print("  - mock:<mode>       (e.g., mock:random, mock:refuse, mock:comply)")
        return 0

    print("Use --strategies, --encodings, or --targets for specific info")
    return 0


def main() -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command == "run":
        return asyncio.run(run_campaign(args))
    elif args.command == "info":
        return show_info(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
