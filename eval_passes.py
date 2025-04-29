#!/usr/bin/env python3
"""Script for running compiler pass optimization experiments.

This script demonstrates how to use the passes module to run experiments
comparing different compiler optimization pass sequences.
"""

import os
import argparse
import json
from pathlib import Path
import datetime
import random
import numpy as np
from typing import Dict, List, Optional

from llm4opt.passes.base import PassSequence, OptimizationGoal, PassLevel
from llm4opt.passes.registry import pass_registry, initialize_pass_registry
from llm4opt.passes.selection import (
    PredefinedPassSelectionStrategy,
    RandomPassSelectionStrategy,
    EvolutionaryPassSelectionStrategy
)
from llm4opt.passes.benchmark import LLVMBenchmark, BenchmarkSuite, BenchmarkResult


def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line argument parser.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Run compiler pass optimization experiments"
    )
    
    parser.add_argument(
        "--source", "-s",
        type=str,
        default="llm4opt/passes/sample_benchmark.c",
        help="Path to the source file to benchmark"
    )
    
    parser.add_argument(
        "--goal", "-g",
        type=str,
        choices=["code_size", "performance", "vectorization"],
        default="performance",
        help="Optimization goal"
    )
    
    parser.add_argument(
        "--strategy", "-t",
        type=str,
        choices=["predefined", "random", "evolutionary"],
        default="predefined",
        help="Pass selection strategy"
    )
    
    parser.add_argument(
        "--level", "-l",
        type=str,
        choices=["O0", "O1", "O2", "O3", "Os", "Oz"],
        default="O2",
        help="Optimization level for predefined strategy"
    )
    
    parser.add_argument(
        "--max-passes", "-m",
        type=int,
        default=10,
        help="Maximum number of passes for random strategy"
    )
    
    parser.add_argument(
        "--population-size", "-p",
        type=int,
        default=20,
        help="Population size for evolutionary strategy"
    )
    
    parser.add_argument(
        "--generations", "-n",
        type=int,
        default=10,
        help="Number of generations for evolutionary strategy"
    )
    
    parser.add_argument(
        "--seed", "-r",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file for results (default: results_<timestamp>.json)"
    )
    
    parser.add_argument(
        "--llvm-path",
        type=str,
        default=None,
        help="Path to LLVM tools (if not in PATH)"
    )
    
    return parser


def get_optimization_goal(goal_str: str) -> OptimizationGoal:
    """Convert goal string to OptimizationGoal enum.
    
    Args:
        goal_str: Goal string
        
    Returns:
        OptimizationGoal enum value
    """
    if goal_str == "code_size":
        return OptimizationGoal.CODE_SIZE
    elif goal_str == "vectorization":
        return OptimizationGoal.VECTORIZATION
    else:
        return OptimizationGoal.PERFORMANCE


def create_pass_sequence(
    strategy: str,
    goal: OptimizationGoal,
    level: str = "O2",
    max_passes: int = 10,
    population_size: int = 20,
    generations: int = 10,
    seed: Optional[int] = None
) -> PassSequence:
    """Create a pass sequence using the specified strategy.
    
    Args:
        strategy: Strategy name
        goal: Optimization goal
        level: Optimization level for predefined strategy
        max_passes: Maximum number of passes for random strategy
        population_size: Population size for evolutionary strategy
        generations: Number of generations for evolutionary strategy
        seed: Random seed for reproducibility
        
    Returns:
        Selected pass sequence
    """
    if strategy == "predefined":
        strategy_obj = PredefinedPassSelectionStrategy(level)
        return strategy_obj.select_passes(goal)
    
    elif strategy == "random":
        strategy_obj = RandomPassSelectionStrategy(max_passes, seed)
        return strategy_obj.select_passes(goal)
    
    elif strategy == "evolutionary":
        strategy_obj = EvolutionaryPassSelectionStrategy(
            population_size=population_size,
            generations=generations,
            seed=seed
        )
        return strategy_obj.select_passes(goal)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def run_experiment(args: argparse.Namespace) -> Dict[str, BenchmarkResult]:
    """Run the experiment with the specified arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary mapping sequence names to benchmark results
    """
    # Set random seed if specified
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Get source file path
    source_file = args.source
    if not os.path.isabs(source_file):
        # If relative path, resolve relative to the current directory
        source_file = os.path.join(os.getcwd(), source_file)
    
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Source file not found: {source_file}")
    
    # Get optimization goal
    goal = get_optimization_goal(args.goal)
    
    # Create benchmark
    benchmark_name = os.path.basename(source_file).split(".")[0]
    benchmark = LLVMBenchmark(benchmark_name, source_file, args.llvm_path)
    
    # Create pass sequence
    sequence = create_pass_sequence(
        args.strategy,
        goal,
        args.level,
        args.max_passes,
        args.population_size,
        args.generations,
        args.seed
    )
    
    # Print info
    print(f"Running benchmark: {benchmark_name}")
    print(f"Source file: {source_file}")
    print(f"Optimization goal: {goal.name}")
    print(f"Strategy: {args.strategy}")
    print(f"Number of passes: {len(sequence.passes)}")
    print("Pass sequence:")
    for i, p in enumerate(sequence.passes, 1):
        print(f"  {i}. {p}")
    
    # Run benchmark
    print("\nRunning benchmark...")
    result = benchmark.evaluate(sequence)
    
    # Print results
    print("\nResults:")
    print(f"  {result}")
    
    # Create results dictionary
    results = {
        "metadata": {
            "benchmark": benchmark_name,
            "source_file": source_file,
            "goal": goal.name,
            "strategy": args.strategy,
            "timestamp": datetime.datetime.now().isoformat(),
            "seed": args.seed
        },
        "pass_sequence": [p.name for p in sequence.passes],
        "results": result.to_dict()
    }
    
    # Save results if output file specified
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"results_{timestamp}.json"
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    return {args.strategy: result}


def main():
    """Main function to run the experiment."""
    # Initialize pass registry
    initialize_pass_registry()
    
    # Parse command line arguments
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Run experiment
    run_experiment(args)


if __name__ == "__main__":
    main() 