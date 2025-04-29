"""Compiler optimization pass selection and phase ordering.

This package provides functionality for selecting and ordering compiler 
optimization passes for various goals like code size and vectorization.

The main components are:
- Core classes: CompilerPass, PassSequence, registry, basic strategies
- Advanced strategies: Evolutionary and learning-based approaches
- Benchmarking: Tools for measuring pass sequence effectiveness

For more information, see the README.md file in the passes directory.
"""

# Core components
from .compiler_passes import (
    CompilerPass,
    PassSequence,
    PassSelectionStrategy,
    PhaseOrderingStrategy,
    OptimizationGoal,
    PassLevel,
    LLVMPass,
    LLVMPassRegistry,
    pass_registry,
    initialize_pass_registry,
    PredefinedPassSelectionStrategy,
    RandomPassSelectionStrategy,
    BasicPhaseOrderingStrategy,
    DependencyBasedPhaseOrderingStrategy
)

# Advanced strategies
from .advanced_strategies import (
    EvolutionaryPassSelectionStrategy,
    LearningBasedPassSelectionStrategy
)

# Benchmarking
from .benchmark import (
    BenchmarkResult,
    Benchmark,
    LLVMBenchmark,
    BenchmarkSuite
)

# Example functions
from .examples import (
    list_available_passes,
    create_predefined_sequence,
    create_random_sequence,
    create_evolutionary_sequence,
    order_passes_by_level,
    order_passes_by_dependencies,
    run_benchmark_example
)

# Initialize the pass registry when the module is imported
initialize_pass_registry()

__all__ = [
    # Core classes and registry
    'CompilerPass',
    'PassSequence',
    'PassSelectionStrategy',
    'PhaseOrderingStrategy',
    'OptimizationGoal',
    'PassLevel',
    'LLVMPass',
    'LLVMPassRegistry',
    'pass_registry',
    'initialize_pass_registry',
    
    # Basic selection strategies
    'PredefinedPassSelectionStrategy',
    'RandomPassSelectionStrategy',
    'BasicPhaseOrderingStrategy',
    'DependencyBasedPhaseOrderingStrategy',
    
    # Advanced strategies
    'EvolutionaryPassSelectionStrategy',
    'LearningBasedPassSelectionStrategy',
    
    # Benchmarking
    'BenchmarkResult',
    'Benchmark',
    'LLVMBenchmark',
    'BenchmarkSuite',
    
    # Example functions
    'list_available_passes',
    'create_predefined_sequence',
    'create_random_sequence',
    'create_evolutionary_sequence',
    'order_passes_by_level',
    'order_passes_by_dependencies',
    'run_benchmark_example'
]
