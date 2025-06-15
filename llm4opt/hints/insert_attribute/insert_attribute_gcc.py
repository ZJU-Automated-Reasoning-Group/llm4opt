"""
GCC Attribute Performance Evaluation Framework

This module implements a performance evaluation framework for GCC compiler attributes,
focusing on measuring the impact of semantics-preserving attributes on program performance.
It's designed to systematically evaluate and record the performance effects of different
attribute combinations.

Key Features:
- Performance benchmarking of programs with and without attributes
- Focus on semantics-preserving attributes (alignment, optimization hints, etc.)
- Systematic evaluation of attribute combinations and their performance impact
- Comprehensive performance metrics collection (execution time, code size, etc.)
- Structured data output for analysis

Main Components:
1. PerformanceEvaluator class: Core logic for attribute insertion and performance measurement
2. Attribute selection: Heuristic-based attribute selection for different program patterns
3. Benchmark execution: Controlled performance measurement with statistical significance
4. Data recording: Structured performance data for analysis

Performance-Focused Attributes:
- Optimization attributes: optimize(), hot, cold, flatten
- Alignment attributes: aligned, assume_aligned
- Inlining hints: always_inline, noinline
- Memory attributes: malloc, returns_nonnull, restrict-like attributes
- Target-specific optimizations: target, target_clones

Usage:
    python insert_attribute_gcc.py --benchmark-suite /path/to/programs

Data Output:
- performance_data/: Structured performance measurements
- benchmarks/: Performance comparison results

This is part of the LLM4OPT project for compiler optimization evaluation.
"""

import multiprocessing
import subprocess
import datetime
import tempfile
import random
import shutil
import glob
import time
import os
import re
import utils
import argparse
from functools import partial
import itertools
import string
from pathlib import Path
import json
import statistics

import optimization
from gcc_attributes import *

# Create timestamped output directories for performance evaluation results
current_time = datetime.datetime.now()
timestamp = current_time.strftime("%Y%m%d_%H%M%S")
CUR_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), f'Performance-GCC-{timestamp}')
PERFORMANCE_DIR = os.path.join(CUR_DIR, 'performance_data')  # Directory for performance measurements
BENCHMARKS_DIR = os.path.join(CUR_DIR, 'benchmarks')         # Directory for benchmark results

# Performance measurement configurations
BENCHMARK_TIMEOUT = 300       # Timeout for benchmark execution (longer for accurate measurements)
WARMUP_RUNS = 3              # Number of warmup runs before measurement
MEASUREMENT_RUNS = 10        # Number of measurement runs for statistical significance
COMPILATION_TIMEOUT = 60     # Timeout for compilation operations

# Performance-focused attributes that are semantics-preserving
# These attributes can impact performance without changing program behavior
performance_attributes = {
    'optimization': [
        'hot', 'cold', 'flatten'
    ],
    'alignment': [
        'aligned(4)', 'aligned(8)', 'aligned(16)', 'aligned(32)', 'aligned(64)'
    ],
    'inlining': [
        'always_inline', 'noinline', 'gnu_inline'
    ],
    'memory': [
        'malloc', 'returns_nonnull', 'pure', 'const', 'leaf'
    ],
    'function_behavior': [
        'artificial', 'externally_visible', 'no_reorder'
    ]
}

# Flatten all performance attributes for easy access
all_performance_attributes = []
for category, attrs in performance_attributes.items():
    all_performance_attributes.extend(attrs)

def generate_random_string(length):
    return ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(length))

class PerformanceEvaluator(object):
    """
    Core class for evaluating the performance impact of GCC attributes.
    
    This class handles:
    - Program analysis and attribute insertion
    - Performance measurement and benchmarking
    - Data recording for analysis
    """
    def __init__(self, prog, optimization_level, work_dir):
        self.prog = prog
        self.optimization_level = optimization_level
        self.work_dir = work_dir

        # Program structure information
        self.struct_var_list = []
        self.var_list = []
        self.func_list = []
    
    def select_attributes_heuristic(self):
        """
        Select attributes based on program characteristics using heuristics.
        
        Returns:
            Dictionary mapping program locations to recommended attributes
        """
        attribute_plan = {}
        
        # Select attributes based on program patterns
        for func in self.func_list:
            line = int(func.get('Line', 0))
            if line == 0:
                continue
                
            func_name = func.get('Name', '')
            
            # Skip main function
            if func_name == 'main':
                continue
            
            # Heuristic: small functions -> always_inline
            if len(func_name) < 8:
                if line not in attribute_plan:
                    attribute_plan[line] = []
                attribute_plan[line].append([int(func.get('Column', 1)), '__attribute__((always_inline))'])
            
            # Heuristic: functions with 'compute' or 'calc' -> hot
            elif any(keyword in func_name.lower() for keyword in ['compute', 'calc', 'process', 'sum']):
                if line not in attribute_plan:
                    attribute_plan[line] = []
                attribute_plan[line].append([int(func.get('Column', 1)), '__attribute__((hot))'])
            
            # Heuristic: functions with 'init' or 'setup' -> cold
            elif any(keyword in func_name.lower() for keyword in ['init', 'setup', 'cleanup']):
                if line not in attribute_plan:
                    attribute_plan[line] = []
                attribute_plan[line].append([int(func.get('Column', 1)), '__attribute__((cold))'])
        
        return attribute_plan
    
    def measure_performance(self, executable, runs=MEASUREMENT_RUNS):
        """
        Measure performance of an executable with statistical significance.
        
        Args:
            executable: Path to the executable to benchmark
            runs: Number of measurement runs
            
        Returns:
            Dictionary with performance metrics (time, memory, etc.)
        """
        execution_times = []
        
        # Warmup runs
        for _ in range(WARMUP_RUNS):
            utils.run_cmd(executable, BENCHMARK_TIMEOUT, self.work_dir)
        
        # Measurement runs
        for _ in range(runs):
            start_time = time.perf_counter()
            result = utils.run_cmd(executable, BENCHMARK_TIMEOUT, self.work_dir)
            end_time = time.perf_counter()
            
            if result[0] == 0:  # Successful execution
                execution_times.append(end_time - start_time)
        
        if not execution_times:
            return None
            
        return {
            'mean_time': statistics.mean(execution_times),
            'median_time': statistics.median(execution_times),
            'std_dev': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
            'min_time': min(execution_times),
            'max_time': max(execution_times),
            'sample_count': len(execution_times)
        }
    
    def evaluate_attribute_impact(self, attribute_plan):
        """
        Evaluate the performance impact of applying specific attributes.
        
        Args:
            attribute_plan: Dictionary mapping locations to attributes
            
        Returns:
            Performance comparison results
        """
        # Compile baseline version
        baseline_exe = f'{self.work_dir}/baseline.exe'
        baseline_cmd = f'gcc -O{self.optimization_level} {self.prog} -o {baseline_exe}'
        baseline_result = utils.run_cmd(baseline_cmd, COMPILATION_TIMEOUT, self.work_dir)
        
        if baseline_result[0] != 0:
            return None
            
        # Measure baseline performance
        baseline_perf = self.measure_performance(baseline_exe)
        if not baseline_perf:
            return None
            
        # Create attributed version
        attributed_prog = f'{self.work_dir}/attributed.c'
        utils.generate_new(attribute_plan, self.prog, attributed_prog)
        
        # Compile attributed version
        attributed_exe = f'{self.work_dir}/attributed.exe'
        attributed_cmd = f'gcc -O{self.optimization_level} {attributed_prog} -o {attributed_exe}'
        attributed_result = utils.run_cmd(attributed_cmd, COMPILATION_TIMEOUT, self.work_dir)
        
        if attributed_result[0] != 0:
            return None
            
        # Measure attributed performance
        attributed_perf = self.measure_performance(attributed_exe)
        if not attributed_perf:
            return None
            
        # Calculate improvement
        speedup = baseline_perf['mean_time'] / attributed_perf['mean_time']
        
        return {
            'baseline': baseline_perf,
            'attributed': attributed_perf,
            'speedup': speedup,
            'improvement_percent': (speedup - 1.0) * 100,
            'attributes_used': attribute_plan
        }
    
    def parse_program_structure(self):
        """Parse program structure and validate compilation."""
        # Test basic compilation
        test_cmd = f'gcc -O{self.optimization_level} {self.prog} -o /tmp/test_compile'
        test_result = utils.run_cmd(test_cmd, COMPILATION_TIMEOUT, self.work_dir)
        if test_result[0] != 0:
            return False

        # Parse program structure
        parse_options = f'-O{self.optimization_level}' 
        ret_info = utils.parse_info(self.prog, parse_options, self.work_dir)
        if not ret_info:
            return False
        [self.struct_var_list, self.var_list, self.func_list] = ret_info
        return True

def evaluate_performance(program_path, optimization_level, work_dir):
    """
    Main function to evaluate performance impact of attributes on a program.
    
    Args:
        program_path: Path to the C program to evaluate
        optimization_level: GCC optimization level (0, 1, 2, 3, s)
        work_dir: Working directory for temporary files
        
    Returns:
        Performance evaluation results
    """
    # Create performance evaluator
    evaluator = PerformanceEvaluator(program_path, optimization_level, work_dir)
    
    # Parse program structure
    if not evaluator.parse_program_structure():
        return None
    
    # Select attributes using heuristics
    attribute_plan = evaluator.select_attributes_heuristic()
    
    # Skip if no attributes to apply
    if not attribute_plan:
        return None
    
    # Evaluate performance impact
    results = evaluator.evaluate_attribute_impact(attribute_plan)
    
    return {
        'program': program_path,
        'optimization_level': optimization_level,
        'function_count': len(evaluator.func_list),
        'variable_count': len(evaluator.var_list),
        'results': results
    }

def run_performance_benchmark(benchmark_suite, optimization_levels=['2']):
    """
    Run performance benchmarks on a suite of programs.
    
    Args:
        benchmark_suite: List of program paths or directory containing programs
        optimization_levels: List of optimization levels to test
        
    Returns:
        Comprehensive benchmark results
    """
    # Create output directories
    os.makedirs(PERFORMANCE_DIR, exist_ok=True)
    os.makedirs(BENCHMARKS_DIR, exist_ok=True)
    
    all_results = []
    
    # Process each program in the benchmark suite
    for program in benchmark_suite:
        if not os.path.exists(program):
            continue
            
        print(f"Evaluating {program}...")
        
        for opt_level in optimization_levels:
            with tempfile.TemporaryDirectory() as work_dir:
                result = evaluate_performance(program, opt_level, work_dir)
                if result:
                    all_results.append(result)
                    
                    # Save individual result
                    result_file = os.path.join(PERFORMANCE_DIR, 
                                             f"{os.path.basename(program)}_O{opt_level}.json")
                    with open(result_file, 'w') as f:
                        json.dump(result, f, indent=2)
    
    # Save comprehensive results
    summary_file = os.path.join(BENCHMARKS_DIR, 'benchmark_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    return all_results

# Main execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCC Attribute Performance Evaluation Framework')
    parser.add_argument('--benchmark-suite', type=str, required=True, help='path to benchmark suite directory or single program')
    parser.add_argument('--optimization-levels', type=str, default='2', help='comma-separated optimization levels (e.g., "1,2,3")')
    parser.add_argument('--output-dir', type=str, help='custom output directory for results')
    args = parser.parse_args()
    
    # Handle custom output directory
    if args.output_dir:
        CUR_DIR = args.output_dir
        PERFORMANCE_DIR = os.path.join(CUR_DIR, 'performance_data')
        BENCHMARKS_DIR = os.path.join(CUR_DIR, 'benchmarks')
    
    # Create output directories
    os.makedirs(PERFORMANCE_DIR, exist_ok=True)
    os.makedirs(BENCHMARKS_DIR, exist_ok=True)
    
    # Performance evaluation
    optimization_levels = args.optimization_levels.split(',')
    
    # Run on benchmark suite
    if os.path.isdir(args.benchmark_suite):
        benchmark_programs = utils.find_c_files(args.benchmark_suite, 'c')
    else:
        benchmark_programs = [args.benchmark_suite]
    
    print(f"Running performance evaluation on {len(benchmark_programs)} programs...")
    results = run_performance_benchmark(benchmark_programs, optimization_levels)
    
    print(f"\nPerformance Evaluation Complete!")
    print(f"Results saved to: {BENCHMARKS_DIR}")
    
    # Print summary statistics
    improvements = [r['results']['improvement_percent'] for r in results if r['results']]
    if improvements:
        avg_improvement = sum(improvements) / len(improvements)
        print(f"Average performance improvement: {avg_improvement:.2f}%")
        print(f"Best improvement: {max(improvements):.2f}%")
        print(f"Programs with positive improvement: {len([i for i in improvements if i > 0])}/{len(improvements)}")
    else:
        print("No successful performance evaluations completed.")