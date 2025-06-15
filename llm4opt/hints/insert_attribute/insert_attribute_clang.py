"""
Clang/LLVM Attribute Performance Evaluation Framework

This module implements a performance evaluation framework for Clang/LLVM compiler attributes,
focusing on measuring the impact of semantics-preserving attributes on program performance.
It's designed to systematically evaluate and record the performance effects of different
Clang-specific attribute combinations.

Key Features:
- Performance benchmarking of programs with and without Clang attributes
- Focus on semantics-preserving attributes (LLVM optimization passes, alignment, etc.)
- Systematic evaluation of attribute combinations and their performance impact
- Comprehensive performance metrics collection (execution time, code size, etc.)
- Structured data output for analysis
- LLVM IR generation and optimization pass performance testing

Main Components:
1. PerformanceEvaluator class: Core logic for attribute insertion and performance measurement
2. Attribute selection: Heuristic-based attribute selection for different program patterns
3. Benchmark execution: Controlled performance measurement with statistical significance
4. Data recording: Structured performance data for analysis
5. LLVM optimization pass evaluation: Performance impact of different optimization passes

Clang-Specific Features:
- Loop attribute performance evaluation (clang_loop, etc.)
- Struct-level attribute performance testing
- LLVM optimization pass selection and performance measurement
- Integration with Clang's AST parsing capabilities for targeted attribute insertion

Performance-Focused Attributes:
- Loop optimization attributes: clang_loop with various pragmas
- Function optimization: optnone, hot, cold, flatten
- Memory attributes: aligned, assume_aligned, malloc, returns_nonnull
- Target-specific optimizations: target, target_clones
- LLVM optimization passes: various IR-level optimizations

Usage:
    python insert_attribute_clang.py --benchmark-suite /path/to/programs

Data Output:
- performance_data/: Structured performance measurements
- benchmarks/: Performance comparison results
- llvm_ir/: LLVM IR files for analysis

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
from llm4opt.hints.insert_attribute.clang_attributes import *

# Create timestamped output directories for performance evaluation results
current_time = datetime.datetime.now()
timestamp = current_time.strftime("%Y%m%d_%H%M%S")
CUR_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), f'Performance-CLANG-{timestamp}')
PERFORMANCE_DIR = os.path.join(CUR_DIR, 'performance_data')  # Directory for performance measurements
BENCHMARKS_DIR = os.path.join(CUR_DIR, 'benchmarks')         # Directory for benchmark results
LLVM_IR_DIR = os.path.join(CUR_DIR, 'llvm_ir')              # Directory for LLVM IR files

# Performance measurement configurations
BENCHMARK_TIMEOUT = 300       # Timeout for benchmark execution (longer for accurate measurements)
WARMUP_RUNS = 3              # Number of warmup runs before measurement
MEASUREMENT_RUNS = 10        # Number of measurement runs for statistical significance
COMPILATION_TIMEOUT = 60     # Timeout for compilation operations

# Performance-focused attributes that are semantics-preserving
# These attributes can impact performance without changing program behavior
performance_attributes = {
    'optimization': [
        'hot', 'cold', 'flatten', 'optnone'
    ],
    'alignment': [
        'aligned(4)', 'aligned(8)', 'aligned(16)', 'aligned(32)', 'aligned(64)'
    ],
    'inlining': [
        'always_inline', 'noinline'
    ],
    'memory': [
        'malloc', 'returns_nonnull', 'pure', 'const'
    ],
    'loop': [
        'clang_loop(vectorize(enable))', 'clang_loop(unroll(enable))', 'clang_loop(interleave(enable))'
    ],
    'function_behavior': [
        'artificial', 'externally_visible'
    ]
}

# Flatten all performance attributes for easy access
all_performance_attributes = []
for category, attrs in performance_attributes.items():
    all_performance_attributes.extend(attrs)

# Mapping of Clang-specific attribute names to their generation functions
# This enables dynamic attribute creation based on program analysis for Clang/LLVM
attribute_function_map = {
    'callable_when': form_callable_when,
    'asm': form_asm,
    'enum_extensibility': form_enum_extensibility,
    'func_simd': form_declare_smid,
    'omp_target': form_omp_target,
    'abi_tag': form_abi_tag,
    '_Noreturn': insert_noreturn,
    'noreturn': insert_noreturn,
    'alloc_size': form_alloc_size,
    'alloc_align': form_alloc_align,
    'assume_aligned': form_assume_aligned,
    'btf_decl_tag': form_btf_decl_tag,
    'warning': form_warning,
    'ifunc': form_ifunc,
    'malloc': form_malloc,
    'min_vector_width': form_min_vector_width,
    'patchable_function_entry': form_patchable_function_entry,
    'target_clones': form_target_clones,
    'try_acquire_capability': form_try_acquire_capability,
    'try_acquire_shared_capability': form_try_acquire_shared_capability,
    'zero_call_used_regs': form_zero_call_used_regs,
    'acquire_handle': form_acquire_handle,
    'release_handle': form_release_handle,
    'use_handle': form_use_handle,
    'nonnull': form_nonnull,
    'returns_nonnull': form_returns_nonnull,
    'align_value': form_align_value,
    'noderef': insert_noderef,
    'aligned': form_aligned,
    'mode': form_mode,
    'sentinel': form_sentinel,
    'vector_size': form_vector_size,
    'code_align': form_code_align,
    'vector_size_func': form_vector_size_func,
    'visibility': form_visibility,
    'clang_loop': form_clang_loop,
    'enforce_tcb': form_enforce_tcb,
    'enforce_tcb_leaf': form_enforce_tcb_leaf,
    'open': form_open,
}

OPT = 'opt'
CLANG = 'clang'

def choose_llvm_optimization(exclude_opt=set()):
    """Choose LLVM optimization passes for performance evaluation."""
    remained_opt_pass = optimization.llvm_opt_pass - exclude_opt
    random_cnt = random.randint(1, 5)
    choose_opt = random.sample(list(remained_opt_pass), random_cnt)
    return choose_opt

def generate_random_string(length):
    return ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(length))

class PerformanceEvaluator(object):
    """
    Core class for evaluating the performance impact of Clang attributes.
    
    This class handles:
    - Program analysis and attribute insertion
    - Performance measurement and benchmarking
    - LLVM IR generation and optimization pass evaluation
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
        self.loop_list = []
        self.struct_list = []

        self.case_list = {}
    
    def select_attributes_heuristic(self):
        """
        Select attributes based on program characteristics using heuristics.
        
        Returns:
            Dictionary mapping program locations to recommended attributes and options
        """
        insert_plan = {}
        option_list = []
        
        # Function-level attributes
        for func in self.func_list:
            if func['Definition'].strip() == 'main':
                continue
                
            line = int(func.get('Line', 0))
            if line == 0:
                continue
                
            func_name = func.get('Name', '')
            attribute_code = ''
            
            # Heuristic: small functions -> always_inline
            if len(func_name) < 8:
                attribute_code += '__attribute__((always_inline))'
            # Heuristic: functions with 'compute' or 'calc' -> hot
            elif any(keyword in func_name.lower() for keyword in ['compute', 'calc', 'process', 'sum']):
                attribute_code += '__attribute__((hot))'
            # Heuristic: functions with 'init' or 'setup' -> cold
            elif any(keyword in func_name.lower() for keyword in ['init', 'setup', 'cleanup']):
                attribute_code += '__attribute__((cold))'
            
            if attribute_code:
                if line not in insert_plan:
                    insert_plan[line] = []
                insert_plan[line].append([int(func.get('Column', 1)), attribute_code])
        
        # Loop-level attributes for performance
        for loop in self.loop_list:
            line = int(loop.get('Line', 0))
            if line == 0:
                continue
                
            # Add vectorization hints for performance
            attribute_code = '#pragma clang loop vectorize(enable)'
            insert_plan[line] = [['loop', int(loop.get('Column', 1)), attribute_code]]
        
        # Variable-level attributes
        for var in self.var_list:
            if 'Local' in var.get('Scope', ''):
                continue
            if var.get('Line') == '0' and var.get('Column') == '0':
                continue
                
            line = int(var.get('Line', 0))
            if line == 0:
                continue
                
            # Add alignment attributes for performance
            if random.random() < 0.3:  # 30% chance
                attribute_code = '__attribute__((aligned(16)))'
                if line not in insert_plan:
                    insert_plan[line] = []
                insert_plan[line].append([int(var.get('Column', 1)), int(var.get('endColumn', 1)), attribute_code])
        
        return insert_plan, option_list
    
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
    
    def emit_llvm_ir(self, compiler, option, prog):
        """Generate LLVM IR for performance analysis."""
        ll_file = f'{self.work_dir}/{os.path.basename(prog)}'
        base_name, ext = os.path.splitext(ll_file)
        ll_file = base_name + '.ll'

        compile_cmd = f'{compiler} -S -emit-llvm {option} {prog} -o {ll_file}'
        compile_ret_code, compile_ret, compile_error = utils.run_cmd(compile_cmd, COMPILATION_TIMEOUT, self.work_dir)

        if compile_ret_code != 0 or not os.path.exists(ll_file):
            return None
            
        return ll_file
    
    def run_llvm_optimization(self, ll_file, opt_passes):
        """Run LLVM optimization passes and measure performance impact."""
        base_name, ext = os.path.splitext(ll_file)
        opt_ll_file = base_name + '.opt.ll'
        opt_passes_str = ','.join(opt_passes)

        opt_command = f'{OPT} -passes={opt_passes_str} {ll_file} -o {opt_ll_file}'
        opt_ret_code, opt_ret, opt_error = utils.run_cmd(opt_command, COMPILATION_TIMEOUT, self.work_dir)

        if opt_ret_code != 0 or not os.path.exists(opt_ll_file):
            return None
            
        return opt_ll_file
    
    def evaluate_attribute_impact(self, attribute_plan, option_list):
        """
        Evaluate the performance impact of applying specific attributes.
        
        Args:
            attribute_plan: Dictionary mapping locations to attributes
            option_list: Additional compilation options
            
        Returns:
            Performance comparison results
        """
        # Compile baseline version
        baseline_exe = f'{self.work_dir}/baseline.exe'
        baseline_cmd = f'{CLANG} -O{self.optimization_level} {self.prog} -o {baseline_exe}'
        baseline_result = utils.run_cmd(baseline_cmd, COMPILATION_TIMEOUT, self.work_dir)
        
        if baseline_result[0] != 0:
            return None
            
        # Measure baseline performance
        baseline_perf = self.measure_performance(baseline_exe)
        if not baseline_perf:
            return None
            
        # Create attributed version
        attributed_prog = f'{self.work_dir}/attributed.c'
        utils.generate_clang_new(attribute_plan, self.prog, attributed_prog)
        
        # Compile attributed version
        attributed_exe = f'{self.work_dir}/attributed.exe'
        options = ' '.join(option_list) if option_list else ''
        attributed_cmd = f'{CLANG} -O{self.optimization_level} {options} {attributed_prog} -o {attributed_exe}'
        attributed_result = utils.run_cmd(attributed_cmd, COMPILATION_TIMEOUT, self.work_dir)
        
        if attributed_result[0] != 0:
            return None
            
        # Measure attributed performance
        attributed_perf = self.measure_performance(attributed_exe)
        if not attributed_perf:
            return None
            
        # Calculate improvement
        speedup = baseline_perf['mean_time'] / attributed_perf['mean_time']
        
        # Generate LLVM IR for analysis
        baseline_ir = self.emit_llvm_ir(CLANG, f'-O{self.optimization_level}', self.prog)
        attributed_ir = self.emit_llvm_ir(CLANG, f'-O{self.optimization_level} {options}', attributed_prog)
        
        # Test LLVM optimization passes
        llvm_results = {}
        if baseline_ir and attributed_ir:
            opt_passes = choose_llvm_optimization()
            baseline_opt_ir = self.run_llvm_optimization(baseline_ir, opt_passes)
            attributed_opt_ir = self.run_llvm_optimization(attributed_ir, opt_passes)
            
            if baseline_opt_ir and attributed_opt_ir:
                llvm_results = {
                    'optimization_passes': opt_passes,
                    'baseline_ir_size': os.path.getsize(baseline_ir) if os.path.exists(baseline_ir) else 0,
                    'attributed_ir_size': os.path.getsize(attributed_ir) if os.path.exists(attributed_ir) else 0,
                    'baseline_opt_ir_size': os.path.getsize(baseline_opt_ir) if os.path.exists(baseline_opt_ir) else 0,
                    'attributed_opt_ir_size': os.path.getsize(attributed_opt_ir) if os.path.exists(attributed_opt_ir) else 0
                }
        
        return {
            'baseline': baseline_perf,
            'attributed': attributed_perf,
            'speedup': speedup,
            'improvement_percent': (speedup - 1.0) * 100,
            'attributes_used': attribute_plan,
            'compilation_options': option_list,
            'llvm_analysis': llvm_results
        }
    
    def parse_program_structure(self):
        """Parse program structure and validate compilation."""
        # Test basic compilation
        test_cmd = f'{CLANG} -O{self.optimization_level} {self.prog} -o /tmp/test_compile'
        test_result = utils.run_cmd(test_cmd, COMPILATION_TIMEOUT, self.work_dir)
        if test_result[0] != 0:
            return False

        # Parse program structure using Clang
        parse_options = f'-O{self.optimization_level}' 
        ret_info = utils.parse_info_clang(self.prog, parse_options, self.work_dir)
        if not ret_info:
            return False
        [self.struct_var_list, self.var_list, self.func_list, self.loop_list, self.struct_list] = ret_info
        return True

def evaluate_performance(program_path, optimization_level, work_dir):
    """
    Main function to evaluate performance impact of Clang attributes on a program.
    
    Args:
        program_path: Path to the C program to evaluate
        optimization_level: Clang optimization level (0, 1, 2, 3, s)
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
    attribute_plan, option_list = evaluator.select_attributes_heuristic()
    
    # Skip if no attributes to apply
    if not attribute_plan:
        return None
    
    # Evaluate performance impact
    results = evaluator.evaluate_attribute_impact(attribute_plan, option_list)
    
    return {
        'program': program_path,
        'optimization_level': optimization_level,
        'function_count': len(evaluator.func_list),
        'variable_count': len(evaluator.var_list),
        'loop_count': len(evaluator.loop_list),
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
    os.makedirs(LLVM_IR_DIR, exist_ok=True)
    
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
    parser = argparse.ArgumentParser(description='Clang/LLVM Attribute Performance Evaluation Framework')
    parser.add_argument('--benchmark-suite', type=str, required=True, help='path to benchmark suite directory or single program')
    parser.add_argument('--optimization-levels', type=str, default='2', help='comma-separated optimization levels (e.g., "1,2,3")')
    parser.add_argument('--output-dir', type=str, help='custom output directory for results')
    args = parser.parse_args()
    
    # Handle custom output directory
    if args.output_dir:
        CUR_DIR = args.output_dir
        PERFORMANCE_DIR = os.path.join(CUR_DIR, 'performance_data')
        BENCHMARKS_DIR = os.path.join(CUR_DIR, 'benchmarks')
        LLVM_IR_DIR = os.path.join(CUR_DIR, 'llvm_ir')
    
    # Create output directories
    os.makedirs(PERFORMANCE_DIR, exist_ok=True)
    os.makedirs(BENCHMARKS_DIR, exist_ok=True)
    os.makedirs(LLVM_IR_DIR, exist_ok=True)
    
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
    