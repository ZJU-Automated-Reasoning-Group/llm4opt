import sys
import os
import pandas as pd
import ast
import numpy as np
from collections import Counter
# import matplotlib.pyplot as plt # No longer needed
import glob
import time
import random
import traceback
# from concurrent.futures import ProcessPoolExecutor, as_completed # Keep commented unless get_instrcount supports it
from tqdm import tqdm
import argparse
import warnings
# import ctypes # No longer needed
import csv
import io
import re
import subprocess
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
# from scipy.spatial.distance import cdist # No longer needed

# --- Project Setup & Imports ---
try:
    # Assume script is in project_root/scripts, so project_root is parent of parent
    current_file_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(current_file_path)
    project_root = os.path.dirname(script_dir) # Adjust if script location changes
    if project_root not in sys.path: sys.path.append(project_root)
    from LLVMEnv.common import get_instrcount
    print("[Setup Info] Successfully imported 'get_instrcount' from LLVMEnv.common.")
except (NameError, ImportError, ModuleNotFoundError) as e:
    print(f"[Setup Warning] Could not import LLVMEnv.common.get_instrcount: {e}")
    # Provide the dummy function if the import fails
    if 'get_instrcount' not in globals():
        print("[Setup Info] Using dummy 'get_instrcount'. Results will be simulated.")
        def get_instrcount(ll_code: str, sequence: List[str], llvm_tools_path: Optional[str]=None) -> Optional[int]:
             """Dummy function for get_instrcount."""
             if not isinstance(ll_code, str) or not ll_code.strip(): return None
             if not isinstance(sequence, list): return None # Basic type check

             # Ensure sequence contains strings
             sequence = [str(item) for item in sequence if isinstance(item, (str, bytes))]
             if not sequence: # Handle empty sequence case
                 # time.sleep(random.uniform(0.001, 0.005)) # Simulate work
                 return random.randint(190, 210)


             if sequence == ['-Oz']:
                 # time.sleep(random.uniform(0.001, 0.005)) # Simulate work
                 return random.randint(95, 105)


             base_cost = 150
             # Simulate cost based on sequence, more passes generally reduce cost but with diminishing returns and noise
             cost_reduction = sum(max(0, 3 * (0.9**i) + random.uniform(-0.5, 0.5)) for i, p in enumerate(sequence))
             cost = base_cost - cost_reduction + random.uniform(-5, 5) # Add some random noise

             # Simulate specific pass effects
             if '-instcombine' in sequence: cost -= 5
             if '-gvn' in sequence: cost -= 3
             if '-sroa' in sequence: cost -= 4
             if '-reg2mem' in sequence: cost += 8 # Example of a pass that might increase instcount in some contexts
             if '-simplifycfg' in sequence: cost -= 2

             final_cost = max(10, int(cost)) # Ensure cost is at least 10
             # time.sleep(random.uniform(0.001, 0.01) * len(sequence)) # Simulate work proportional to sequence length
             return final_cost

# --- Helper Functions ---
def print_header(title):
    """Prints a formatted header."""
    print("\n" + "=" * 70)
    print(f"===== {title.upper()} =====")
    print("=" * 70 + "\n")

def print_separator():
    """Prints a separator line."""
    print("-" * 70)

def sanitize_sequence(seq: Any) -> List[str]:
    """
    Ensures sequence is a list of non-empty strings.
    Handles strings representing lists ('["-pass1", "-pass2"]').
    """
    if seq is None: return []
    if isinstance(seq, (list, tuple)):
        pass # Already a list or tuple, will be processed
    elif isinstance(seq, str):
        try:
            # Attempt to evaluate string as a list/tuple literal
            evaluated_seq = ast.literal_eval(seq)
            if isinstance(evaluated_seq, (list, tuple)):
                seq = list(evaluated_seq) # Convert to list if it was a tuple
            else:
                 # If not a list/tuple after eval, treat original string as a single pass if non-empty
                 seq = [seq.strip()] if seq.strip() else []
        except (ValueError, SyntaxError):
             # If literal_eval fails, treat original string as a single pass if non-empty
             seq = [seq.strip()] if seq.strip() else []
        except Exception: # Catch any other unexpected errors during eval
            seq = [] # Default to empty list on other errors
    else:
        # For other types, try to convert to string and treat as a single pass
        try:
            seq = [str(seq).strip()] if str(seq).strip() else []
        except Exception:
            return [] # Default to empty on conversion error

    # Final sanitization: convert all items to string, strip whitespace, remove Nones/empty strings
    sanitized = [str(p).strip() for p in seq if p is not None] # Ensure items are strings and not None
    sanitized = [p for p in sanitized if p] # Remove empty strings after stripping
    return sanitized


# --- Function to evaluate a single candidate sequence ---
def evaluate_candidate(original_ir_str: str, sequence_to_eval: List[str], llvm_tools_path: str) -> Optional[int]:
    """Evaluates a single sequence and returns its instruction count."""
    return get_instrcount(original_ir_str, sequence_to_eval, llvm_tools_path=llvm_tools_path)


# --- GA Refinement (Modified for Multi-threading) ---
def run_ga_refinement_on_sequence(
    original_ir: str,
    initial_sequence: List[str],
    initial_count: int,
    llvm_tools_path: str,
    num_candidates_to_generate: int = 10,
    max_workers_ga: int = 4,
    allow_add_duplicates: bool = False
) -> Tuple[List[str], int, str]:
    if not initial_sequence or initial_count is None or initial_count == float('inf'):
        return initial_sequence, initial_count, str(initial_sequence)

    unique_passes_in_sequence = sorted(list(set(initial_sequence)))
    if not unique_passes_in_sequence:
         return initial_sequence, initial_count, str(initial_sequence)

    evaluated_candidates = {tuple(initial_sequence): initial_count}
    best_seq_list = list(initial_sequence)
    best_count = initial_count

    mutation_types = ['remove', 'swap', 'replace', 'shuffle_subsequence']
    if allow_add_duplicates:
        mutation_types.append('add_duplicate')

    all_generated_candidate_tuples = {tuple(initial_sequence)}
    sequences_to_evaluate_in_parallel = []

    generation_attempts = 0
    max_generation_attempts = num_candidates_to_generate * 5

    while len(sequences_to_evaluate_in_parallel) < num_candidates_to_generate and generation_attempts < max_generation_attempts:
        current_base_for_mutation = list(best_seq_list)
        if not current_base_for_mutation:
            generation_attempts +=1
            continue

        mutated_seq = list(current_base_for_mutation)
        mutation_type = random.choice(mutation_types)
        applied_mutation = False

        try:
            if mutation_type == 'remove' and len(mutated_seq) > 1:
                remove_pos = random.randrange(len(mutated_seq))
                del mutated_seq[remove_pos]
                applied_mutation = True
            elif mutation_type == 'swap' and len(mutated_seq) >= 2:
                pos1, pos2 = random.sample(range(len(mutated_seq)), 2)
                mutated_seq[pos1], mutated_seq[pos2] = mutated_seq[pos2], mutated_seq[pos1]
                applied_mutation = True
            elif mutation_type == 'replace' and len(mutated_seq) >= 1:
                 replace_pos = random.randrange(len(mutated_seq))
                 current_pass = mutated_seq[replace_pos]
                 possible_replacements = [p for p in unique_passes_in_sequence if p != current_pass]
                 if not possible_replacements:
                     possible_replacements = unique_passes_in_sequence
                 if possible_replacements:
                    mutated_seq[replace_pos] = random.choice(possible_replacements)
                    applied_mutation = True
            elif mutation_type == 'shuffle_subsequence' and len(mutated_seq) >= 3:
                if len(mutated_seq) == 3:
                    start = random.choice([0,1])
                    end = start + 1
                else:
                    start = random.randrange(len(mutated_seq) - 2)
                    end = random.randrange(start + 1, len(mutated_seq)-1)

                if start < end :
                    subsequence = mutated_seq[start:end+1]
                    original_subsequence_tuple = tuple(subsequence)
                    shuffled_subsequence = list(subsequence)
                    random.shuffle(shuffled_subsequence)
                    if tuple(shuffled_subsequence) != original_subsequence_tuple:
                        mutated_seq[start:end+1] = shuffled_subsequence
                        applied_mutation = True
            elif mutation_type == 'add_duplicate' and allow_add_duplicates and len(mutated_seq) >= 1 :
                if initial_sequence:
                    pass_to_add = random.choice(initial_sequence)
                    insert_pos = random.randrange(len(mutated_seq) + 1)
                    mutated_seq.insert(insert_pos, pass_to_add)
                    applied_mutation = True
        except IndexError:
            generation_attempts += 1
            continue
        except Exception:
            generation_attempts += 1
            continue

        if not applied_mutation:
            generation_attempts += 1
            continue

        mutated_seq_sanitized = sanitize_sequence(mutated_seq)
        if not mutated_seq_sanitized:
            generation_attempts += 1
            continue

        candidate_tuple = tuple(mutated_seq_sanitized)

        if candidate_tuple not in all_generated_candidate_tuples:
            all_generated_candidate_tuples.add(candidate_tuple)
            sequences_to_evaluate_in_parallel.append(mutated_seq_sanitized)
        generation_attempts += 1

    if sequences_to_evaluate_in_parallel:
        num_actual_workers = min(max_workers_ga, len(sequences_to_evaluate_in_parallel), os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=num_actual_workers) as executor:
            future_to_seq = {
                executor.submit(evaluate_candidate, original_ir, seq_list, llvm_tools_path): tuple(seq_list)
                for seq_list in sequences_to_evaluate_in_parallel
            }
            for future in as_completed(future_to_seq):
                candidate_tuple_evaluated = future_to_seq[future]
                try:
                    current_count = future.result()
                    evaluated_candidates[candidate_tuple_evaluated] = current_count
                    if current_count is not None and not np.isnan(current_count) and current_count < best_count:
                        best_count = current_count
                        best_seq_list = list(candidate_tuple_evaluated)
                except Exception as exc:
                    # tqdm.write(f"  GA Refinement: Candidate {list(candidate_tuple_evaluated)} generated an exception: {exc}")
                    evaluated_candidates[candidate_tuple_evaluated] = None

    final_best_seq_str = str(best_seq_list)
    return best_seq_list, best_count, final_best_seq_str

# --- Configuration & Paths ---
DEFAULT_BASE_TEST_IR_DIR = "./dataset/test_final/"
DEFAULT_CLUSTER_RESULTS_BASE_DIR = "./results_all_clusters/"
DEFAULT_LLVM_TOOLS_PATH = "./llvm_tools"
DEFAULT_N_CLUSTERS = 100
DEFAULT_OUTPUT_CSV_PREFIX = "test_set_evaluation_results"

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Main Script ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Top-1 sequence from each cluster on test IR files, applying a specified policy/refinement.")
    parser.add_argument("--base_test_dir", type=str, default=DEFAULT_BASE_TEST_IR_DIR, help="Base directory containing dataset subdirectories.")
    parser.add_argument("--specific_test_subdir", type=str, required=True, help="Specific dataset subdirectory within base_test_dir to process (e.g., cbench-v1).")
    parser.add_argument("--results_dir", type=str, default=DEFAULT_CLUSTER_RESULTS_BASE_DIR, help="Base directory of cluster optimization results.")
    parser.add_argument("--llvm_tools", type=str, default=DEFAULT_LLVM_TOOLS_PATH, help="Path to LLVM tools directory.")
    parser.add_argument("--num_clusters", type=int, default=DEFAULT_N_CLUSTERS, help="Number of cluster result directories.")
    parser.add_argument("--output_csv_prefix", type=str, default=DEFAULT_OUTPUT_CSV_PREFIX, help="Prefix for output CSV file names.")
    parser.add_argument('--max_workers', type=int, default=max(1, os.cpu_count() // 2), help="Max parallel workers for main loop.")
    parser.add_argument('--max_workers_ga', type=int, default=4, help="Max parallel workers specifically for GA refinement candidate evaluation. Default: 4")
    parser.add_argument('--refinement_method', type=str, default='none',
                        choices=['none', 'prefix', 'ga_seq', 'oz'],
                        help="Policy/Refinement for sequences. 'oz' method will fallback to Oz if initial best > Oz. Other methods will NOT fallback unless explicitly coded to (which this version does not for non-oz).")
    parser.add_argument('--ga_candidates_factor', type=int, default=20, help="Factor for GA candidates.")
    parser.add_argument('--ga_allow_add_duplicates', action='store_true', help="Allow GA to add duplicate passes.")

    args = parser.parse_args()

    BASE_TEST_IR_DIR = args.base_test_dir
    SPECIFIC_TEST_SUBDIR = args.specific_test_subdir
    CLUSTER_RESULTS_BASE_DIR = args.results_dir
    LLVM_TOOLS_PATH = args.llvm_tools
    N_CLUSTERS = args.num_clusters
    OUTPUT_CSV_PREFIX = args.output_csv_prefix
    MAX_WORKERS_MAIN_LOOP = args.max_workers
    MAX_WORKERS_GA_REFINEMENT = args.max_workers_ga
    REFINEMENT_METHOD = args.refinement_method
    GA_CANDIDATES_FACTOR = args.ga_candidates_factor
    GA_ALLOW_ADD_DUPLICATES = args.ga_allow_add_duplicates

    ACTUAL_TEST_IR_DIR = os.path.join(BASE_TEST_IR_DIR, SPECIFIC_TEST_SUBDIR)
    OUTPUT_RESULTS_CSV = f"{OUTPUT_CSV_PREFIX}_{SPECIFIC_TEST_SUBDIR}_{REFINEMENT_METHOD}.csv"

    if not os.path.isdir(ACTUAL_TEST_IR_DIR):
        print(f"Error: Specific test IR directory not found: {ACTUAL_TEST_IR_DIR}")
        sys.exit(1)
    if not os.path.isdir(CLUSTER_RESULTS_BASE_DIR):
        print(f"Error: Cluster results base directory not found: {CLUSTER_RESULTS_BASE_DIR}")
        sys.exit(1)

    print_header("Configuration")
    print(f"Base Test IR Directory: {BASE_TEST_IR_DIR}")
    print(f"Processing Specific Subdirectory: {SPECIFIC_TEST_SUBDIR}")
    print(f"Full Test Path: {ACTUAL_TEST_IR_DIR}")
    print(f"Cluster Results Base Dir: {CLUSTER_RESULTS_BASE_DIR}")
    print(f"LLVM Tools Path: {LLVM_TOOLS_PATH}")
    print(f"Number of Clusters (Result Dirs to Check): {N_CLUSTERS}")
    print(f"Policy/Refinement Method: {REFINEMENT_METHOD.upper()}")
    if REFINEMENT_METHOD == 'ga_seq':
        num_mutation_ops_base = 4
        if GA_ALLOW_ADD_DUPLICATES:
            num_mutation_ops_base += 1
        num_ga_candidates_generated = GA_CANDIDATES_FACTOR * num_mutation_ops_base
        print(f"  Simplified GA Candidates Generated: ~{num_ga_candidates_generated} (Factor {GA_CANDIDATES_FACTOR} * {num_mutation_ops_base} ops)")
        print(f"  Simplified GA Allow Add Duplicates: {GA_ALLOW_ADD_DUPLICATES}")
        print(f"  Simplified GA Max Workers: {MAX_WORKERS_GA_REFINEMENT}")
    # MODIFIED: Fallback logic print statement
    if REFINEMENT_METHOD == 'oz':
         print(f"Oz Policy: Will use Oz if initial cluster best is worse than Oz count.")
    else:
         print(f"Policy {REFINEMENT_METHOD.upper()}: Will use refined/initial cluster sequence even if worse than Oz count (no automatic Oz fallback).")
    print(f"Max Workers (Main Loop): {MAX_WORKERS_MAIN_LOOP}")
    print(f"Output Results CSV: {OUTPUT_RESULTS_CSV}")
    print_separator()

    print_header("Loading Top-1 Sequence from Each Cluster Result")
    top1_sequences_per_cluster: Dict[int, Optional[Tuple[List[str], str]]] = {}
    common_result_files = [
        "genetic_algorithm_graph_ops_cross_dataset_results.csv",
        "genetic_algorithm_random_ops_cross_dataset_results.csv",
        "beam_search_cross_dataset_results.csv",
        "best_sequence_results.csv",
        "final_results.csv",
    ]
    for cluster_idx in range(N_CLUSTERS):
        cluster_result_dir = os.path.join(CLUSTER_RESULTS_BASE_DIR, f"results_cluster_{cluster_idx}")
        passes_csv_path = None
        found_path = False
        if not os.path.isdir(cluster_result_dir):
            top1_sequences_per_cluster[cluster_idx] = None
            continue
        for fname in common_result_files:
            p_path = os.path.join(cluster_result_dir, fname)
            if os.path.exists(p_path):
                passes_csv_path = p_path
                found_path = True
                break
        if not found_path:
            top1_sequences_per_cluster[cluster_idx] = None
            continue
        try:
            df_passes = pd.read_csv(passes_csv_path, usecols=['sequence'], nrows=1, skipinitialspace=True)
            if df_passes.empty or 'sequence' not in df_passes.columns:
                top1_sequences_per_cluster[cluster_idx] = None
                continue
            top_1_sequence_str = df_passes.iloc[0, 0]
            cleaned_seq_list = sanitize_sequence(top_1_sequence_str)
            if cleaned_seq_list:
                top1_sequences_per_cluster[cluster_idx] = (cleaned_seq_list, top_1_sequence_str)
            else:
                top1_sequences_per_cluster[cluster_idx] = None
        except pd.errors.EmptyDataError:
             top1_sequences_per_cluster[cluster_idx] = None
        except ValueError as ve:
             if 'sequence' in str(ve).lower():
                 top1_sequences_per_cluster[cluster_idx] = None
             else: raise ve
        except Exception as e:
            # tqdm.write(f"  Cluster {cluster_idx}: Error reading/parsing {os.path.basename(passes_csv_path)}: {type(e).__name__} - {e}")
            top1_sequences_per_cluster[cluster_idx] = None
    loaded_top1_count = sum(1 for seq_data in top1_sequences_per_cluster.values() if seq_data is not None)
    print(f"\nSuccessfully loaded and sanitized top-1 sequences for {loaded_top1_count}/{N_CLUSTERS} clusters.")
    if loaded_top1_count == 0:
        print("Error: Could not load any valid top-1 sequences from cluster results. Exiting.")
        sys.exit(1)
    print_separator()

    print_header(f"Processing Test Files in {ACTUAL_TEST_IR_DIR}")
    try:
        test_files_all = os.listdir(ACTUAL_TEST_IR_DIR)
    except FileNotFoundError:
        print(f"Error: Test directory {ACTUAL_TEST_IR_DIR} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error listing files in {ACTUAL_TEST_IR_DIR}: {e}")
        sys.exit(1)
    test_files = sorted([
        os.path.join(ACTUAL_TEST_IR_DIR, f)
        for f in test_files_all
        if f.endswith('.ll') and os.path.isfile(os.path.join(ACTUAL_TEST_IR_DIR, f))
    ])
    if not test_files:
        print("No .ll files found in the test directory. Exiting.")
        sys.exit(0)
    print(f"Found {len(test_files)} .ll files to process.")
    results_data = []

    for test_file_path in tqdm(test_files, desc="Evaluating Test Files", unit="file"):
        filename = os.path.basename(test_file_path)
        original_ir = None
        try:
            try:
                with open(test_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    original_ir = f.read()
                if not original_ir or original_ir.isspace():
                    tqdm.write(f"  Skipping {filename}: File is empty or contains only whitespace.")
                    continue
            except FileNotFoundError:
                 tqdm.write(f"  Skipping {filename}: File not found error during read.")
                 continue
            except Exception as e:
                 tqdm.write(f"  Skipping {filename}: Error reading file - {type(e).__name__}: {e}")
                 continue

            oz_options = ["-Oz"]
            oz_count = get_instrcount(original_ir, oz_options, llvm_tools_path=LLVM_TOOLS_PATH)
            if oz_count is None or oz_count == float('inf') or np.isnan(oz_count):
                tqdm.write(f"  Skipping {filename}: Failed to get a valid instruction count for -Oz baseline.")
                continue

            initial_best_count = float('inf')
            initial_best_seq_list = None
            initial_best_seq_str = None
            initial_best_source_cluster = None
            for source_cluster_idx, seq_data in top1_sequences_per_cluster.items():
                if seq_data is None: continue
                candidate_seq_list, candidate_seq_str = seq_data
                if not isinstance(candidate_seq_list, list) or not all(isinstance(p, str) for p in candidate_seq_list):
                    # tqdm.write(f"  Warning ({filename}): Invalid sequence format for cluster {source_cluster_idx}. Skipping.")
                    continue
                if not candidate_seq_list:
                     continue
                current_count = get_instrcount(original_ir, candidate_seq_list, llvm_tools_path=LLVM_TOOLS_PATH)
                if current_count is not None and not np.isnan(current_count) and current_count != float('inf') and current_count < initial_best_count:
                    initial_best_count = current_count
                    initial_best_seq_list = candidate_seq_list
                    initial_best_seq_str = candidate_seq_str
                    initial_best_source_cluster = source_cluster_idx

            final_best_count = np.nan
            final_best_seq_list = []
            final_best_seq_str = "N/A"
            final_source_info = "N/A"
            refinement_applied_type = "N/A"
            initial_best_cluster_seq_count_for_stats = initial_best_count


            if initial_best_count == float('inf'): # No valid cluster sequence found
                 final_best_count = oz_count
                 final_best_seq_list = ["-Oz"]
                 final_best_seq_str = "['-Oz']"
                 final_source_info = "N/A -> Using Oz (No Cluster Seq Valid)" # Changed message
                 # If method was 'oz', it's still Oz policy. Otherwise, it's a direct Oz use due to no alternatives.
                 if REFINEMENT_METHOD == 'oz':
                      refinement_applied_type = "Oz Policy (No Cluster Seq -> Used Oz)"
                 else:
                      refinement_applied_type = "Used Oz (No Cluster Seq)"
            else: # A valid initial best cluster sequence WAS found
                if REFINEMENT_METHOD == 'oz':
                    if initial_best_count > oz_count: # Oz policy: if cluster best is worse, use Oz
                        final_best_count = oz_count
                        final_best_seq_list = ["-Oz"]
                        final_best_seq_str = "['-Oz']"
                        final_source_info = f"Cluster {initial_best_source_cluster} ({initial_best_count}) -> Oz Policy Fallback ({oz_count})"
                        refinement_applied_type = "Oz Policy (Used Oz)"
                    else: # Oz policy: cluster best is better or equal, use it
                        final_best_count = initial_best_count
                        final_best_seq_list = initial_best_seq_list
                        final_best_seq_str = initial_best_seq_str
                        final_source_info = f"Cluster {initial_best_source_cluster} (Used)"
                        refinement_applied_type = "Oz Policy (Used Cluster Seq)"
                else: # Methods 'none', 'prefix', 'ga_seq'
                    current_eval_count = initial_best_count
                    current_eval_seq_list = initial_best_seq_list
                    current_eval_seq_str = initial_best_seq_str
                    current_source_info_base = f"Cluster {initial_best_source_cluster}"
                    refinement_applied_type = "None (Used Cluster Seq)"

                    refinement_trigger = (
                        REFINEMENT_METHOD in ['prefix', 'ga_seq'] and
                        initial_best_seq_list is not None and len(initial_best_seq_list) > 0 and
                        initial_best_count <= oz_count # Refine only if cluster best is potentially good
                    )

                    if refinement_trigger:
                        if REFINEMENT_METHOD == 'prefix':
                            best_refined_count_for_prefix = current_eval_count
                            best_refined_seq_list_for_prefix = current_eval_seq_list
                            best_refined_seq_str_for_prefix = current_eval_seq_str
                            for k in range(1, len(current_eval_seq_list)):
                                current_prefix_list = current_eval_seq_list[:k]
                                if not current_prefix_list: continue
                                prefix_count = get_instrcount(original_ir, current_prefix_list, llvm_tools_path=LLVM_TOOLS_PATH)
                                if prefix_count is not None and not np.isnan(prefix_count) and prefix_count < best_refined_count_for_prefix:
                                    best_refined_count_for_prefix = prefix_count
                                    best_refined_seq_list_for_prefix = current_prefix_list
                                    best_refined_seq_str_for_prefix = str(current_prefix_list)
                            if best_refined_count_for_prefix < current_eval_count:
                                 current_eval_count = best_refined_count_for_prefix
                                 current_eval_seq_list = best_refined_seq_list_for_prefix
                                 current_eval_seq_str = best_refined_seq_str_for_prefix
                                 refinement_applied_type = "Prefix (Improved)"
                            else:
                                 refinement_applied_type = "Prefix (Checked, No Improvement)"
                        elif REFINEMENT_METHOD == 'ga_seq':
                            num_mutation_ops = 4 + (1 if GA_ALLOW_ADD_DUPLICATES else 0)
                            num_ga_candidates_to_generate = GA_CANDIDATES_FACTOR * num_mutation_ops
                            ga_best_seq_list, ga_best_count, ga_best_seq_str_from_ga = run_ga_refinement_on_sequence(
                                original_ir,
                                current_eval_seq_list,
                                current_eval_count,
                                LLVM_TOOLS_PATH,
                                num_candidates_to_generate=num_ga_candidates_to_generate,
                                max_workers_ga=MAX_WORKERS_GA_REFINEMENT,
                                allow_add_duplicates=GA_ALLOW_ADD_DUPLICATES
                            )
                            if ga_best_count is not None and not np.isnan(ga_best_count) and ga_best_count < current_eval_count:
                                current_eval_count = ga_best_count
                                current_eval_seq_list = ga_best_seq_list
                                current_eval_seq_str = ga_best_seq_str_from_ga
                                refinement_applied_type = "GA-Seq (Improved)"
                            else:
                                refinement_applied_type = "GA-Seq (Checked, No Improvement)"

                    # For 'none', 'prefix', 'ga_seq', the final result is what came out of the above logic
                    # NO automatic fallback to Oz if worse than Oz for these methods anymore.
                    final_best_count = current_eval_count
                    final_best_seq_list = current_eval_seq_list
                    final_best_seq_str = current_eval_seq_str
                    final_source_info = f"{current_source_info_base} ({refinement_applied_type})"
                    # The `refinement_applied_type` already reflects the outcome (e.g., "None (Used Cluster Seq)", "Prefix (Improved)", etc.)

            if final_best_count is None or np.isnan(final_best_count) or final_best_count == float('inf'):
                 tqdm.write(f"  Skipping {filename}: Could not determine any valid final instruction count.")
                 continue

            over_oz_improvement = np.nan
            try:
                if oz_count > 0:
                    over_oz_improvement = ((oz_count - final_best_count) / oz_count) * 100.0
                elif oz_count == 0:
                    over_oz_improvement = 0.0 if final_best_count == 0 else -np.inf
                else: # oz_count < 0
                     if oz_count != 0:
                        over_oz_improvement = ((oz_count - final_best_count) / abs(oz_count)) * 100.0
                     else: over_oz_improvement = np.nan
                if np.isinf(over_oz_improvement) and not (oz_count == 0 and final_best_count != 0) :
                     tqdm.write(f"  Warning ({filename}): Infinite improvement value ({over_oz_improvement}). Oz={oz_count}, Final={final_best_count}. Setting to NaN.")
                     over_oz_improvement = np.nan
                elif np.isnan(over_oz_improvement):
                    tqdm.write(f"  Warning ({filename}): NaN improvement value. Oz={oz_count}, Final={final_best_count}.")
            except Exception as calc_e:
                 tqdm.write(f"  Warning ({filename}): Error during improvement calculation - {type(calc_e).__name__}: {calc_e}. Setting to NaN.")
                 over_oz_improvement = np.nan

            results_data.append({
                "Filename": filename,
                "Test File Path": test_file_path,
                "Oz Count": oz_count,
                "Initial Best Cluster Count": initial_best_cluster_seq_count_for_stats,
                "Final Best Sequence Count": final_best_count,
                "OverOz Improvement (%)": over_oz_improvement,
                "Final Best Sequence String": final_best_seq_str,
                "Best Sequence Source Info": final_source_info,
                "Refinement Method Used": refinement_applied_type
            })
        except Exception as e:
            tqdm.write(f"  Critical Error processing {filename}: {type(e).__name__} - {e}. Skipping this file.")
            traceback.print_exc()

    summary_stats = {
        "Dataset": SPECIFIC_TEST_SUBDIR,
        "Refinement": REFINEMENT_METHOD,
        "Total Files Processed": 0,
        "Valid for Stats": 0,
        "Arithmetic Mean Improvement (%)": np.nan,
        "Geometric Mean Improvement (%)": np.nan,
        "Median Improvement (%)": np.nan,
        "Refinement Success Rate (%)": np.nan,
        "Initial Worse Than Oz Rate (%)": np.nan,
        "Final Worse Than Oz Rate (%)": np.nan,
    }

    if results_data:
        print_header("Saving Results")
        df_results = pd.DataFrame(results_data)
        df_results["Oz Count"] = pd.to_numeric(df_results["Oz Count"], errors='coerce').astype('Int64')
        df_results["Initial Best Cluster Count"] = pd.to_numeric(df_results["Initial Best Cluster Count"], errors='coerce').astype('Int64')
        df_results["Final Best Sequence Count"] = pd.to_numeric(df_results["Final Best Sequence Count"], errors='coerce').astype('Int64')
        df_results["OverOz Improvement (%)"] = pd.to_numeric(df_results["OverOz Improvement (%)"], errors='coerce')
        df_results.to_csv(OUTPUT_RESULTS_CSV, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"Results saved successfully to {OUTPUT_RESULTS_CSV}")

        summary_stats["Total Files Processed"] = len(df_results)
        df_stats = df_results.dropna(subset=["Oz Count", "Initial Best Cluster Count", "Final Best Sequence Count", "OverOz Improvement (%)"])
        num_valid_for_stats = len(df_stats)
        summary_stats["Valid for Stats"] = num_valid_for_stats

        if num_valid_for_stats > 0:
            finite_improvements = df_stats["OverOz Improvement (%)"][np.isfinite(df_stats["OverOz Improvement (%)"])]
            if not finite_improvements.empty:
                summary_stats["Arithmetic Mean Improvement (%)"] = finite_improvements.mean()
                summary_stats["Median Improvement (%)"] = finite_improvements.median()
                summary_stats["Refinement Success Rate (%)"] = (finite_improvements > 0).mean() * 100

            geomean_numer = df_stats["Oz Count"].astype(float)
            geomean_denom = df_stats["Final Best Sequence Count"].astype(float)
            valid_geomean_indices = (geomean_numer > 0) & (geomean_denom > 0)
            if valid_geomean_indices.any():
                 speedup_ratios = geomean_numer[valid_geomean_indices] / geomean_denom[valid_geomean_indices]
                 speedup_ratios_safe = speedup_ratios[speedup_ratios > 0]
                 if len(speedup_ratios_safe) > 0:
                     geomean_speedup = np.exp(np.mean(np.log(speedup_ratios_safe)))
                     summary_stats["Geometric Mean Improvement (%)"] = (1 - (1 / geomean_speedup)) * 100 if geomean_speedup > 0 else np.nan

            initial_worse_count = (df_stats["Initial Best Cluster Count"] > df_stats["Oz Count"]).sum()
            summary_stats["Initial Worse Than Oz Rate (%)"] = (initial_worse_count / num_valid_for_stats) * 100

            final_worse_count = (df_stats["Final Best Sequence Count"] > df_stats["Oz Count"]).sum()
            summary_stats["Final Worse Than Oz Rate (%)"] = (final_worse_count / num_valid_for_stats) * 100

        print("\n--- SCRIPT_SUMMARY_STATS_START ---")
        for key, value in summary_stats.items():
            if isinstance(value, float): print(f"{key}:{value:.2f}")
            else: print(f"{key}:{value}")
        print("--- SCRIPT_SUMMARY_STATS_END ---")
    else:
        print("\nNo results were generated.")
        print("\n--- SCRIPT_SUMMARY_STATS_START ---")
        for key, value in summary_stats.items():
            if isinstance(value, float): print(f"{key}:{value:.2f}")
            else: print(f"{key}:{value}")
        print("--- SCRIPT_SUMMARY_STATS_END ---")

    print_header("Script Finished for Subdir")
    