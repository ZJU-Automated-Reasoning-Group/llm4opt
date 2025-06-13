"""
GCC Attribute Insertion Tool for Compiler Testing

This module implements a comprehensive testing framework for GCC compiler optimization
by automatically inserting various GCC-specific attributes into C programs. It's designed
to discover compiler bugs, crashes, and optimization issues through systematic attribute
injection and differential testing.

Key Features:
- Automatic parsing of C programs to identify insertion points (variables, functions, structs)
- Random insertion of GCC-specific attributes (optimization, alignment, sanitizer, etc.)
- Differential testing between different optimization levels and compilers
- Crash detection and bug reporting with detailed logging
- Support for both CSmith-generated and YARPGen test programs
- Sanitizer integration for runtime error detection

Main Components:
1. RunTest class: Core testing logic for attribute insertion and compilation testing
2. Attribute mapping: Maps attribute names to their generation functions
3. Oracle compilation: Reference compilation for differential testing  
4. Bug detection: Identifies crashes, miscompilations, and runtime errors
5. Logging system: Comprehensive logging of bugs, errors, and test information

Usage:
    python insert_attribute_gcc.py --compiler gcc --test-type csmith --num-tests 1000

The tool generates timestamped output directories containing:
- crash/: Programs that caused compiler crashes
- bug/: Programs that revealed compiler bugs
- Detailed logs of all testing activities

This is part of the LLM4OPT project for automated compiler testing and optimization.
"""

import multiprocessing
import subprocess
import datetime
import tempfile
import logging
import random
import shutil
import signal
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

import sanitize
import optimization
from gcc_attributes import *

import pdb

# Create timestamped output directories for test results
current_time = datetime.datetime.now()
timestamp = current_time.strftime("%Y%m%d_%H%M%S")
CUR_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), f'Arise-GCC-{timestamp}')
CRASH_DIR = os.path.join(CUR_DIR, 'crash')  # Directory for programs that crash the compiler
BUG_DIR = os.path.join(CUR_DIR, 'bug')      # Directory for programs that reveal bugs

# Timeout configurations for different operations
COMPILATION_TIMEOUT = 60      # Timeout for compilation operations
RUN_TIMEOUT = 30             # Timeout for program execution
SAN_COMPILE_TIMEOUT = 90     # Timeout for sanitizer compilation (longer due to instrumentation)
CSMITH_HOME = '/home/compiler/csmith/runtime'  # CSmith runtime library path
MAX_NUM = 5000000            # Maximum number for random generation
MUTANT_NUM = 500             # Number of mutants to generate per test case

TEST_SUITE_DIR = '/home/compiler/gcc/gcc/testsuite'

GCC_CRASH_INFO = 'please submit a full bug report'
CLANG_CRASH_INFO = 'please submit a bug report to'

SAN_GCC = 'gcc'
SAN_CLANG = 'clang'

# Mapping of optimization levels to their enabled/disabled flag sets
# Used for differential testing between optimization levels
opt_set_map = {
    '-O0': utils.form_optimization_set(set(), optimization.option_O3),
    '-O1': utils.form_optimization_set(optimization.option_O1, optimization.option_O3 - optimization.option_O1),
    '-O2': utils.form_optimization_set(optimization.option_O2, optimization.option_O3 - optimization.option_O2),
    '-Os': utils.form_optimization_set(optimization.option_Os, optimization.option_O3 - optimization.option_Os),
    '-O3': utils.form_optimization_set(optimization.option_O3, set()),
}

opt_attributes = [f'optimize("{_}")' for _ in optimization.all_optimization]

opt_level_attributes = [
    'optimize (0)',
    'optimize (1)',
    'optimize (2)',
    'optimize (3)',
    'optimize ("Os")',
]

# Mapping of attribute names to their generation functions
# This enables dynamic attribute creation based on program analysis
attribute_function_map = {
    'aligned': form_aligned,
    'nonstring': insert_nonstring,
    'strict_flex_array': form_strict_flex_array,
    'vector_size': form_vector_size,
    'warn_if_not_aligned': form_warn_if_not_aligned,
    'mode': form_mode,
    'visibility': form_visibility,
    'access': form_access,
    'alloc_size': form_alloc_size,
    'alloc_align': form_alloc_align,
    'assume_aligned': form_assume_aligned,
    'nonnull': form_nonnull,
    'noreturn': form_noreturn,
    'null_terminated_string_arg': form_null_terminated_string_arg,
    'returns_nonnull': form_returns_nonnull,
    'sentinel': form_sentinel,
    'target': form_target,
    'target_clones': form_target_clones,
    'zero_call_used_regs': form_zero_call_used_regs,
    'tls_model': insert_tls_model
}

def generate_random_string(len):
    return ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(len))

def sanitize_check(prog, work_dir):
    out_file = f'{work_dir}/san.o'
    comp_cmd = f'{SAN_GCC} -w -O0 -fsanitize=undefined,address,leak {prog} -o {out_file}'
    comp_res = utils.run_cmd(comp_cmd, SAN_COMPILE_TIMEOUT)
    if comp_res[0] != 0:
        return -1

    gcc_run_res = utils.run_cmd(out_file, RUN_TIMEOUT)
    if ('runtime error:' in gcc_run_res[1]) or ('runtime error:' in gcc_run_res[2]):
        return -1
    comp_cmd = f'{SAN_CLANG} -w -O0 -fsanitize=undefined,address {prog} -o {out_file}'
    comp_res = utils.run_cmd(comp_cmd, SAN_COMPILE_TIMEOUT)
    if comp_res[0] != 0:
        return -1

    clang_run_res = utils.run_cmd(out_file, RUN_TIMEOUT)
    if ('runtime error:' in clang_run_res[1]) or ('runtime error:' in clang_run_res[2]):
        return -1
    
    if gcc_run_res[0] != clang_run_res[0]:
        return -1

def write_bug_desc_to_file(to_file, data):
    datas = data.split('\n')
    with open(to_file, "a") as f:
        for _ in datas:
            f.write(f"/* {_} */\n")

def get_logger(log_dir, name):
    if not os.path.exists(CUR_DIR):
        os.mkdir(CUR_DIR)
        
    logger = logging.getLogger(name)
    filename = f'{log_dir}/{name}.log'
    fh = logging.FileHandler(filename, mode='w+', encoding='utf-8')
    formatter = logging.Formatter('%(levelname)s:\n%(message)s')
    logger.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

bug_logger = get_logger(CUR_DIR, 'BUG')
err_logger = get_logger(CUR_DIR, 'ERR')
info_logger = get_logger(CUR_DIR, 'INFO')

know_bug = set()

class RunTest(object):
    def __init__(self, prog, pre, opt, org_opt, link_dir, work_dir):
        self.prog = prog
        self.pre = pre
        self.opt = opt
        self.org_opt = org_opt
        self.link_dir = link_dir
        self.work_dir = work_dir

        self.strut_var_list = []
        self.var_list = []
        self.func_list = []
        self.refine_list = {}

        self.case_list = {}
        self.invalid_cnt = 0
    
    def pre_run(self, comp):
        pre_comp = self.get_oracle(comp, f'{self.org_opt}', self.prog)
        if pre_comp[0] !=  0:
            err_logger.error(f'[Pre Compilation Error]: {self.prog}\n')
            return -1

        parse_options = f'-I{self.link_dir} {self.org_opt}' 
        ret_info = utils.parse_info(self.prog, parse_options, self.work_dir)
        if not ret_info:
            err_logger.error(f'[Parse Error]: {self.prog}\n')
            return -1
        [self.strut_var_list, self.var_list, self.func_list] = ret_info

    def form_insert_plan(self):
        insert_plan = {}
        for struct_var in self.strut_var_list:
            inserted = random.randint(0, 9)
            if inserted < 2:
                continue
            chose_attributes = random.choice(struct_related_attributes)
            attribute_code = ''
            if chose_attributes in attribute_function_map.keys():
                form_function = attribute_function_map[chose_attributes]
                # print(chose_attributes, form_function)
                attribute_code = form_function(struct_var)
                if not attribute_code:
                    continue
            else:
                attribute_code = f'__attribute__(({chose_attributes}))'
            if not struct_var['Line'].isdigit():
                continue
            if int(struct_var['Line']) not in insert_plan.keys():
                insert_plan[int(struct_var['Line'])] = [[int(struct_var['Column']), attribute_code]]
            else:
                insert_plan[int(struct_var['Line'])].append([int(struct_var['Column']), attribute_code])

        for var in self.var_list:
            if 'Local' in var['Scope']:
                continue
            if var['Line'] == '0' and var['Column'] == '0' and var['endColumn'] == '0':
                continue
            inserted = random.randint(0, 9)
            if inserted < 4:
                continue
            chose_attributes = random.choice(variable_related_attributes)
            attribute_code = ''
            if chose_attributes in attribute_function_map.keys():
                form_function = attribute_function_map[chose_attributes]
                # print(chose_attributes, form_function)
                attribute_code = form_function(var)
                if not attribute_code:
                    continue
            else:
                attribute_code = f'__attribute__(({chose_attributes}))'
            if not var['Line'].isdigit():
                continue
            if int(var['Line']) not in insert_plan.keys():
                insert_plan[int(var['Line'])] = [[int(var['Column']), int(var['endColumn']), attribute_code]]
            else:
                insert_plan[int(var['Line'])].append([int(var['Column']), int(var['endColumn']), attribute_code])

        attr_option_map, san_attr_list = sanitize.choose_gcc_sanitize()
        option_list = []
        for func in self.func_list:
            if func['Definition'].strip() == 'main':
                continue
            all_attributes = function_related_attributes + san_attr_list
            chose_opt_attributes = random.sample(opt_attributes, random.randint(3,8))
            chose_attributes = random.sample(all_attributes, random.randint(2,5))
            inline_attr = random.choice(inline_attributes)
            chose_attributes = chose_attributes + chose_opt_attributes
            chose_attributes.append(inline_attr)
            
            attribute_code = ''
            for chose_attribute in chose_attributes:
                if chose_attribute in attribute_function_map.keys():
                    form_function = attribute_function_map[chose_attribute]
                    # print(chose_attributes, form_function)
                    code = form_function(func)
                    if code:
                        attribute_code += f' {code}'
                else:
                    attribute_code += f' __attribute__(({chose_attribute}))'
                if chose_attribute in function_related_attributes_option.keys():
                    options = function_related_attributes_option[chose_attribute]
                    if not isinstance(options, str):
                        options = random.choice(options)
                    option_list.append(options)
                if chose_attribute in attr_option_map.keys():
                    option_list.append(attr_option_map[chose_attribute])
            if not func['Line'].isdigit():
                continue
            if int(func['Line']) not in insert_plan.keys():
                insert_plan[int(func['Line'])] = [[int(func['Column']), attribute_code]]
            else:
                insert_plan[int(func['Line'])].append([int(func['Column']), attribute_code])

        return insert_plan, option_list

    def insert_attribute(self):
        base_name = os.path.basename(self.prog)
        insert_nums = MUTANT_NUM
        for index in range(insert_nums):
            file_name, ext = os.path.splitext(base_name)
            new_base_name = f'{file_name}+insert{index}{ext}'
            new_test_case = f'{self.work_dir}/{new_base_name}'
            insert_plan, option_list = self.form_insert_plan()
            if not insert_plan:
                continue
            utils.generate(insert_plan, self.refine_list, self.prog, new_test_case)
            # utils.generate_new(insert_plan, self.prog, new_test_case)
            options = ' '.join(list(option_list))
            self.case_list[new_test_case] = options
    
    def insert_optimize_attribute(self):
        insert_plan = {}
        option_list = []
        func_num = len(self.func_list)
        main_idx = 0
        for func in self.func_list:
            if func['Definition'].strip() == 'main':
                func_num -= 1
                self.func_list.pop(main_idx)
                break
            main_idx += 1
        all_combo = itertools.product(opt_level_attributes, repeat=func_num)
        base_name = os.path.basename(self.prog)
        file_name, ext = os.path.splitext(base_name)
        if func_num > 6:
            all_combo = random.sample(list(all_combo), int(5 ** 6 / 2))
        for idx, combo in enumerate(all_combo):
            insert_plan = {}
            for func_idx, func in enumerate(self.func_list):
                attribute_code = f'__attribute__(({combo[func_idx]}))'
                if not func['Line'].isdigit():
                    continue
                if int(func['Line']) not in insert_plan.keys():
                    insert_plan[int(func['Line'])] = [[int(func['Column']), attribute_code]]
                else:
                    insert_plan[int(func['Line'])].append([int(func['Column']), attribute_code])
            if not insert_plan:
                continue
            new_base_name = f'{file_name}+insert{idx}{ext}'
            new_test_case = f'{self.work_dir}/{new_base_name}'
            utils.generate_new(insert_plan, self.prog, new_test_case)
            self.case_list[new_test_case] = ''

    def get_oracle(self, compiler, option, prog):
        global know_bug
        if 'gcc' in compiler:
            CRASH_INFO = GCC_CRASH_INFO
        if 'clang' in compiler:
            CRASH_INFO = CLANG_CRASH_INFO
        # compile
        if self.pre != '-o':
            compile_cmd = f'{compiler} -I{self.link_dir} {option} {prog} {self.pre}'
        else:
            out_file = f'{self.work_dir}/{os.path.basename(prog)}'
            base_name, ext = os.path.splitext(out_file)
            out_file = base_name + '.out'
            compile_cmd = f'{compiler} -I{self.link_dir} {option} {prog} -o {out_file}'
        compile_ret_code, compile_ret, compile_error = utils.run_cmd(compile_cmd, COMPILATION_TIMEOUT, self.work_dir)
        # compile_ret_code = -1
        # compile_ret = 0
        if compile_ret_code != 0:
            if CRASH_INFO in compile_error.lower() and not utils.duplicate(prog, compile_error.lower(), know_bug):
                if 'gcc' in compiler:
                    small_error = utils.filter_crash(compile_error, 'internal compiler error:')
                    bug_info = small_error.split('internal compiler error:')
                    know_bug.add(bug_info[-1].strip())
                # bug_logger.critical(f"[Compiler]: {compiler}\n[Prog]: {prog}\n[Reference]: {compile_cmd}\n[Error Code]: {compile_ret_code}\n[Error Message]: {compile_error}\n")
                write_bug_desc_to_file(prog, compile_cmd)
                write_bug_desc_to_file(prog, compile_error.lower())
                if not os.path.exists(f'{CRASH_DIR}/{os.path.basename(prog)}'):
                    shutil.copy(prog, CRASH_DIR)
                return (compile_ret_code, '', '')
            if (compile_ret_code == 139) or (compile_ret_code == 134):
                # bug_logger.critical(f"[Compiler]: {compiler}\n[Prog]: {prog}\n[Reference]: {compile_cmd}\n[Error Code]: {compile_ret_code}\n[Error Message]: {compile_error}\n")
                write_bug_desc_to_file(prog, compile_cmd)
                write_bug_desc_to_file(prog, compile_error.lower())
                if not os.path.exists(f'{CRASH_DIR}/{os.path.basename(prog)}'):
                    shutil.copy(prog, CRASH_DIR)
                return (compile_ret_code, '', '')
            if option == '':
                self.invalid_cnt += 1
            return (compile_ret_code, '', '')
        
        if self.pre != '-o':
            return (compile_ret_code, '', '')

        if not os.path.exists(out_file):
            return (compile_ret_code, '', '')

        run_file_cmd = f'{out_file}'
        run_ret_code, run_ret, run_error = utils.run_cmd(run_file_cmd, RUN_TIMEOUT, self.work_dir)
        if run_ret_code != 0:
            err_logger.error(f'[Prog]:{prog}\n[run_ret_code]:{run_ret_code}\n[run_ret]:{run_ret}\n[run_error]:{run_error}\n')
        if 'runtime error:' in run_error and not run_ret:
            run_ret = 'runtime error'
        if os.path.exists(out_file):
            os.remove(out_file)
        return (compile_ret_code, run_ret_code, run_ret)
    
    def get_res(self, compiler):
        # print(len(self.case_list))
        o_set = set()
        for prog, comp_o in self.case_list.items():
            oracle_list = set()
            for o in ['-O0', '-O1', '-O2', '-O3', '-Os']:
                if 'gcc' in compiler:
                    sub_options = sorted(random.sample(opt_set_map[o], random.randint(1,10)))
                    sub_options = ' '.join(sub_options)
                    opt = f'{o} {sub_options}'
                    while opt in o_set:
                        sub_options = sorted(random.sample(opt_set_map[o], random.randint(1,10)))
                        sub_options = ' '.join(sub_options)
                        opt = f'{o} {sub_options}'
                    o_set.add(opt)
                    oracle = self.get_oracle(compiler, opt, prog)
                else:
                    oracle = self.get_oracle(compiler, o, prog)
                if oracle[1] != '':
                    if int(oracle[1]) != 0:
                        continue
                oracle_list.add(oracle)
            if len(oracle_list) != 1:
                san_res = sanitize_check(prog, self.work_dir)
                if san_res == -1:
                    err_logger.error(f'[Sanitize Error]:\n[Prog]:{prog}\n')
                else:
                    bug_logger.critical(f"[Compiler]: {compiler}\n[Prog]: {prog}\n")
                    if not os.path.exists(f'{BUG_DIR}/{os.path.basename(prog)}'):
                        shutil.copy(prog, BUG_DIR)    
            if os.path.exists(prog):
                os.remove(prog)    

def run(test_case, compiler):
    if not os.path.exists(test_case):
        return
    pre, opt, org_opt = utils.parse_run_option(test_case)
    pre = '-o'
    print(test_case, flush=True)

    n = generate_random_string(8)
    work_dir = Path(__file__).parent / 'work' / n
    work_dir.mkdir(parents=True, exist_ok=True)
    base_name = os.path.basename(test_case)
    file_name, ext = os.path.splitext(base_name)
    src = str(work_dir/f'{file_name}.c')
    shutil.copy(test_case, src)
    
    run_test = RunTest(src, '-o', utils.filter_opt(opt), org_opt, os.path.dirname(test_case), work_dir)
    pre_res = run_test.pre_run(compiler)
    if pre_res == -1:
        if os.path.exists(str(work_dir)):
            shutil.rmtree(str(work_dir))
        return
    run_test.insert_attribute()
    run_test.get_res(compiler)

    info_logger.info(f'[Compilation Done]: [Prog]:{run_test.prog}\n[Invalid]:{run_test.invalid_cnt}\n')
    if os.path.exists(str(work_dir)):
        shutil.rmtree(str(work_dir))
        
def run_csmith(i, compiler):
    # pdb.set_trace()
    print(f'[Seed]: {i}', flush=True)
    n = generate_random_string(8)
    work_dir = Path(__file__).parent / 'work' / n
    work_dir.mkdir(parents=True, exist_ok=True)
    
    prog = utils.gen(work_dir, i)
    if (prog == -1) or (prog == -2):
        return
    run_test = RunTest(prog, '-o', '', '', CSMITH_HOME, work_dir)
    pre_res = run_test.pre_run(compiler)
    if pre_res == -1:
        if os.path.exists(str(work_dir)):
            shutil.rmtree(str(work_dir))
        return
    run_test.insert_attribute()
    run_test.get_res(compiler)

    info_logger.info(f'[Compilation Done]: [Prog]:{run_test.prog}\n[Invalid]:{run_test.invalid_cnt}\n')
    if os.path.exists(str(work_dir)):
        shutil.rmtree(str(work_dir))

def run_yarpgen(i, compiler):
    print(f'[Seed]: {i}', flush=True)
    n = generate_random_string(8)
    work_dir = Path(__file__).parent / 'work' / n
    work_dir.mkdir(parents=True, exist_ok=True)

    prog = utils.gen_yarpgen(work_dir, i)
    if (prog == -1) or (prog == -2):
        return
    run_test = RunTest(prog, '-o', '', '', work_dir, work_dir)
    pre_res = run_test.pre_run(compiler)
    if pre_res == -1:
        if os.path.exists(str(work_dir)):
            shutil.rmtree(str(work_dir))
        return
    run_test.insert_attribute()
    run_test.get_res(compiler)

    info_logger.info(f'[Compilation Done]: [Prog]:{run_test.prog}\n[Invalid]:{run_test.invalid_cnt}\n')
    if os.path.exists(str(work_dir)):
        shutil.rmtree(str(work_dir))

# export LD_LIBRARY_PATH=/home/software/gcc-trunk/lib64:$LD_LIBRARY_PATH
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arise pipeline.')
    parser.add_argument('--compiler', type=str, default='gcc', help='the testing compiler')
    parser.add_argument('--source', type=int, default=0, help='seed program source: 0.testsuite, 1.Csmith, 2.YARPGen')
    parser.add_argument('--multi', type=int, default=0, help='set to 0 if only want to run with single core')
    args = parser.parse_args()
    
    if not os.path.exists(BUG_DIR):
        os.mkdir(BUG_DIR)

    if not os.path.exists(CRASH_DIR):
        os.mkdir(CRASH_DIR)
    
    compiler = args.compiler
        
    if args.multi:
        proc_num = multiprocessing.cpu_count() - 2
        match args.source:
            case 0:
                c_test_cases = utils.find_c_files(TEST_SUITE_DIR, 'c')
                partial_run = partial(run, compiler=compiler)
                with multiprocessing.Pool(processes=proc_num) as pool:
                    pool.map(partial_run, c_test_cases)
            case 1:
                partial_run_csmith = partial(run_csmith, compiler=compiler)
                with multiprocessing.Pool(processes=proc_num) as pool:
                    pool.map(partial_run_csmith, list(range(MAX_NUM)))
            case 2:
                partial_run_yarpgen = partial(run_yarpgen, compiler=compiler)
                with multiprocessing.Pool(processes=proc_num) as pool:
                    pool.map(partial_run_yarpgen, list(range(MAX_NUM)))
    else:
        match args.source:
            case 0:
                c_test_cases = utils.find_c_files(TEST_SUITE_DIR, 'c')
                for test_case in c_test_cases:
                    run(test_case, compiler)
            case 1:
                for i in range(MAX_NUM):
                    run_csmith(i, compiler)
            case 2:
                for i in range(MAX_NUM):
                    run_yarpgen(i, compiler)