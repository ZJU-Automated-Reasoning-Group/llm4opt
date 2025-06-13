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

from clang_attributes import *

import pdb

current_time = datetime.datetime.now()
timestamp = current_time.strftime("%Y%m%d_%H%M%S")
CUR_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), f'Arise-CLANG-{timestamp}')
CRASH_DIR = os.path.join(CUR_DIR, 'crash')
BUG_DIR = os.path.join(CUR_DIR, 'bug')

COMPILATION_TIMEOUT = 60
RUN_TIMEOUT = 30
SAN_COMPILE_TIMEOUT = 90
CSMITH_HOME = '/home/compiler/csmith/runtime'
MAX_NUM = 5000000
MUTANT_NUM = 5

TEST_SUITE_DIR = '/home/compiler/gcc/gcc/testsuite'

GCC_CRASH_INFO = 'please submit a full bug report'
CLANG_CRASH_INFO = 'please submit a bug report to'

OPT = 'opt'
CLANG = 'clang'
SAN_GCC = 'gcc'
SAN_CLANG = 'clang'


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
    # 'cpu_dispatch': form_cpu_dispatch,
    # 'cpu_specific': form_cpu_specific,
    'warning': form_warning,
    'ifunc': form_ifunc,
    'malloc': form_malloc,
    'min_vector_width': form_min_vector_width,
    'patchable_function_entry': form_patchable_function_entry,
    # 'target': form_target,
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
    # 'xray_log_args': form_xray_log_args,
    'open': form_open,
}


bug_logger = utils.get_logger(CUR_DIR, 'BUG')
err_logger = utils.get_logger(CUR_DIR, 'ERR')
info_logger = utils.get_logger(CUR_DIR, 'INFO')

know_bug = []

def choose_llvm_optimization(exclude_opt=set()):
    remained_opt_pass = optimization.llvm_opt_pass - exclude_opt
    random_cnt = random.randint(1,5)
    choose_opt = random.sample(list(remained_opt_pass), random_cnt)
    return choose_opt

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

class RunTest(object):
    
    def __init__(self, prog, pre, opt, org_opt, link_dir, work_dir):
        self.prog = prog
        self.pre = pre
        self.opt = opt
        self.org_opt = org_opt
        self.link_dir = link_dir
        self.work_dir = work_dir

        self.struct_var_list = []
        self.var_list = []
        self.func_list = []
        self.loop_list = []
        self.struct_list = []
        self.refine_list = {}

        self.case_list = {}
        self.invalid_cnt = 0
    
    
    def pre_run(self, comp):
        pre_comp = self.get_oracle(comp, self.opt, self.prog)
        # pdb.set_trace()
        if pre_comp[0] !=  0:
            err_logger.error(f'[Compile Error]: {self.prog}\n')
            return -1

        parse_options = f'-I{self.link_dir} {self.opt}' 
        ret_info = utils.parse_info_clang(self.prog, parse_options, self.work_dir)
        if not ret_info:
            err_logger.error(f'[Parse Error]: {self.prog}\n')
            return -1
        [self.struct_var_list, self.var_list, self.func_list, self.loop_list, self.struct_list] = ret_info

    def form_insert_plan(self):
        insert_plan = {}
        for struct_var in self.struct_var_list:
            inserted = random.randint(0, 9)
            if inserted < 3:
                continue
            chose_attributes = random.choice(struct_related_attributes)
            attribute_code = chose_attributes
            if chose_attributes in attribute_function_map.keys():
                form_function = attribute_function_map[chose_attributes]
                attribute_code = form_function(struct_var)
                if not attribute_code:
                    continue
            if int(struct_var['Line']) not in insert_plan.keys():
                insert_plan[int(struct_var['Line'])] = [[int(struct_var['Column']), attribute_code]]
            else:
                insert_plan[int(struct_var['Line'])].append([int(struct_var['Column']), attribute_code])

        for struct in self.struct_list:
            inserted = random.randint(0, 9)
            if inserted < 2:
                continue
            chose_attributes = random.choice(struct_attributes)
            attribute_code = chose_attributes
            if chose_attributes in attribute_function_map.keys():
                form_function = attribute_function_map[chose_attributes]
                attribute_code = form_function(struct)
                if not attribute_code:
                    continue
            #[endline: [endcolumn, code]]
            if int(struct['endLine']) not in insert_plan.keys():
                insert_plan[int(struct['endLine'])] = [[int(struct['endColumn']), attribute_code]]
            else:
                insert_plan[int(struct['endLine'])].append([int(struct['endColumn']), attribute_code])

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
                attribute_code = form_function(var)
                if not attribute_code:
                    continue
            if not var['Line'].isdigit():
                continue
            if int(var['Line']) not in insert_plan.keys():
                insert_plan[int(var['Line'])] = [[int(var['Column']), int(var['endColumn']), attribute_code]]
            else:
                insert_plan[int(var['Line'])].append([int(var['Column']), int(var['endColumn']), attribute_code])

        attr_option_map, san_attr_list = sanitize.choose_llvm_sanitize()
        option_list = []
        opt_choose = False
        idx = 0
        for func in self.func_list:
            if func['Definition'].strip() == 'main':
                continue
            idx += 1
            all_attributes = function_related_attributes + san_attr_list
            chose_attributes = random.sample(all_attributes, random.randint(2,5))
            while '__attribute__((preserve_all))' in chose_attributes and '__attribute__((fastcall))' in chose_attributes:
                chose_attributes = random.sample(all_attributes, random.randint(2,5))
            opt_select = random.randint(0,1)
            if opt_select:
                opt_choose = True
                attribute_code = '__attribute__((optnone))'
            else:
                attribute_code = ''
            if (not opt_choose) and (idx == len(self.func_list)):
                attribute_code = '__attribute__((optnone))'
            for chose_attribute in chose_attributes:
                if chose_attribute in attribute_function_map.keys():
                    form_function = attribute_function_map[chose_attribute]
                    # print(chose_attributes, form_function)
                    code = form_function(func)
                    if code:
                        attribute_code += f' {code}'
                else:
                    attribute_code += f' {chose_attribute}'
                if chose_attribute in function_related_attributes_option.keys():
                    options = function_related_attributes_option[chose_attribute]
                    if not isinstance(options, str):
                        options = random.choice(options)
                    option_list.append(options)
                if chose_attribute in attr_option_map.keys():
                    option_list.append(attr_option_map[chose_attribute])
            if int(func['Line']) not in insert_plan.keys():
                insert_plan[int(func['Line'])] = [[int(func['Column']), attribute_code]]
            else:
                insert_plan[int(func['Line'])].append([int(func['Column']), attribute_code])

        for loop in self.loop_list:
            inserted = random.randint(0, 9)
            if inserted < 5:
                continue
            chose_attributes = random.choice(loop_related_attributes)
            attribute_code = chose_attributes
            if chose_attributes in attribute_function_map.keys():
                form_loop = attribute_function_map[chose_attributes]
                attribute_code = form_loop(loop)
                if not attribute_code:
                    continue
            #[line: [column, code]]
            insert_plan[int(loop['Line'])] = [['loop', int(loop['Column']), attribute_code]]

        return insert_plan, option_list


    def insert_attribute(self):
        # pdb.set_trace()
        base_name = os.path.basename(self.prog)
        insert_nums = MUTANT_NUM
        for index in range(insert_nums):
            file_name, ext = os.path.splitext(base_name)
            new_base_name = f'{file_name}+insert{index}{ext}'
            new_test_case = f'{self.work_dir}/{new_base_name}'
            insert_plan, option_list = self.form_insert_plan()
            if not insert_plan:
                continue
            utils.generate_clang_new(insert_plan, self.prog, new_test_case)
            options = ' '.join(list(option_list))
            # pdb.set_trace()
            self.case_list[new_test_case] = options
    

    def emit_ll(self, compiler, option, prog):
        ll_file = f'{self.work_dir}/{os.path.basename(prog)}'
        base_name, ext = os.path.splitext(ll_file)
        ll_file = base_name + '.ll'

        compile_cmd = f'{compiler} -S -emit-llvm {option} {prog} -o {ll_file}'
        compile_ret_code, compile_ret, compile_error = utils.run_cmd(compile_cmd, COMPILATION_TIMEOUT, self.work_dir)
        # pdb.set_trace()

        if compile_ret_code != 0:
            if CLANG_CRASH_INFO in compile_error.lower():
                bug_logger.critical(f"[Compiler]: {compiler}\n[Prog]: {prog}\n[Reference]: {compile_cmd}\n[Error Code]: {compile_ret_code}\n[Error Message]: {compile_error}\n")
                if not os.path.exists(f'{CRASH_DIR}/{os.path.basename(prog)}'):
                    shutil.copy(prog, CRASH_DIR)
                return (compile_ret_code, '', '')
            if (compile_ret_code == 139) or (compile_ret_code == 134):
                bug_logger.critical(f"[Compiler]: {compiler}\n[Prog]: {prog}\n[Reference]: {compile_cmd}\n[Error Code]: {compile_ret_code}\n[Error Message]: {compile_error}\n")
                if not os.path.exists(f'{CRASH_DIR}/{os.path.basename(prog)}'):
                    shutil.copy(prog, CRASH_DIR)
                return (compile_ret_code, '', '')
            if '-O0' in option:
                self.invalid_cnt += 1
        
        if not os.path.exists(ll_file):
            return (-1, '', '')

        return (compile_ret_code, compile_ret, compile_error)


    def run_opt(self, opt, option, o, prog):
        p = re.compile(r'.*unknown.*pass.*')
        ll_file = f'{self.work_dir}/{os.path.basename(prog)}'
        base_name, ext = os.path.splitext(ll_file)
        ll_file = base_name + '.ll'
        opt_ll_file = base_name + '.opt.ll'
        o = ','.join(o)

        opt_command = f'{opt} -passes={o} {ll_file} -o {opt_ll_file}'
        opt_ret_code, opt_ret, opt_error = utils.run_cmd(opt_command, COMPILATION_TIMEOUT, self.work_dir)

        if opt_ret_code != 0:
            if CLANG_CRASH_INFO in opt_error.lower():
                if 'instruction combining did not reach a fixpoint after 1 iterations' in opt_error.lower():
                    return (-1, '', '')
                bug_logger.critical(f"[Compiler]: {opt}\n[Prog]: {prog}\n[Option]: {option}\n[Reference]: {opt_command}\n[Error Code]: {opt_ret_code}\n[Error Message]: {opt_error}\n")
                if not os.path.exists(f'{CRASH_DIR}/{os.path.basename(prog)}'):
                    shutil.copy(prog, CRASH_DIR)
                if not os.path.exists(f'{CRASH_DIR}/{os.path.basename(ll_file)}'):
                    shutil.copy(ll_file, CRASH_DIR)
                return (opt_ret_code, '', '')
            if (opt_ret_code == 139) or (opt_ret_code == 134):
                bug_logger.critical(f"[Compiler]: {opt}\n[Prog]: {prog}\n[Option]: {option}\n[Reference]: {opt_command}\n[Error Code]: {opt_ret_code}\n[Error Message]: {opt_error}\n")
                if not os.path.exists(f'{CRASH_DIR}/{os.path.basename(prog)}'):
                    shutil.copy(prog, CRASH_DIR)
                if not os.path.exists(f'{CRASH_DIR}/{os.path.basename(ll_file)}'):
                    shutil.copy(ll_file, CRASH_DIR)
                return (opt_ret_code, '', '')
            if p.match(opt_error.lower()):
                unknown_pass = opt_error.lower().split("'")[-2]
                return unknown_pass
        
        if not os.path.exists(opt_ll_file):
            return (-1, '', '')

        return (opt_ret_code, opt_ret, opt_error)


    def get_oracle(self, compiler, option, prog):
        # compile
        if self.pre != '-o':
            compile_cmd = f'{compiler} -I{self.link_dir} {option} {prog} {self.pre}'
        else:
            out_file = f'{self.work_dir}/{os.path.basename(prog)}'
            base_name, ext = os.path.splitext(out_file)
            out_file = base_name + '.out'
            compile_cmd = f'{compiler} -I{self.link_dir} {option} {prog} -o {out_file}'
        compile_ret_code, compile_ret, compile_error = utils.run_cmd(compile_cmd, COMPILATION_TIMEOUT, self.work_dir)
        # pdb.set_trace()
        if compile_ret_code != 0:
            if CLANG_CRASH_INFO in compile_error.lower() and not utils.duplicate(prog, compile_error.lower(), know_bug):
                bug_logger.critical(f"[Compiler]: {compiler}\n[Prog]: {prog}\n[Reference]: {compile_cmd}\n[Error Code]: {compile_ret_code}\n[Error Message]: {compile_error}\n")
                if not os.path.exists(f'{CRASH_DIR}/{os.path.basename(prog)}'):
                    shutil.copy(prog, CRASH_DIR)
                return (compile_ret_code, '', '')
            if (compile_ret_code == 139) or (compile_ret_code == 134):
                bug_logger.critical(f"[Compiler]: {compiler}\n[Prog]: {prog}\n[Reference]: {compile_cmd}\n[Error Code]: {compile_ret_code}\n[Error Message]: {compile_error}\n")
                if not os.path.exists(f'{CRASH_DIR}/{os.path.basename(prog)}'):
                    shutil.copy(prog, CRASH_DIR)
                return (compile_ret_code, '', '')
            return (compile_ret_code, '', '')
        
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
        return (compile_ret_code, run_ret_code, run_ret)
    
    
    def get_res_opt(self, compiler):
        for prog, comp_o in self.case_list.items():
            # pdb.set_trace()
            for o in ['-O0', '-O1', '-O2', '-O3', '-Os']:
                option = f'{self.org_opt} {comp_o} {o}'
                ll_res = self.emit_ll(CLANG, option, prog)
                # pdb.set_trace()
                if ll_res[0] != 0:
                    continue
                
                choose_opt_passes = choose_llvm_optimization()
                opt_res = self.run_opt(OPT, option, choose_opt_passes, prog)
                unknown_opt_set = set()
                while True:
                    if not isinstance(opt_res, str):
                        break
                    unknown_opt_set.add(opt_res)
                    choose_opt_passes = choose_llvm_optimization(unknown_opt_set)
                    opt_res = self.run_opt(OPT, option, choose_opt_passes, prog)
                
                
    def get_res(self, compiler):
        for prog, comp_o in self.case_list.items():
            oracle_list = set()
            for o in ['-O0', '-O1', '-O2', '-O3', '-Os']:
                comp_option = f'{o} {self.opt} {comp_o}'
                oracle = self.get_oracle(compiler, comp_option, prog)
                if oracle[1] != '':
                    if int(oracle[1]) < 0:
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
    print(test_case, flush=True)
    
    n = generate_random_string(8)
    work_dir = Path(__file__).parent / 'work' / n
    work_dir.mkdir(parents=True, exist_ok=True)
    base_name = os.path.basename(test_case)
    file_name, ext = os.path.splitext(base_name)
    src = str(work_dir/f'{file_name}.c')
    shutil.copy(test_case, src)

    run_test = RunTest(src, '-o', '', '', os.path.dirname(test_case), work_dir)
    pre_res = run_test.pre_run(compiler)
    if pre_res == -1:
        if os.path.exists(str(work_dir)):
            shutil.rmtree(str(work_dir))
        return
    try:
        run_test.insert_attribute()
    except Exception as e:
        print(f'[Insert Error]:[Prog]:{test_case}\n[Exception]:{e}')
    run_test.get_res_opt(compiler)
    
    info_logger.info(f'[Compilation Done]: [Prog]:{run_test.prog}\n[Invalid]:{run_test.invalid_cnt}\n')
    if os.path.exists(str(work_dir)):
        shutil.rmtree(str(work_dir))

def run_csmith(i, compiler):
    print(f'[Seed]: {i}', flush=True)
    n = generate_random_string(8)
    work_dir = Path(__file__).parent / 'work' / n
    work_dir.mkdir(parents=True, exist_ok=True)
    
    prog = utils.gen(work_dir, i)
    if (prog == -1) or (prog == -2):
        return
    run_test = RunTest(prog, '-c', '', '', CSMITH_HOME, work_dir)
    pre_res = run_test.pre_run(compiler)
    if pre_res == -1:
        if os.path.exists(str(work_dir)):
            shutil.rmtree(str(work_dir))
        return
    run_test.insert_attribute()
    run_test.get_res_opt(compiler)

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
    run_test.get_res_opt(compiler)

    info_logger.info(f'[Compilation Done]: [Prog]:{run_test.prog}\n[Invalid]:{run_test.invalid_cnt}\n')
    if os.path.exists(str(work_dir)):
        shutil.rmtree(str(work_dir))


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
    