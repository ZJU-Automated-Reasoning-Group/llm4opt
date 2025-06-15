import random
import string
import re

struct_attributes = [
    '__single_inheritance',
    '__multiple_inheritance',
    '__virtual_inheritance',
    '__unspecified_inheritance',
    '__attribute__((deprecated))',
    '__attribute__((lto_visibility_public))',
    '__attribute__((randomize_layout))',
    '__attribute__((no_randomize_layout))',
    '__attribute__((enforce_read_only_placement))',
]

enum_attributes = [
    'enum_extensibility',
    '__attribute__((flag_enum))',

]

union_attributes = [
    '__attribute__((transparent_union))',
    # '__declspec(empty_bases)',
]

struct_related_attributes = [
    'btf_decl_tag',
    'aligned',
    '__attribute__((packed))',
    'vector_size'
]

variable_related_attributes = [
    # 'asm',
    '__attribute__((deprecated))',
    '__attribute__((weak))',
    'omp_target',
    # '_Nonnull',
    # '_Null_unspecified',
    # '_Nullable',
    # '_Nullable_result',
    # '__attribute__((opencl_global_device))',
    # '__attribute__((opencl_global_host))',
    # '__attribute__((opencl_generic))',
    # '__attribute__((opencl_global))',
    # '__attribute__((opencl_local))',
    # '__attribute__((opencl_private))',
    'open'
    'align_value',
    'noderef',
    'aligned',
    '__attribute__((common))',
    # 'mode',
    '__attribute__((nocommon))',
    'visibility',
    'release_handle',
]

function_related_attributes = [
    '__attribute__((fastcall))',
    '__attribute__((preserve_all))',
    '__attribute__((preserve_most))',
    '__attribute__((preserve_none))',
    # '__attribute__((regcall))',
    '__attribute__((riscv_vector_cc))',
    '__attribute__((vectorcall))',
    'callable_when',
    '__attribute__((param_typestate))',
    # 'asm',
    '__attribute__((deprecated))',
    '__attribute__((weak))',
    # 'func_simd',
    # 'omp_target',
    # '#pragma omp declare variant',
    '_Noreturn',
    'abi_tag',
    '__attribute__((acquire_capability))',
    '__attribute__((acquire_shared_capability))',
    '__attribute__((exclusive_lock_function))',
    '__attribute__((shared_lock_function))',
    'alloc_size',
    'alloc_align',
    # '__declspec(allocator)',
    '__attribute__((always_inline))',
    '__attribute__((artificial))',
    '__attribute__((assert_capability))',
    '__attribute__((assert_shared_capability))',
    'assume_aligned',
    # 'callback',
    'btf_decl_tag',
    '__attribute__((cold))',
    '__attribute__((constructor))',
    '__attribute__((convergent))',
    # 'cpu_dispatch',
    # 'cpu_specific',
    '__attribute__((destructor))',
    '__attribute__((disable_tail_calls))',
    'enforce_tcb',
    'enforce_tcb_leaf',
    'warning',
    '__attribute__((flatten))',
    '__attribute__((force_align_arg_pointer))',
    '__attribute__((gnu_inline))',
    '__attribute__((hot))',
    # 'ifunc',
    'malloc',
    'min_vector_width',
    '__attribute__((minsize))',
    '__attribute__((no_builtin))',
    '__attribute__((no_caller_saved_registers))',
    '__attribute__((no_speculative_load_hardening))',
    '__attribute__((speculative_load_hardening))',
    # '__declspec(noalias)',
    # '__attribute__((noconvergent))',
    '__attribute__((warn_unused_result))',
    '__attribute__((noduplicate))',
    '__attribute__((noinline))',
    'noreturn',
    '__attribute__((nothrow))',
    '__attribute__((nouwtable))',
    # '__attribute__((optnone))',
    'patchable_function_entry',
    '__attribute__((release_capability))',
    '__attribute__((release_shared_capability))',
    '__attribute__((release_generic_capability))',
    '__attribute__((unlock_function))',
    '__attribute__((retain))',
    # 'target',
    # 'target_clones',
    'try_acquire_capability',
    'try_acquire_shared_capability',
    '__attribute__((unsafe_buffer_usage))',
    '__attribute__((used))',
    '__attribute__((xray_always_instrument))',
    '__attribute__((xray_never_instrument))',
    # '__attribute__((xray_log_args(1)))',
    'xray_log_args',
    'zero_call_used_regs',
    'acquire_handle',
    'use_handle',
    'nonnull',
    'returns_nonnull',
    # '__attribute__((opencl_global_device))',
    # '__attribute__((opencl_global_host))',
    # '__attribute__((opencl_constant))',
    # '__attribute__((opencl_generic))',
    # '__attribute__((opencl_global))',
    # '__attribute__((opencl_local))',
    # '__attribute__((opencl_private))',
    'open',
    # '__attribute__((allocating))',
    # '__attribute__((blocking))',
    # '__attribute__((nonallocating))',
    # '__attribute__((nonblocking))',
    '__attribute__((nomerge))',
    'aligned',
    '__attribute__((const))',
    '__attribute__((pure))',
    '__attribute__((returns_twice))',
    'sentinel',
    'vector_size_func',
    'visibility',
    # '__attribute__((weakref))',
    '__attribute__((called_once))',
    '__attribute__((unused))',
    # '__declspec(empty_bases)',
    # '__attribute__(())',
    '__attribute__((no_split_stack))',
    '__attribute__((no_stack_protector))',
    '__attribute__((nocf_check))',
    '__attribute__((no_instrument_function))',
    '__attribute__((nodebug))'
]

loop_related_attributes = [
    '#pragma omp simd',
    'clang_loop',
    '#pragma unroll',
    '#pragma nounroll',
    '#pragma unroll_and_jam',
    '#pragma unroll_and_jam',
    '#pragma unroll_and_jam',
    # '__attribute__((fallthrough))',
    '__attribute__((opencl_unroll_hint))',
    'code_align'
]

function_related_attributes_option = {
    '__attribute__((no_split_stack))': '-fsplit-stack',
    '__attribute__((no_stack_protector))': '-fstack-protector',
    '__attribute__((nocf_check))': '-fcf-protection',
    '__attribute__((no_instrument_function))': '-finstrument-functions',
    '__attribute__((nodebug))': '-g'
}

def form_callable_when(info):
    consumed_type_list = ['unconsumed', 'consumed', 'unknown']
    consumed_type = random.choice(consumed_type_list)
    return f'__attribute__ ((callable_when("{consumed_type}")))'

def form_asm(info):
    all_characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(all_characters) for i in range(random.randint(3,10)))
    return f'asm("{random_string}")'

def form_enum_extensibility(info):
    type_list = ['open', 'closed']
    t = random.choice(type_list)
    return f'__attribute__ ((callable_when("{t}")))'

def form_declare_smid(info):
    para_list = info['Parameter list']
    clauses_type_lists = ['simdlen', 'linear', 'aligned', 'uniform', 'inbranch', 'notinbranch', '']
    clauses_type = random.choice(clauses_type_lists)
    if clauses_type == 'inbranch':
        return f'#pragma omp declare simd {clauses_type}'
    if clauses_type == 'notinbranch':
        return f'#pragma omp declare simd {clauses_type}'
    if clauses_type == 'simdlen':
        simdlen = random.randint(1,8)
        return f'#pragma omp declare simd simdlen({simdlen})'
    if clauses_type == 'linear':
        return f'#pragma omp declare simd linear()'
    if clauses_type == 'uniform':
        return f'#pragma omp declare simd uniform()'
    if clauses_type == 'aligned':
        return f'#pragma omp declare simd aligned()'
    if clauses_type == '':
        return f'#pragma omp declare simd'

def form_omp_target(info):
    pass  

def insert_noreturn(info):
    func_ret_type = info['Type']
    if func_ret_type == 'void':
        return '_Noreturn'

def form_abi_tag(info):
    # all_characters = string.ascii_letters + string.digits + string.punctuation
    all_characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(all_characters) for i in range(random.randint(3,10)))
    return f'__attribute__((abi_tag("{random_string}")))'

def form_callback(info):
    pass

def form_btf_decl_tag(info):
    all_characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(all_characters) for i in range(random.randint(3,10)))
    return f'__attribute__((btf_decl_tag("{random_string}")))'

def form_cpu_dispatch(info):
    func_name = info['Definition']
    if func_name == 'main':
        return
    cpu_type_list = ['ivybridge', 'atom', 'sandybridge']
    cpu_type = random.choice(cpu_type_list)
    return f'__attribute__ ((cpu_dispatch({cpu_type})))'

def form_cpu_specific(info):
    func_name = info['Definition']
    if func_name == 'main':
        return
    cpu_type_list = ['ivybridge', 'atom', 'sandybridge']
    cpu_type = random.choice(cpu_type_list)
    return f'__attribute__((cpu_specific({cpu_type})))'

def form_warning(info):
    all_characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(all_characters) for i in range(random.randint(3,10)))
    return f'__attribute__((warning("{random_string}")))'

def form_ifunc(info):
    pass

def form_malloc(info):
    func_ret_type = info['Type']
    if "*" in func_ret_type:
        return '__attribute__((malloc))'

def form_min_vector_width(info):
    width_list = [16, 32, 64, 128, 256, 521]
    width = random.choice(width_list)
    return f'__attribute__((min_vector_width({width})))'

def form_noreturn(info):
    func_ret_type = info['Type']
    if func_ret_type == 'void':
        return '__attribute__((noreturn))'

def form_patchable_function_entry(info):
    n_list = [4, 8, 16, 32, 64, 128]
    n = random.choice(n_list)
    m = random.randint(0,4)
    return f'__attribute__((patchable_function_entry({n}, {m})))'

def form_try_acquire_capability(info):
    boolean = random.randint(0,1)
    return f'__attribute__((try_acquire_capability({boolean})))'

def form_try_acquire_shared_capability(info):
    boolean_list = ['1', '0']
    boolean = random.choice(boolean_list)
    return f'__attribute__((try_acquire_shared_capability({boolean})))'

def form_acquire_handle(info):
    all_characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(all_characters) for i in range(random.randint(3,10)))
    return f'__attribute__((acquire_handle("{random_string}")))'

def form_release_handle(info):
    all_characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(all_characters) for i in range(random.randint(3,10)))
    return f'__attribute__((release_handle("{random_string}")))'

def form_use_handle(info):
    all_characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(all_characters) for i in range(random.randint(3,10)))
    return f'__attribute__((use_handle("{random_string}")))'

def insert_noderef(info):
    var_type = info['Type']
    if '*' in var_type:
        return '__attribute__((noderef))'

def form_align_value(info):
    var_type = info['Type']
    if '*' in var_type:
        num_candidate = ['4', '8', '16', '32', '64', '128']
        num = random.choice(num_candidate)
        return f'__attribute__((align_value({num})))'

def form_code_align(info):
    num_candidate = ['4', '8', '16', '32', '64', '128']
    num = random.choice(num_candidate)
    return f'__attribute__((code_align({num})))'

def form_aligned(info):
    num_candidate = ['4', '8', '16', '32', '64']
    num = random.choice(num_candidate)
    return f'__attribute__((aligned({num})))'

def form_alloc_size(info):
    func_ret_type, para_list = info['Type'], info['Parameter list']
    if 'size_t' not in para_list:
        return
    para_list = para_list.strip('()').split(',')
    if '*' not in func_ret_type:
        return
    if len(para_list) == 0:
        return
    size_t_pos = [index + 1 for index, item in enumerate(para_list) if item == 'size_t']
    if not size_t_pos:
        return
    if len(size_t_pos) == 1:
        return f'__attribute__((alloc_size({size_t_pos[0]}))'
    two_pos = random.sample(size_t_pos, 2)
    return f'__attribute__((alloc_size({two_pos[0]}, {two_pos[1]}))'

def form_xray_log_args(info):
    para_list = info['Parameter list']
    para_list = para_list.strip('()').split(',')
    pos = random.randint(0, len(para_list)-1)
    return f'__attribute__((form_xray_log_args({pos})))'

def insert_nonstring(info):
    var_type = info['Type']
    var_type = var_type.replace(' ', '')
    if 'char[' in var_type:
        return '__attribute__((nonstring))'

def form_mode(info):
    var_type = info['Type']
    if '[' in var_type:
        return
    if '*' in var_type:
        mode_type = ['byte', '__byte__', 'word', '__word__', 'pointer', '__pointer__']
        mode = random.choice(mode_type)
        return f'__attribute__((mode({mode})))'
    else:
        mode_type = ['byte', '__byte__', 'word', '__word__']
        mode = random.choice(mode_type)
        return f'__attribute__((mode({mode})))'
        
def form_strict_flex_array(info):
    var_type = info['Type']
    if not re.search(r"\[.*?\]", var_type):
        return
    level_list = [0,1,2,3]
    level = random.choice(level_list)
    return f'__attribute__((strict_flex_array({level})))'

def insert_tls_model(info):
    var_type = info['Type']
    if var_type == '__thread':
        return '__attribute__((tls_model("tls_model")))'

def form_vector_size(info):
    num_candidate = ['4', '8', '16', '32', '64']
    num = random.choice(num_candidate)
    return f'__attribute__((vector_size({num})))'

def form_vector_size_func(info):
    func_name = info['Definition']
    if func_name == 'main':
        return
    func_ret_type = info['Type'].strip()
    if func_ret_type != 'void':
        num_candidate = ['4', '8', '16', '32', '64']
        num = random.choice(num_candidate)
        return f'__attribute__((vector_size({num})))'

def form_warn_if_not_aligned(info):
    num_candidate = ['4', '8', '16', '32', '64', '__BIGGEST_ALIGNMENT__']
    num = random.choice(num_candidate)
    return f'__attribute__((warn_if_not_aligned({num})))'

def form_visibility(info):
    visibility_type_list = ['default', 'hidden', 'internal', 'protected']
    visibility_type = random.choice(visibility_type_list)
    return f'__attribute__ ((visibility("{visibility_type}")))'

def form_access(info):
    para_list = info['Parameter list']
    if '*' not in para_list:
        return
    mode_list = ['read_only', 'write_only', 'read_write']
    para_list = para_list.strip('()').split(',')
    size_t_pos = [index + 1 for index, item in enumerate(para_list) if item == 'size_t']
    pointer_pos = [index + 1 for index, item in enumerate(para_list) if '*' in item]
    if not pointer_pos:
        return
    if len(size_t_pos) == 0:
        if len(pointer_pos) == 1:
            mode = random.choice(mode_list)
            return f'__attribute__((access({mode}, {pointer_pos[0]})))'
        else:
            para_pos = random.choice(pointer_pos)
            mode = random.choice(mode_list)
            return f'__attribute__((access({mode}, {para_pos})))'
    else:
        size_t = random.choice(size_t_pos)
        if len(pointer_pos) == 1:
            mode = random.choice(mode_list)
            return f'__attribute__((access({mode}, {pointer_pos[0]}, {size_t})))'
        else:
            para_pos = random.choice(pointer_pos)
            mode = random.choice(mode_list)
            return f'__attribute__((access({mode}, {para_pos}, {size_t})))'

def form_alloc_align(info):
    func_ret_type, para_list = info['Type'], info['Parameter list']
    if 'size_t' not in para_list:
        return
    para_list = para_list.strip('()').split(',')
    if '*' not in func_ret_type:
        return
    if len(para_list) == 0:
        return
    size_t_pos = [index + 1 for index, item in enumerate(para_list) if item == 'size_t']
    if not size_t_pos:
        return
    size_t = random.choice(size_t_pos)
    return f'__attribute__((alloc_align({size_t}))'

def form_assume_aligned(info):
    func_ret_type = info['Type']
    if '*' not in func_ret_type:
        return
    num_candidate = ['4', '8', '16', '32', '64']
    num = random.choice(num_candidate)
    if num == '1':
        return f'__attribute__((assume_aligned({num})))'
    else:
        offset_use = random.randint(0, 1)
        if offset_use:
            return f'__attribute__((assume_aligned({num}, {random.randint(1,int(num)-1)})))'
        else:
            return f'__attribute__((assume_aligned({num})))'

def form_nonnull(info):
    para_list = info['Parameter list']
    if '*' not in para_list:
        return
    para_list = para_list.strip('()').split(',')
    pointer_pos = [index + 1 for index, item in enumerate(para_list) if '*' in item]
    if not pointer_pos:
        return
    if len(pointer_pos) == 1:
        return f'__attribute__((nonnull({pointer_pos[0]})))'
    else:
        chose_pointer_list = random.sample(pointer_pos, random.randint(2, len(pointer_pos)))
        ppos = ','.join(map(str, chose_pointer_list))
        return f'__attribute__((nonnull({ppos})))'

def form_null_terminated_string_arg(info):
    para_list = info['Parameter list']
    para_list = para_list.strip('()').split(',')
    para_list = [_.replace(' ', '') for _ in para_list]
    char_para_pos = [index + 1 for index, item in enumerate(para_list) if 'char*' in item]
    if len(char_para_pos) == 0:
        return
    else:
        pos = random.choice(char_para_pos)
        return f'__attribute__((null_terminated_string_arg({pos})))'

def form_returns_nonnull(info):
    func_ret_type = info['Type']
    if '*' in func_ret_type:
        return '__attribute__((returns_nonnull))'

def form_sentinel(info):
    para_list = info['Parameter list']
    if '...' in para_list:
        return '__attribute__((sentinel))'

def form_target(info):
    isa_list = [
        'sse', 'sse2', 'sse3', 'sse4.1', 'sse4.2', 'sse4a',
        'avx', 'avx2', 'avx512f', 'avx512cd', 'avx512er', 'avx512pf', 'avx512vl', 'avx512bw', 'avx512dq',
        'mmx', 'popcnt', 'bmi', 'bmi2', 'fma', 'xop'
    ]
    isa = random.choice(isa_list)
    return f'__attribute__((target("{isa}")))'

def form_target_clones(info):
    isa_list = [
        'sse', 'sse2', 'sse3', 'sse4.1', 'sse4.2', 'sse4a',
        'avx', 'avx2', 'avx512f', 'avx512cd', 'avx512er', 'avx512pf', 'avx512vl', 'avx512bw', 'avx512dq',
        'mmx', 'popcnt', 'bmi', 'bmi2', 'fma', 'xop'
    ]
    isa = random.choice(isa_list)
    isa_list.remove(isa)
    isas = random.sample(isa_list, 4)
    isas = [f'"{_}"' for _ in isas]
    form_isa = ','.join(map(str, isas))
    return f'__attribute__((target("{isa}"), target_clones({form_isa})))'

def form_zero_call_used_regs(info):
    choice_list = ['skip', 'used', 'used-gpr', 'used-arg', 'used-gpr-arg', 'all', 'all-gpr', 
                   'all-arg', 'all-gpr-arg', 'leafy', 'leafy-gpr', 'leafy-arg', 'leafy-gpr-arg']
    choice = random.choice(choice_list)
    return f'__attribute__((zero_call_used_regs("{choice}")))'

def form_clang_loop(info):
    loop_option_list = ['unroll(enable)', 'unroll(disable)', 'vectorize(enable)', 'vectorize(disable)', 'interleave(enable)', 'interleave(disable)', 'distribute(enable)', 'distribute(disable)']
    loop_option = random.choice(loop_option_list)
    return f'#pragma clang loop {loop_option}'

def form_enforce_tcb(info):
    all_characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(all_characters) for i in range(random.randint(3,10)))
    return f'__attribute__((enforce_tcb("{random_string}")))'

def form_enforce_tcb_leaf(info):
    all_characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(all_characters) for i in range(random.randint(3,10)))
    return f'__attribute__((enforce_tcb_leaf("{random_string}")))'

def form_open(info):
    type_list = ['opencl_global_device', 'opencl_global_host', 'opencl_generic', 'opencl_global', 'opencl_local', 'opencl_private']
    chose_type = random.choice(type_list)
    return f'__attribute__(({chose_type}))'