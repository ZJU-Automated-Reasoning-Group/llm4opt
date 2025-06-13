"""
GCC-Specific Attribute Definitions and Generation Functions

This module defines GCC-specific attributes and provides functions to generate
properly formatted attribute strings for insertion into C programs. It supports
a comprehensive set of GCC attributes across different categories including
variable attributes, function attributes, and struct attributes.

Key Components:

1. Attribute Categories:
   - struct_related_attributes: Attributes applicable to struct members and variables
   - variable_related_attributes: Attributes for global and local variables  
   - function_related_attributes: Attributes for function declarations
   - function_optimize_attributes: Optimization-level attributes for functions
   - inline_attributes: Function inlining control attributes

2. Attribute Generation Functions:
   - Form properly formatted __attribute__((...)) strings
   - Handle attribute parameters and syntax variations
   - Validate attribute applicability based on context (type, scope, etc.)
   - Generate random but valid attribute parameters

3. Context-Aware Attribute Generation:
   - form_aligned(): Memory alignment attributes with valid alignment values
   - form_alloc_size(): Function allocation size hints based on parameters
   - form_vector_size(): Vector type size attributes
   - form_mode(): Variable mode attributes for different data types
   - form_access(): Function parameter access patterns
   - And many more specialized attribute generators

4. Attribute-Option Relationships:
   - Maps attributes to their corresponding compiler options
   - Ensures attributes are only used when relevant compiler flags are active
   - Supports conditional attribute insertion based on compilation context

Key Features:
- Type-aware attribute generation (only applies attributes to compatible types)
- Parameter validation (ensures attribute parameters are syntactically correct)
- Random parameter generation within valid ranges
- Support for both simple and complex attribute syntax
- Integration with GCC's extensive attribute system

Supported Attribute Types:
- Memory alignment and layout control
- Function optimization and inlining hints
- Parameter and return value annotations
- Visibility and linkage control
- Target-specific optimizations
- Sanitizer control attributes
- Stack and security attributes

Usage:
    This module is used by insert_attribute_gcc.py to:
    1. Select appropriate attributes for different program elements
    2. Generate syntactically correct attribute strings
    3. Ensure type compatibility and parameter validity
    4. Support systematic testing of GCC's attribute system

The attribute_function_map dictionary maps attribute names to their
corresponding generation functions, enabling dynamic attribute creation
based on program analysis results.

Part of the LLM4OPT project for automated compiler testing and optimization.
"""

import random
import string
import re

# Attributes that can be applied to struct members and variables
# These control memory layout, alignment, and compiler warnings
struct_related_attributes = [
    'aligned',              # Control memory alignment
    'nonstring',           # Indicate non-null-terminated character arrays
    'packed',              # Pack struct members tightly
    'strict_flex_array',   # Control flexible array member behavior
    # 'unavailable',       # Mark as unavailable (commented out for testing)
    'unused',              # Suppress unused variable warnings
    'vector_size',         # Specify vector type size
    'warn_if_not_aligned'  # Warn if not properly aligned
]

variable_related_attributes = [
    'aligned',
    'common',
    'nocommon',
    'deprecated',
    'mode',
    'no_icf',
    'noinit',
    'persistent', #init
    # 'unavailable',
    'unused',
    'used',
    'retain',
    'vector_size',
    # 'visibility',
    # 'weak'
]

function_optimize_attributes = [
    'optimize("-O0")',
    'optimize("-O1")',
    'optimize("-O2")',
    'optimize("-O3")',
    'optimize("-Og")',
    'optimize("-Oz")',
    'optimize("-Os")',
    'optimize("-Ofast")',
    # 'always_inline',
    # 'noinline',
    # 'artificial',
    # 'cold',
    # 'hot',
    # 'flatten',
]

inline_attributes = [
    'always_inline',
    'noinline',
]

# Attributes that can be applied to function declarations
# These control function optimization, behavior, and compiler analysis
function_related_attributes = [
    'access',                        # Specify parameter access patterns
    'aligned',                       # Function alignment requirements
    'alloc_align',                   # Allocation alignment hints
    # 'always_inline',               # Force inlining (handled separately)
    'artificial',                    # Mark as compiler-generated
    'assume_aligned',                # Assume parameter alignment
    'alloc_align',                   # Duplicate entry (should be cleaned up)
    'alloc_size',                    # Allocation size hints
    'cold',                          # Rarely executed function
    'const',                         # No side effects, no global memory access
    'constructor',                   # Run before main()
    'destructor',                    # Run after main()
    'deprecated',                    # Mark as deprecated
    'externally_visible',            # Force external visibility
    'flatten',                       # Inline all called functions
    'gnu_inline',                    # Use GNU inline semantics
    'hot',                           # Frequently executed function
    'leaf',                          # Function doesn't call other functions
    'malloc',                        # Returns newly allocated memory
    'no_icf',                        # Disable identical code folding
    'no_profile_instrument_function', # Disable profiling instrumentation
    'no_reorder',                    # Don't reorder with other functions
    'no_stack_protector',            # Disable stack protection
    'noclone',                       # Disable function cloning
    # 'noinline',                    # Prevent inlining (handled separately)
    'noipa',                         # No interprocedural analysis
    'nonnull',                       # Parameters must not be NULL
    'noplt',                         # Don't use PLT for calls
    'noreturn',                      # Function never returns
    'nothrow',                       # Function doesn't throw exceptions
    'patchable_function_entry',      # Patchable function entry point
    'pure',                          # No side effects, may read global memory
    'retain',                        # Don't eliminate unused function
    'returns_nonnull',               # Return value is never NULL
    'returns_twice',                 # Function may return multiple times
    'sentinel',                      # Variadic function sentinel
    'smid',                          # SIMD function variant
    'target',                        # Target-specific optimization
    'target_clones',                 # Generate multiple target variants
    # 'unavailable',                 # Mark as unavailable
    'unused',                        # Suppress unused function warnings
    'used',                          # Force function to be emitted
    # 'visibility',                  # Control symbol visibility
    'warn_unused_result',            # Warn if return value unused
    # 'weak',                        # Weak symbol
    # 'weakref',                     # Weak reference
    'expected_throw',                # Control flow hardening
    'no_instrument_function',        # Disable function instrumentation
    'no_split_stack',                # Disable split stack
    'no_stack_limit',                # Disable stack limit checking
    'stack_protect',                 # Enable stack protection
    'tainted_args',                  # Mark arguments as tainted
]

# attribute_related_functions_option = {
#     '__attribute__((no_sanitize_address))': '-fsanitize=address',
#     '__attribute__((no_address_safety_analysis))': '-fsanitize=address',
#     '__attribute__((no_sanitize_thread))': '-fsanitize=thread',
#     '__attribute__((no_sanitize_undefined))': '-fsanitize=undefined',
#     '__attribute__((expected_throw))': '-fharden-control-flow-redundancy',
#     '__attribute__((no_instrument_function))': ['-finstrument-functions', '-p', '-pg'],
#     '__attribute__((no_split_stack))': '-fsplit-stack',
#     '__attribute__((no_stack_limit))': ['-fstack-limit-register', '-fstack-limit-symbol'],
#     '__attribute__((stack_protect))': ['-fstack-protector', '-fstack-protector-strong', '-fstack-protector-explicit'],
#     '__attribute__((tainted_args))': ['-Wanalyzer-tainted-allocation-size', '-Wanalyzer-tainted-array-index', '-Wanalyzer-tainted-divisor', '-Wanalyzer-tainted-offset', '-Wanalyzer-tainted-size'],
# }

function_related_attributes_option = {
    'expected_throw': '-fharden-control-flow-redundancy',
    'no_instrument_function': ['-finstrument-functions', '-p', '-pg'],
    'no_split_stack': '-fsplit-stack',
    'no_stack_limit': ['-fstack-limit-register', '-fstack-limit-symbol'],
    'stack_protect': ['-fstack-protector', '-fstack-protector-strong', '-fstack-protector-explicit'],
    'tainted_args': ['-Wanalyzer-tainted-allocation-size', '-Wanalyzer-tainted-array-index', '-Wanalyzer-tainted-divisor', '-Wanalyzer-tainted-offset', '-Wanalyzer-tainted-size'], 
}


def form_aligned(info):
    """
    Generate an aligned attribute with a valid alignment value.
    
    Selects from common alignment values that are powers of 2 or
    the compiler's maximum alignment constant.
    
    Args:
        info: Dictionary containing variable/function information
        
    Returns:
        String containing the formatted aligned attribute
    """
    num_candidate = ['4', '8', '16', '32', '64', '__BIGGEST_ALIGNMENT__']
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

def insert_nonstring(info):
    var_type = info['Type']
    var_type = var_type.replace(' ', '')
    if 'char[' in var_type:
        return '__attribute__((nonstring))'

def form_mode(info):
    var_type = info['Type']
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

def form_noreturn(info):
    func_ret_type = info['Type']
    if func_ret_type == 'void':
        return '__attribute__((noreturn))'

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
    isas = random.sample(isa_list, 4)
    isas = [f'"{_}"' for _ in isas]
    form_isa = ','.join(map(str, isas))
    return f'__attribute__((target_clones({form_isa})))'

def form_zero_call_used_regs(info):
    choice_list = ['skip', 'used', 'used-gpr', 'used-arg', 'used-gpr-arg', 'all', 'all-gpr', 
                   'all-arg', 'all-gpr-arg', 'leafy', 'leafy-gpr', 'leafy-arg', 'leafy-gpr-arg']
    choice = random.choice(choice_list)
    return f'__attribute__((zero_call_used_regs("{choice}")))'