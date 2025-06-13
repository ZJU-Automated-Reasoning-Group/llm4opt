"""
Sanitizer Configuration and Attribute Mapping

This module defines sanitizer configurations and their corresponding attributes
for both GCC and Clang/LLVM compilers. It provides functionality to select
compatible sanitizer combinations and generate appropriate no_sanitize attributes
for testing compiler behavior with different sanitizer configurations.

Key Components:

1. GCC Sanitizers:
   - AddressSanitizer (ASan): Memory error detection
   - ThreadSanitizer (TSan): Data race detection  
   - UndefinedBehaviorSanitizer (UBSan): Undefined behavior detection
   - LeakSanitizer: Memory leak detection
   - Various UBSan sub-sanitizers for specific undefined behaviors

2. LLVM/Clang Sanitizers:
   - Similar to GCC but with some differences in naming and availability
   - DataFlowSanitizer: Information flow tracking
   - MemorySanitizer: Uninitialized memory access detection
   - SafeStack: Stack buffer overflow protection

3. Sanitizer Exclusion Rules:
   - ThreadSanitizer is incompatible with AddressSanitizer and LeakSanitizer
   - MemorySanitizer is incompatible with AddressSanitizer and ThreadSanitizer
   - Automatic handling of mutually exclusive sanitizer combinations

4. Attribute Generation:
   - Maps sanitizer flags to corresponding no_sanitize attributes
   - Supports both GCC and Clang attribute syntax variations
   - Random selection of compatible sanitizer combinations

Functions:
- choose_gcc_sanitize(): Selects compatible GCC sanitizers and generates attributes
- choose_llvm_sanitize(): Selects compatible LLVM sanitizers and generates attributes

Usage:
    This module is used by the attribute insertion testing framework to:
    1. Generate random sanitizer combinations for testing
    2. Create corresponding no_sanitize attributes for selective disabling
    3. Ensure sanitizer compatibility to avoid compilation errors

The sanitizer attributes generated here are inserted into test programs to
evaluate compiler behavior when sanitizers are selectively disabled on
specific functions or variables.

Part of the LLM4OPT project for automated compiler testing and optimization.
"""

import random

gcc_san = [
    '-fsanitize=address',
    '-fsanitize=pointer-compare -fsanitize=address',
    '-fsanitize=pointer-subtract -fsanitize=address',
    # '-fsanitize=shadow-call-stack',
    '-fsanitize=thread',
    '-fsanitize=leak',
    '-fsanitize=undefined',
    '-fsanitize=shift',
    '-fsanitize=shift-exponent',
    '-fsanitize=shift-base',
    '-fsanitize=integer-divide-by-zero',
    '-fsanitize=unreachable',
    '-fsanitize=vla-bound',
    '-fsanitize=null',
    '-fsanitize=return',
    '-fsanitize=signed-integer-overflow',
    '-fsanitize=bounds',
    '-fsanitize=bounds-strict',
    '-fsanitize=alignment',
    '-fsanitize=object-size',
    '-fsanitize=float-divide-by-zero',
    '-fsanitize=float-cast-overflow',
    '-fsanitize=nonnull-attribute',
    '-fsanitize=returns-nonnull-attribute',
    '-fsanitize=bool',
    '-fsanitize=enum',
    '-fsanitize=vptr',
    '-fsanitize=pointer-overflow',
    '-fsanitize=builtin'
]


gcc_san_attributes = {
    '-fsanitize=address': 'no_sanitize_address',
    '-fsanitize=thread': 'no_sanitize_thread',
    '-fsanitize=undefined': 'no_sanitize_undefined',
    '-fsanitize=shift': 'no_sanitize ("shift")',
    '-fsanitize=shift-exponent': 'no_sanitize ("shift-exponent")',
    '-fsanitize=shift-base': 'no_sanitize ("shift-base")',
    '-fsanitize=integer-divide-by-zero': 'no_sanitize ("integer-divide-by-zero")',
    '-fsanitize=unreachable': 'no_sanitize ("unreachable")',
    '-fsanitize=vla-bound': 'no_sanitize ("vla-bound")',
    '-fsanitize=null': 'no_sanitize ("null")',
    '-fsanitize=return': 'no_sanitize ("return")',
    '-fsanitize=signed-integer-overflow': 'no_sanitize ("signed-integer-overflow")',
    '-fsanitize=bounds': 'no_sanitize ("bounds")',
    '-fsanitize=bounds-strict': 'no_sanitize ("bounds-strict")',
    '-fsanitize=alignment': 'no_sanitize ("alignment")',
    '-fsanitize=object-size': 'no_sanitize ("object-size")',
    '-fsanitize=float-divide-by-zero': 'no_sanitize ("float-divide-by-zero")',
    '-fsanitize=float-cast-overflow': 'no_sanitize ("float-cast-overflow")',
    '-fsanitize=nonnull-attribute': 'no_sanitize ("nonnull-attribute")',
    '-fsanitize=returns-nonnull-attribute': 'no_sanitize ("returns-nonnull-attribute")',
    '-fsanitize=bool': 'no_sanitize ("bool")',
    '-fsanitize=enum': 'no_sanitize ("enum")',
    '-fsanitize=vptr': 'no_sanitize ("vptr")',
    '-fsanitize=pointer-overflow': 'no_sanitize ("pointer-overflow")',
    '-fsanitize=builtin': 'no_sanitize ("builtin")'
}

llvm_san_attributes ={
    '-fsanitize=address': ['__attribute__((disable_sanitizer_instrumentation))', '__attribute__((no_sanitize_address))', '__attribute__((no_address_safety_analysis))'],
    '-fsanitize=thread': '__attribute__((no_sanitize_thread))',
    '-fsanitize=memory': '__attribute__((no_sanitize_memory))',
    '-fsanitize=undefined': '__attribute__((disable_sanitizer_instrumentation))',
    '-fsanitize=dataflow': '__attribute__((disable_sanitizer_instrumentation))',
}

llvm_san = [
    '-fsanitize=address',
    '-fsanitize=thread',
    '-fsanitize=memory',
    '-fsanitize=undefined',
    '-fsanitize=dataflow',
    '-fsanitize=safe-stack'
]

gcc_exclude_san = {
    '-fsanitize=thread': set(['-fsanitize=address', '-fsanitize=leak', '-fsanitize=pointer-compare -fsanitize=address', '-fsanitize=pointer-subtract -fsanitize=address']),
    '-fsanitize=leak': set(['-fsanitize=thread']),
    '-fsanitize=address': set(['-fsanitize=thread']),
    '-fsanitize=pointer-compare -fsanitize=address': set(['-fsanitize=thread']),
    '-fsanitize=pointer-subtract -fsanitize=address': set(['-fsanitize=thread'])
}

llvm_exclude_san = {
    '-fsanitize=thread': set(['-fsanitize=address', '-fsanitize=memory', '-fsanitize=safe-stack']),
    '-fsanitize=address': set(['-fsanitize=thread', '-fsanitize=memory']),
    '-fsanitize=memory': set(['-fsanitize=address', '-fsanitize=thread']),
    '-fsanitize=safe-stack': set(['-fsanitize=thread'])
}


def choose_gcc_sanitize():
    select_sans = set()
    san_list = gcc_san
    random.shuffle(san_list)
    for san in san_list:
        if random.choice([0,1]):
            select_sans.add(san)
    exsans = set()
    for san in select_sans:
        if san in exsans:
            continue
        if san in gcc_exclude_san.keys():
            esan = gcc_exclude_san[san]
            exsans = exsans | esan
    select_sans = select_sans - exsans
    san_attr = []
    attribute_func_map = {}
    for san in select_sans:
        if san in gcc_san_attributes.keys():
            san_attr.append(gcc_san_attributes[san])
            attribute_func_map[gcc_san_attributes[san]] = san
    return attribute_func_map, san_attr

def choose_llvm_sanitize():
    select_sans = set()
    san_list = llvm_san
    random.shuffle(san_list)
    for san in san_list:
        if random.choice([0,1]):
            select_sans.add(san)
    exsans = set()
    for san in select_sans:
        if san in exsans:
            continue
        if san in llvm_exclude_san.keys():
            esan = llvm_exclude_san[san]
            exsans = exsans | esan
    select_sans = select_sans - exsans
    san_attr = set()
    attribute_func_map = {}
    for san in select_sans:
        if san in llvm_san_attributes.keys():
            if isinstance(llvm_san_attributes[san], str):
                san_attr.add(llvm_san_attributes[san])
                attribute_func_map[llvm_san_attributes[san]] = san
            else:
                random_san = random.choice(llvm_san_attributes[san])
                san_attr.add(random_san)
                attribute_func_map[random_san] = san
    return attribute_func_map, list(san_attr)