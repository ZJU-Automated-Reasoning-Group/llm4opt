"""Module for compiler optimization hints.

This module provides functions and decorators for generating hints
to guide the compiler in optimizing code. These hints correspond to
compiler-specific attributes, pragmas, and directives.

The hints are organized into categories:
- Memory access patterns and alignment
- Pointer aliasing
- Branch prediction and control flow
- Function optimization attributes
- Memory allocation behavior
- Stack protection and management

Example:
    @restrict
    @aligned(16)
    def process_vector(vec: List[float]):
        # Compiler knows vec is 16-byte aligned and non-aliasing
        pass
        
    if likely(condition):
        # This branch is predicted to be taken most often
        pass
"""

# Import main categories of hints
from llm4opt.hints.cpp_hints import (
    # Memory alignment and access patterns
    aligned, assume_aligned, cache_aligned, prefetch_locality,
    nontemporal, stride_pattern, no_flush,
    
    # Pointer aliasing
    restrict, no_alias, may_alias,
    
    # Function optimization attributes
    inline, noinline, always_inline, hot, cold, pure, const,
    optimize, flatten, section,
    
    # Memory allocation hints
    malloc_like, returns_nonnull, alloc_size, returns_twice,
    
    # Stack and buffer management
    stack_protect, no_stack_protector, no_sanitize,
    
    # Branch and control flow
    likely, unlikely, expect_true, expect_false
)

# Define what's available through the module
__all__ = [
    # Memory alignment and access patterns
    'aligned', 'assume_aligned', 'cache_aligned', 'prefetch_locality',
    'nontemporal', 'stride_pattern', 'no_flush',
    
    # Pointer aliasing
    'restrict', 'no_alias', 'may_alias',
    
    # Function optimization attributes
    'inline', 'noinline', 'always_inline', 'hot', 'cold', 'pure', 'const',
    'optimize', 'flatten', 'section',
    
    # Memory allocation hints
    'malloc_like', 'returns_nonnull', 'alloc_size', 'returns_twice',
    
    # Stack and buffer management
    'stack_protect', 'no_stack_protector', 'no_sanitize',
    
    # Branch and control flow
    'likely', 'unlikely', 'expect_true', 'expect_false'
]