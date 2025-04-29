"""Examples demonstrating how to use the llm4opt.hints module.

This file contains practical examples of how to apply compiler optimization hints
to common programming patterns in C/C++ code, expressed through Python decorators
and context managers.
"""

from typing import List, Optional
import numpy as np  # Just for type annotations in examples

from llm4opt.hints import (
    # Memory alignment and access patterns
    aligned, cache_aligned, prefetch_locality, nontemporal, stride_pattern,
    
    # Pointer aliasing
    restrict, no_alias, may_alias,
    
    # Function optimization attributes
    inline, noinline, hot, cold, pure, const, optimize, flatten,
    
    # Memory allocation
    malloc_like, returns_nonnull, alloc_size,
    
    # Stack protection
    stack_protect, no_stack_protector,
    
    # Branch prediction
    likely, unlikely, expect_true, expect_false
)


# -------------------------------------------------------------------------
# Example 1: Vector operations with alignment and aliasing hints
# -------------------------------------------------------------------------

@restrict
@aligned(16)
def vector_add(a: List[float], b: List[float], c: List[float], n: int) -> None:
    """Add two vectors: c[i] = a[i] + b[i].
    
    Args:
        a: First input vector, aligned to 16 bytes
        b: Second input vector, aligned to 16 bytes
        c: Output vector, aligned to 16 bytes
        n: Vector length
    """
    # In C/C++, this would be compiled with:
    # - Vectorization (due to restricted pointers and alignment)
    # - No pointer aliasing checks (due to @restrict)
    # - Aligned vector loads/stores (due to @aligned)
    for i in range(n):
        c[i] = a[i] + b[i]


# -------------------------------------------------------------------------
# Example 2: Memory streaming with non-temporal hint
# -------------------------------------------------------------------------

@nontemporal
def copy_large_buffer(src: List[float], dst: List[float], n: int) -> None:
    """Copy a large buffer using non-temporal stores.
    
    This is useful for data that won't be reused soon, avoiding cache pollution.
    
    Args:
        src: Source buffer
        dst: Destination buffer
        n: Buffer size
    """
    # In C/C++, this would be compiled with non-temporal stores
    # (like _mm_stream_* instructions in x86)
    for i in range(n):
        dst[i] = src[i]


# -------------------------------------------------------------------------
# Example 3: Function optimization attributes
# -------------------------------------------------------------------------

@inline
def min_value(a: int, b: int) -> int:
    """Return the minimum of two values.
    
    This small function is a good candidate for inlining.
    
    Args:
        a: First value
        b: Second value
    
    Returns:
        int: The minimum value
    """
    return a if a < b else b


@hot
@optimize("O3")
def performance_critical_function(data: List[float], n: int) -> float:
    """Compute the sum of squares of a vector.
    
    This function is performance-critical and should be highly optimized.
    
    Args:
        data: Input vector
        n: Vector length
    
    Returns:
        float: Sum of squares
    """
    result = 0.0
    for i in range(n):
        result += data[i] * data[i]
    return result


@cold
@noinline
def error_handler(message: str) -> None:
    """Handle an error condition.
    
    This function is rarely called and shouldn't be inlined.
    
    Args:
        message: Error message
    """
    # Error handling code
    print(f"Error: {message}")


# -------------------------------------------------------------------------
# Example 4: Memory allocation with custom allocator
# -------------------------------------------------------------------------

@malloc_like
@returns_nonnull
@alloc_size(1)
def aligned_allocate(size: int) -> List[float]:
    """Allocate an aligned buffer.
    
    Args:
        size: Buffer size in elements
    
    Returns:
        List[float]: Newly allocated buffer
    """
    # This would correspond to a custom allocator in C/C++
    # The hints tell the compiler:
    # - The function behaves like malloc (returns newly allocated memory)
    # - It never returns NULL
    # - The allocation size is determined by the first parameter
    return [0.0] * size


# -------------------------------------------------------------------------
# Example 5: Branch prediction hints
# -------------------------------------------------------------------------

def process_data_with_branching(data: List[float], threshold: float) -> List[float]:
    """Process data with conditional branching.
    
    Args:
        data: Input data
        threshold: Threshold value for branching condition
    
    Returns:
        List[float]: Processed data
    """
    result = []
    
    for value in data:
        # Hint that this condition is usually true
        if likely(value > threshold):
            # This is the fast path, expected to be taken most often
            result.append(value * 2.0)
        else:
            # This is the slow path, less frequently taken
            result.append(0.0)
            
    return result


@expect_false
def is_error_condition(status_code: int) -> bool:
    """Check if a status code represents an error.
    
    This function's return value is expected to be false most of the time.
    
    Args:
        status_code: Status code to check
    
    Returns:
        bool: True if error, False otherwise
    """
    return status_code != 0


# -------------------------------------------------------------------------
# Example 6: Stack protection for buffer operations
# -------------------------------------------------------------------------

def process_user_input_safely(user_input: str) -> str:
    """Process user input with stack protection.
    
    Args:
        user_input: Input string from user
    
    Returns:
        str: Processed output
    """
    # Use stack protection for code that deals with user input
    with stack_protect():
        buffer = [0] * 1024
        # Fill buffer from user input (potentially unsafe)
        for i, char in enumerate(user_input[:1024]):
            buffer[i] = ord(char)
        # Process buffer
        result = ''.join(chr(x) for x in buffer if x != 0)
    
    return result


@no_stack_protector
def internal_buffer_operation(size: int) -> List[int]:
    """Internal buffer operation that doesn't need stack protection.
    
    This function is used for internal calculations with known-size buffers,
    so stack protection can be disabled for better performance.
    
    Args:
        size: Buffer size
    
    Returns:
        List[int]: Result buffer
    """
    buffer = [0] * size
    # Fill buffer with calculated values
    for i in range(size):
        buffer[i] = i * i
    
    return buffer


# -------------------------------------------------------------------------
# Example 7: Prefetching for performance
# -------------------------------------------------------------------------

def process_matrix(matrix: List[List[float]]) -> float:
    """Process a matrix with prefetching hints.
    
    Args:
        matrix: Input matrix
    
    Returns:
        float: Result of processing
    """
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    result = 0.0
    
    for i in range(rows):
        # Prefetch the next row with high temporal locality
        with prefetch_locality(3):
            next_row = matrix[i+1] if i+1 < rows else None
        
        # Process current row
        for j in range(cols):
            result += matrix[i][j]
    
    return result


# -------------------------------------------------------------------------
# Example 8: Function attributes for mathematical operations
# -------------------------------------------------------------------------

@pure
def dot_product(a: List[float], b: List[float], n: int) -> float:
    """Compute the dot product of two vectors.
    
    This is a pure function with no side effects.
    
    Args:
        a: First vector
        b: Second vector
        n: Vector length
    
    Returns:
        float: Dot product
    """
    result = 0.0
    for i in range(n):
        result += a[i] * b[i]
    return result


@const
def square(x: float) -> float:
    """Square a number.
    
    This is a const function with no side effects and no global memory access.
    
    Args:
        x: Input value
    
    Returns:
        float: x^2
    """
    return x * x


# -------------------------------------------------------------------------
# Example 9: Combined optimization techniques for matrix multiplication
# -------------------------------------------------------------------------

@restrict
@hot
@optimize("O3")
def matrix_multiply(
    a: List[List[float]],
    b: List[List[float]],
    c: List[List[float]],
    m: int,
    n: int,
    k: int
) -> None:
    """Multiply matrices: C = A * B.
    
    Uses multiple optimization hints:
    - restrict: A, B, and C don't alias
    - hot: This function is frequently called
    - optimize("O3"): Use highest optimization level
    
    Args:
        a: First matrix (m x n)
        b: Second matrix (n x k)
        c: Result matrix (m x k)
        m: Rows in A
        n: Columns in A / Rows in B
        k: Columns in B
    """
    for i in range(m):
        for j in range(k):
            c[i][j] = 0.0
            for l in range(n):
                c[i][j] += a[i][l] * b[l][j] 