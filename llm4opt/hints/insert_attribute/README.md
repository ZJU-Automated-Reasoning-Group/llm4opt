# Compiler Optimization Hints

This module provides a collection of decorators, context managers, and utility functions that serve as hints for compiler optimizations. These hints correspond to compiler-specific attributes, pragmas, and directives in C/C++ that can help the compiler generate more efficient code.

## Overview

The hints are organized into several categories:

1. **Memory alignment and access patterns** - Control how memory is accessed and aligned
2. **Pointer aliasing** - Specify aliasing relationships between pointers
3. **Function optimization attributes** - Control how functions are optimized
4. **Memory allocation behavior** - Specify allocation and deallocation behavior
5. **Stack and buffer management** - Control stack protection and memory safety
6. **Branch and control flow** - Provide branch prediction hints

## Usage

Import the specific hints you need:

```python
from llm4opt.hints import aligned, restrict, likely, inline
```

### Memory Alignment and Access Patterns

```python
@aligned(16)
def process_vector(data):
    """Function that works with aligned data."""
    # Compiler knows data is 16-byte aligned
    pass

@cache_aligned
def process_cache_line(data):
    """Function that works with cache-line aligned data."""
    pass

with prefetch_locality(3):
    # Code that benefits from data prefetching with high temporal locality
    data = load_data()

@nontemporal
def stream_process(data):
    """Function that processes data in a streaming fashion."""
    # Compiler can use non-temporal instructions
    pass

@stride_pattern("unit")
def process_contiguous(array):
    """Process array with unit stride (contiguous access)."""
    pass
```

### Pointer Aliasing

```python
@restrict
def vector_add(a, b, c):
    """Add vectors a and b, store result in c."""
    # Compiler knows a, b, c don't alias (like restrict in C)
    pass

@no_alias("src", "dst")
def my_memcpy(src, dst, size):
    """Copy from src to dst."""
    # Compiler knows src and dst don't overlap
    pass

@may_alias("a", "b")
def might_overlap(a, b):
    """Function where a and b might overlap."""
    # Compiler is cautious with optimizations
    pass
```

### Function Optimization Attributes

```python
@inline
def small_function(x):
    """Small function that's a good candidate for inlining."""
    return x * 2

@noinline
def logging_function(msg):
    """Function that shouldn't be inlined."""
    pass

@hot
def frequently_called():
    """Function that's called frequently."""
    # Compiler optimizes this function heavily
    pass

@cold
def error_handler():
    """Rarely executed function."""
    # Compiler optimizes for size, not speed
    pass

@pure
def calculate_hash(data):
    """Pure function with no side effects."""
    # Compiler can perform common subexpression elimination
    pass

@const
def square(x):
    """Const function (no side effects and doesn't read global memory)."""
    # Compiler can cache results or remove redundant calls
    return x * x

@optimize("O3")
def performance_critical():
    """Function that needs maximum optimization."""
    pass

@flatten
def wrapper_function():
    """Function where all called functions should be inlined."""
    pass
```

### Memory Allocation Behavior

```python
@malloc_like
def custom_allocator(size):
    """Function that allocates memory."""
    # Compiler knows return value is a newly allocated pointer
    pass

@returns_nonnull
def get_resource():
    """Function that never returns NULL."""
    # Compiler can optimize out NULL checks
    pass

@alloc_size(1, 2)
def matrix_allocate(rows, cols):
    """Function that allocates based on parameter values."""
    # Compiler knows allocation size is rows*cols
    pass

@returns_twice
def my_setjmp(buffer):
    """Function like setjmp that returns multiple times."""
    pass
```

### Stack and Buffer Management

```python
with stack_protect():
    # Code that needs stack protection
    buffer = create_buffer(1024)
    process_user_input(buffer)

@no_stack_protector
def performance_critical_no_overflow():
    """Function where stack protection can be disabled safely."""
    pass

@no_sanitize("address", "undefined")
def raw_memory_access():
    """Function that does low-level memory access."""
    pass
```

### Branch and Control Flow

```python
if likely(condition):
    # This branch is optimized for the true case
    pass

while unlikely(error_condition):
    # Error handling code is moved out of the main execution path
    handle_error()

@expect_true
def validation_check():
    """Function whose return value is usually true."""
    return check_condition()

@expect_false
def is_error_state():
    """Function whose return value is usually false."""
    return check_error()
```

## Implementation Details

These hints are implemented as Python decorators and context managers that don't actually modify the compiled code. In a real compiler implementation, these would be translated to corresponding C/C++ attributes or compiler directives.

The primary purpose of these hints is to document programmer intent and to provide a framework for static analysis tools or source-to-source transformers that can insert the appropriate compiler directives.

## C/C++ Equivalents

Here are the C/C++ equivalents for some of the hints:

* `aligned(16)` → `__attribute__((aligned(16)))` (GCC/Clang)
* `restrict` → `__restrict__` or C99's `restrict` keyword
* `inline` → `inline` keyword or `__attribute__((always_inline))`
* `likely(x)` → `__builtin_expect(x, 1)` (GCC/Clang)
* `unlikely(x)` → `__builtin_expect(x, 0)` (GCC/Clang)
* `pure` → `__attribute__((pure))` (GCC/Clang)
* `const` → `__attribute__((const))` (GCC/Clang)
* `malloc_like` → `__attribute__((malloc))` (GCC/Clang)

These hints can be used to express optimization opportunities that the compiler might not be able to derive automatically through static analysis. 