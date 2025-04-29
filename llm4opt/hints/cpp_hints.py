"""C/C++ Compiler Optimization Hints.

This module provides a comprehensive collection of hints for C/C++ compiler
optimizations, including memory access patterns, aliasing, and function attributes.
These hints correspond to various compiler-specific attributes and directives in C/C++.

The module is organized into several categories:
- Memory alignment and access patterns
- Pointer aliasing
- Function optimization attributes
- Memory allocation behavior
- Stack and buffer management

Example:
    @restrict
    @aligned(16)
    def process_vector(vec: List[float]):
        # Compiler knows vec is 16-byte aligned and non-aliasing
        pass

    @inline
    @hot
    def performance_critical_function(x: int) -> int:
        # This function will be inlined and optimized for speed
        return x * 2

    with prefetch_locality(3):
        # Data will be prefetched with high temporal locality
        data = load_data()
"""

from functools import wraps
from typing import Callable, TypeVar, Any, Optional, List, Dict, Union, ContextManager
from contextlib import contextmanager

T = TypeVar('T')

# ==========================================
# Memory Access Pattern Hints
# ==========================================

def aligned(bytes_alignment: int):
    """Decorator to indicate that memory accessed in the function is aligned.
    
    This is similar to GCC's __attribute__((aligned(bytes_alignment))).
    
    Args:
        bytes_alignment: Alignment in bytes (typically a power of 2)
        
    Returns:
        Callable: A decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def assume_aligned(alignment: int):
    """Decorator factory to indicate that returned pointer is aligned.
    
    This is equivalent to GCC's __attribute__((assume_aligned(alignment))).
    
    Args:
        alignment: Alignment in bytes (typically a power of 2)
        
    Returns:
        Callable: A decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def cache_aligned(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to indicate that memory accessed in the function is cache-line aligned.
    
    This typically means 64-byte alignment on modern x86 processors.
    
    Args:
        func: The function to decorate
        
    Returns:
        Callable: The decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


@contextmanager
def prefetch_locality(level: int = 3) -> ContextManager[None]:
    """Context manager to suggest prefetching with specified temporal locality.
    
    Similar to __builtin_prefetch in C/C++.
    
    Args:
        level: Temporal locality level (0-3, where 3 is highest)
        
    Example:
        with prefetch_locality(3):
            data = load_data()
    
    Yields:
        None
    """
    yield


def nontemporal(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to indicate that memory accessed in the function doesn't need to be cached.
    
    This is useful for streaming operations where data is only accessed once,
    similar to using non-temporal store instructions (_mm_stream_* in x86).
    
    Args:
        func: The function to decorate
        
    Returns:
        Callable: The decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def stride_pattern(pattern: str):
    """Decorator factory to provide a hint about memory access stride pattern.
    
    Args:
        pattern: String describing the stride pattern ('unit', 'constant', 'irregular')
        
    Returns:
        Callable: A decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


@contextmanager
def no_flush() -> ContextManager[None]:
    """Context manager to suggest not flushing cache after memory operations.
    
    Example:
        with no_flush():
            write_data(buffer)
    
    Yields:
        None
    """
    yield


# ==========================================
# Pointer Aliasing Hints
# ==========================================

def restrict(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to indicate all pointer parameters are restricted (non-aliasing).
    
    This is equivalent to using the 'restrict' keyword in C99/C++ or __restrict in GCC/Clang.
    
    Args:
        func: The function to decorate
        
    Returns:
        Callable: The decorated function with __restrict__ hint
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def no_alias(*param_names: str) -> Callable:
    """Decorator factory to specify which parameters don't alias each other.
    
    Args:
        *param_names: Names of parameters that don't alias
        
    Returns:
        Callable: A decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def may_alias(*param_names: str) -> Callable:
    """Decorator factory to specify which parameters may alias each other.
    
    Args:
        *param_names: Names of parameters that may alias
        
    Returns:
        Callable: A decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ==========================================
# Function Optimization Hints
# ==========================================

def inline(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to suggest inlining the function at call sites.
    
    This is equivalent to C/C++'s 'inline' keyword.
    
    Args:
        func: The function to decorate
        
    Returns:
        Callable: The decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def noinline(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to suggest not inlining the function.
    
    This is equivalent to GCC's __attribute__((noinline)).
    
    Args:
        func: The function to decorate
        
    Returns:
        Callable: The decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def always_inline(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to force inlining the function at call sites.
    
    This is equivalent to GCC's __attribute__((always_inline)).
    
    Args:
        func: The function to decorate
        
    Returns:
        Callable: The decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def hot(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to indicate that a function is hot (frequently executed).
    
    This is equivalent to GCC's __attribute__((hot)).
    
    Args:
        func: The function to decorate
        
    Returns:
        Callable: The decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def cold(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to indicate that a function is cold (rarely executed).
    
    This is equivalent to GCC's __attribute__((cold)).
    
    Args:
        func: The function to decorate
        
    Returns:
        Callable: The decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def pure(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to indicate that a function is pure (no side effects).
    
    This is equivalent to GCC's __attribute__((pure)).
    Pure functions can be subject to common subexpression elimination and
    loop optimization.
    
    Args:
        func: The function to decorate
        
    Returns:
        Callable: The decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def const(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to indicate that a function is const (no side effects and doesn't read global memory).
    
    This is equivalent to GCC's __attribute__((const)).
    Const functions can be subject to common subexpression elimination and 
    can be executed fewer times than the calls appear in the source.
    
    Args:
        func: The function to decorate
        
    Returns:
        Callable: The decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def optimize(level: str):
    """Decorator factory to specify optimization level for a function.
    
    This is similar to GCC's __attribute__((optimize("-O2"))) or #pragma optimize.
    
    Args:
        level: Optimization level ('O0', 'O1', 'O2', 'O3', 'Os', 'Ofast')
        
    Returns:
        Callable: A decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def flatten(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to suggest flattening a function (inlining all function calls within it).
    
    This is equivalent to GCC's __attribute__((flatten)).
    
    Args:
        func: The function to decorate
        
    Returns:
        Callable: The decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def section(name: str):
    """Decorator factory to specify which section a function should be placed in.
    
    This is equivalent to GCC's __attribute__((section("name"))).
    
    Args:
        name: Section name
        
    Returns:
        Callable: A decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ==========================================
# Memory Allocation Hints
# ==========================================

def malloc_like(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to indicate that a function behaves like malloc.
    
    This is equivalent to GCC's __attribute__((malloc)).
    Functions with this attribute return a pointer that cannot alias
    any other pointer valid when the function returns.
    
    Args:
        func: The function to decorate
        
    Returns:
        Callable: The decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def returns_nonnull(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to indicate that a function never returns NULL.
    
    This is equivalent to GCC's __attribute__((returns_nonnull)).
    
    Args:
        func: The function to decorate
        
    Returns:
        Callable: The decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def alloc_size(*param_indices: int):
    """Decorator factory to specify which parameters determine allocation size.
    
    This is equivalent to GCC's __attribute__((alloc_size(...))).
    
    Args:
        *param_indices: Indices of parameters that determine allocation size (1-based)
        
    Returns:
        Callable: A decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def returns_twice(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to indicate that a function may return more than once.
    
    This is useful for functions like setjmp.
    This is equivalent to GCC's __attribute__((returns_twice)).
    
    Args:
        func: The function to decorate
        
    Returns:
        Callable: The decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


# ==========================================
# Stack and Buffer Management Hints
# ==========================================

@contextmanager
def stack_protect() -> ContextManager[None]:
    """Context manager to enable stack protection for the enclosed block.
    
    This is equivalent to GCC's -fstack-protector.
    
    Example:
        with stack_protect():
            buffer = create_buffer(1024)
            process_data(buffer)
    
    Yields:
        None
    """
    yield


def no_stack_protector(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to disable stack protection for a function.
    
    This is equivalent to GCC's __attribute__((no_stack_protector)).
    
    Args:
        func: The function to decorate
        
    Returns:
        Callable: The decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def no_sanitize(*sanitizers: str):
    """Decorator factory to disable specific sanitizers for a function.
    
    This is equivalent to GCC's __attribute__((no_sanitize(...)))
    
    Args:
        *sanitizers: Names of sanitizers to disable (e.g., 'address', 'thread')
        
    Returns:
        Callable: A decorator function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ==========================================
# Branch and Control Flow Hints
# ==========================================

def likely(condition: bool) -> bool:
    """Hint that a condition is likely to be true.
    
    This is equivalent to GCC's __builtin_expect(condition, 1).
    
    Args:
        condition: The condition to evaluate
        
    Returns:
        bool: The original condition value
    """
    return condition


def unlikely(condition: bool) -> bool:
    """Hint that a condition is likely to be false.
    
    This is equivalent to GCC's __builtin_expect(condition, 0).
    
    Args:
        condition: The condition to evaluate
        
    Returns:
        bool: The original condition value
    """
    return condition


def expect_true(func: Callable[..., bool]) -> Callable[..., bool]:
    """Decorator to indicate a function's return value is likely to be true.
    
    Args:
        func: The function to decorate
        
    Returns:
        Callable: The decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return likely(func(*args, **kwargs))
    return wrapper


def expect_false(func: Callable[..., bool]) -> Callable[..., bool]:
    """Decorator to indicate a function's return value is likely to be false.
    
    Args:
        func: The function to decorate
        
    Returns:
        Callable: The decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return unlikely(func(*args, **kwargs))
    return wrapper 