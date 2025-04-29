"""
Generate hints for compilers (typically at source code level)
Mainly for C/C++ (that can be "understood" by clang/gcc, such as __restrict__, __no_alias__, etc.)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Set, Dict, Any
import re

# Import all available hints from the unified cpp_hints module
from llm4opt.hints import (
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


class HintGenerator:
    """Class to analyze code and generate compiler hints."""
    
    def __init__(self, verbose: bool = False, output_format: str = "inline"):
        """Initialize the hint generator.
        
        Args:
            verbose: Whether to print verbose output
            output_format: Format of the generated hints (inline, comment, separate)
        """
        self.verbose = verbose
        self.output_format = output_format
        self.supported_extensions = {'.c', '.cpp', '.cc', '.h', '.hpp'}
    
    def is_supported_file(self, file_path: Path) -> bool:
        """Check if the file is supported for hint generation."""
        return file_path.suffix.lower() in self.supported_extensions
    
    def analyze_file(self, file_path: Path) -> Dict[str, List[Any]]:
        """Analyze a file and identify potential optimization opportunities.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Dict of function names to list of suggested hints
        """
        if self.verbose:
            print(f"Analyzing {file_path}")
        
        # Read the file content
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Dictionary to store function names and their suggested hints
        hints = {}
        
        # Analyze for potential aliasing hints
        self._analyze_aliasing(content, hints)
        
        # Analyze for potential branch prediction hints
        self._analyze_branch_prediction(content, hints)
        
        # Analyze for potential memory access pattern hints
        self._analyze_memory_access(content, hints)
        
        # Analyze for potential function attribute hints
        self._analyze_function_attributes(content, hints)
        
        # Analyze for potential stack protection hints
        self._analyze_stack_protection(content, hints)
        
        return hints
    
    def _analyze_aliasing(self, content: str, hints: Dict[str, List[Any]]) -> None:
        """Analyze code for potential aliasing hints."""
        # Simple pattern to detect functions with pointer parameters
        func_pattern = re.compile(r'(\w+)\s+(\w+)\s*\(([^)]*)\)')
        for match in func_pattern.finditer(content):
            return_type, func_name, params = match.groups()
            
            # Check if the function has multiple pointer parameters
            if params.count('*') > 1:
                if func_name not in hints:
                    hints[func_name] = []
                hints[func_name].append(('restrict', 'Consider using restrict for pointer parameters'))
                
                # Check for common patterns like memcpy, where src and dst shouldn't alias
                if any(x in func_name.lower() for x in ['copy', 'memcpy', 'move']):
                    params_list = [p.strip() for p in params.split(',')]
                    src_dst_pattern = ['src', 'dst'] if 'src' in ''.join(params_list) else ['source', 'destination']
                    if any(src in ''.join(params_list) for src in src_dst_pattern):
                        hints[func_name].append(('no_alias', f'Consider using no_alias for {src_dst_pattern[0]} and {src_dst_pattern[1]} parameters'))
    
    def _analyze_branch_prediction(self, content: str, hints: Dict[str, List[Any]]) -> None:
        """Analyze code for potential branch prediction hints."""
        # Patterns suggesting error handling or rare conditions
        error_patterns = ['error', 'exception', 'fail', 'invalid', 'unlikely']
        check_pattern = re.compile(r'if\s*\(\s*([^)]*)\s*\)')
        func_pattern = re.compile(r'(\w+)\s+(\w+)\s*\(([^)]*)\)[^{]*{')
        
        current_func = None
        for line_num, line in enumerate(content.splitlines()):
            # Track current function
            func_match = func_pattern.search(line)
            if func_match:
                current_func = func_match.group(2)
            
            # Check for conditional statements that might benefit from branch hints
            check_match = check_pattern.search(line)
            if check_match and current_func:
                condition = check_match.group(1).lower()
                
                # Check if the condition suggests an error case
                is_error_case = any(error in condition for error in error_patterns)
                if is_error_case:
                    if current_func not in hints:
                        hints[current_func] = []
                    hints[current_func].append(('unlikely', 'Consider using unlikely for error conditions'))
                
                # Check for conditions that might be hot paths
                if not is_error_case and ('== 0' in condition or '!= 0' in condition):
                    if current_func not in hints:
                        hints[current_func] = []
                    hints[current_func].append(('likely', 'Consider using likely for common conditions'))
    
    def _analyze_memory_access(self, content: str, hints: Dict[str, List[Any]]) -> None:
        """Analyze code for potential memory access pattern hints."""
        # Simple pattern to detect array accesses in loops
        array_access_pattern = re.compile(r'for\s*\([^)]*\).*\[[^\]]*\]')
        func_pattern = re.compile(r'(\w+)\s+(\w+)\s*\(([^)]*)\)[^{]*{')
        
        current_func = None
        for line_num, line in enumerate(content.splitlines()):
            # Track current function
            func_match = func_pattern.search(line)
            if func_match:
                current_func = func_match.group(2)
            
            # Check for array accesses in loops
            if array_access_pattern.search(line) and current_func:
                if current_func not in hints:
                    hints[current_func] = []
                    
                # Suggest alignment hints for array accesses
                hints[current_func].append(('aligned', 'Consider alignment hints for array accesses'))
                
                # Check for sequential access patterns
                if 'i+' in line or 'i + ' in line or '[i]' in line:
                    hints[current_func].append(('prefetch_locality', 'Consider prefetch hints for sequential access'))
                
                # Check for large memory operations
                if 'memcpy' in line or 'memset' in line or 'memmove' in line:
                    hints[current_func].append(('nontemporal', 'Consider non-temporal hints for large memory operations'))
    
    def _analyze_function_attributes(self, content: str, hints: Dict[str, List[Any]]) -> None:
        """Analyze code for potential function attribute hints."""
        # Simple pattern to detect functions
        func_pattern = re.compile(r'(\w+)\s+(\w+)\s*\(([^)]*)\)')
        for match in func_pattern.finditer(content):
            return_type, func_name, params = match.groups()
            
            # Check if the function is small and called frequently
            if len(params.split(',')) <= 2:  # Simple heuristic for small functions
                if func_name not in hints:
                    hints[func_name] = []
                hints[func_name].append(('inline', 'Consider inlining small functions'))
            
            # Check for pure mathematical functions
            if return_type != 'void' and not '*' in return_type:
                if any(x in func_name.lower() for x in ['calc', 'compute', 'sum', 'min', 'max', 'square', 'sqrt']):
                    if func_name not in hints:
                        hints[func_name] = []
                    hints[func_name].append(('pure', 'Consider pure attribute for mathematical functions'))
            
            # Check for error handling functions
            if any(x in func_name.lower() for x in ['error', 'handle', 'exception', 'fail']):
                if func_name not in hints:
                    hints[func_name] = []
                hints[func_name].append(('cold', 'Consider cold attribute for error handling functions'))
            
            # Check for hot functions (performance-critical)
            if any(x in func_name.lower() for x in ['process', 'main', 'update', 'render', 'critical']):
                if func_name not in hints:
                    hints[func_name] = []
                hints[func_name].append(('hot', 'Consider hot attribute for performance-critical functions'))
                hints[func_name].append(('optimize', 'Consider setting optimization level for performance-critical functions'))
    
    def _analyze_stack_protection(self, content: str, hints: Dict[str, List[Any]]) -> None:
        """Analyze code for potential stack protection hints."""
        # Check for potentially dangerous buffer operations
        buffer_patterns = ['buffer', 'strcpy', 'memcpy', 'sprintf', 'gets']
        func_pattern = re.compile(r'(\w+)\s+(\w+)\s*\(([^)]*)\)[^{]*{')
        
        current_func = None
        for line_num, line in enumerate(content.splitlines()):
            # Track current function
            func_match = func_pattern.search(line)
            if func_match:
                current_func = func_match.group(2)
            
            # Check for potentially dangerous buffer operations
            if any(pattern in line for pattern in buffer_patterns) and current_func:
                if current_func not in hints:
                    hints[current_func] = []
                hints[current_func].append(('stack_protect', 'Consider stack protection for buffer operations'))
    
    def generate_hints(self, file_path: Path) -> str:
        """Generate compiler hints for a file.
        
        Args:
            file_path: Path to the file to generate hints for
            
        Returns:
            String containing the generated hints
        """
        hints = self.analyze_file(file_path)
        
        if not hints:
            return f"No optimization hints generated for {file_path}"
        
        result = [f"Compiler hints for {file_path}:"]
        for func_name, func_hints in hints.items():
            result.append(f"\nFunction: {func_name}")
            for hint_type, hint_desc in func_hints:
                result.append(f"  - {hint_desc}")
                
                # Add specific hint implementation based on the hint type
                if self.output_format == "inline":
                    if hint_type == 'restrict':
                        result.append(f"    Suggested implementation: __restrict__ or 'restrict' keyword in C99")
                    elif hint_type == 'no_alias':
                        result.append(f"    Suggested implementation: __attribute__((nonnull, access(read_only, 1), access(write_only, 2)))")
                    elif hint_type == 'likely':
                        result.append(f"    Suggested implementation: __builtin_expect(condition, 1)")
                    elif hint_type == 'unlikely':
                        result.append(f"    Suggested implementation: __builtin_expect(condition, 0)")
                    elif hint_type == 'aligned':
                        result.append(f"    Suggested implementation: __attribute__((aligned(16)))")
                    elif hint_type == 'prefetch_locality':
                        result.append(f"    Suggested implementation: __builtin_prefetch(addr, 0, 3)")
                    elif hint_type == 'nontemporal':
                        result.append(f"    Suggested implementation: Use non-temporal store intrinsics like _mm_stream_*")
                    elif hint_type == 'inline':
                        result.append(f"    Suggested implementation: inline or __attribute__((always_inline))")
                    elif hint_type == 'pure':
                        result.append(f"    Suggested implementation: __attribute__((pure))")
                    elif hint_type == 'cold':
                        result.append(f"    Suggested implementation: __attribute__((cold))")
                    elif hint_type == 'hot':
                        result.append(f"    Suggested implementation: __attribute__((hot))")
                    elif hint_type == 'optimize':
                        result.append(f"    Suggested implementation: __attribute__((optimize(\"O3\")))")
                    elif hint_type == 'stack_protect':
                        result.append(f"    Suggested implementation: Compile with -fstack-protector or use stack_protect context")
        
        return "\n".join(result)
    
    def process_path(self, path: Path) -> None:
        """Process a file or directory to generate hints.
        
        Args:
            path: Path to the file or directory to process
        """
        if path.is_file():
            if self.is_supported_file(path):
                print(self.generate_hints(path))
            else:
                if self.verbose:
                    print(f"Skipping unsupported file: {path}")
        elif path.is_dir():
            for file_path in path.glob('**/*'):
                if file_path.is_file() and self.is_supported_file(file_path):
                    print(self.generate_hints(file_path))


def main():
    """Main entry point for the CLI tool."""
    parser = argparse.ArgumentParser(
        description="Generate compiler hints for C/C++ code"
    )
    parser.add_argument(
        "path", 
        type=str, 
        help="Path to a file or directory to analyze"
    )
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true", 
        help="Enable verbose output"
    )
    parser.add_argument(
        "-f", "--format", 
        choices=["inline", "comment", "separate"], 
        default="inline",
        help="Format of the generated hints (inline, comment, separate)"
    )
    parser.add_argument(
        "-o", "--output", 
        type=str, 
        help="Output file to write the hints to (default: stdout)"
    )
    
    args = parser.parse_args()
    
    # Create the hint generator
    generator = HintGenerator(
        verbose=args.verbose,
        output_format=args.format
    )
    
    # Process the path
    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path does not exist: {path}")
        sys.exit(1)
    
    # If output is specified, redirect stdout to the output file
    if args.output:
        with open(args.output, 'w') as f:
            # Redirect stdout
            original_stdout = sys.stdout
            sys.stdout = f
            
            # Process the path
            generator.process_path(path)
            
            # Restore stdout
            sys.stdout = original_stdout
    else:
        # Process the path with output to stdout
        generator.process_path(path)


if __name__ == "__main__":
    main()





