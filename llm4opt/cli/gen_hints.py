"""Generate compiler hints for C/C++ code optimization."""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import re


class HintGenerator:
    """Analyze code and generate compiler hints."""
    
    SUPPORTED_EXTENSIONS = {'.c', '.cpp', '.cc', '.h', '.hpp'}
    HINT_IMPLEMENTATIONS = {
        'restrict': "__restrict__ or 'restrict' keyword in C99",
        'no_alias': "__attribute__((nonnull, access(read_only, 1), access(write_only, 2)))",
        'likely': "__builtin_expect(condition, 1)",
        'unlikely': "__builtin_expect(condition, 0)",
        'aligned': "__attribute__((aligned(16)))",
        'prefetch_locality': "__builtin_prefetch(addr, 0, 3)",
        'nontemporal': "Use non-temporal store intrinsics like _mm_stream_*",
        'inline': "inline or __attribute__((always_inline))",
        'pure': "__attribute__((pure))",
        'cold': "__attribute__((cold))",
        'hot': "__attribute__((hot))",
        'optimize': "__attribute__((optimize(\"O3\")))",
        'stack_protect': "Compile with -fstack-protector or use stack_protect context"
    }
    
    def __init__(self, verbose: bool = False, output_format: str = "inline"):
        self.verbose = verbose
        self.output_format = output_format
    
    def is_supported_file(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS
    
    def analyze_file(self, file_path: Path) -> Dict[str, List[Tuple[str, str]]]:
        """Analyze file and identify optimization opportunities."""
        if self.verbose:
            print(f"Analyzing {file_path}")
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        hints = {}
        lines = content.splitlines()
        
        # Combined analysis patterns
        func_pattern = re.compile(r'(\w+)\s+(\w+)\s*\(([^)]*)\)')
        check_pattern = re.compile(r'if\s*\(\s*([^)]*)\s*\)')
        array_access_pattern = re.compile(r'for\s*\([^)]*\).*\[[^\]]*\]')
        
        current_func = None
        for line_num, line in enumerate(lines):
            func_match = func_pattern.search(line)
            if func_match:
                current_func = func_match.group(2)
                self._analyze_function_hints(func_match, hints)
            
            if current_func:
                self._analyze_line_hints(line, current_func, hints, check_pattern, array_access_pattern)
        
        return hints
    
    def _analyze_function_hints(self, func_match, hints):
        """Analyze function-level hints."""
        return_type, func_name, params = func_match.groups()
        
        if func_name not in hints:
            hints[func_name] = []
        
        # Aliasing hints
        if params.count('*') > 1:
            hints[func_name].append(('restrict', 'Consider using restrict for pointer parameters'))
            if any(x in func_name.lower() for x in ['copy', 'memcpy', 'move']):
                hints[func_name].append(('no_alias', 'Consider using no_alias for src/dst parameters'))
        
        # Function attributes
        if len(params.split(',')) <= 2:
            hints[func_name].append(('inline', 'Consider inlining small functions'))
        
        if return_type != 'void' and '*' not in return_type:
            if any(x in func_name.lower() for x in ['calc', 'compute', 'sum', 'min', 'max', 'square', 'sqrt']):
                hints[func_name].append(('pure', 'Consider pure attribute for mathematical functions'))
        
        if any(x in func_name.lower() for x in ['error', 'handle', 'exception', 'fail']):
            hints[func_name].append(('cold', 'Consider cold attribute for error handling functions'))
        
        if any(x in func_name.lower() for x in ['process', 'main', 'update', 'render', 'critical']):
            hints[func_name].extend([
                ('hot', 'Consider hot attribute for performance-critical functions'),
                ('optimize', 'Consider setting optimization level for performance-critical functions')
            ])
    
    def _analyze_line_hints(self, line, current_func, hints, check_pattern, array_access_pattern):
        """Analyze line-level hints."""
        if current_func not in hints:
            hints[current_func] = []
        
        # Branch prediction
        check_match = check_pattern.search(line)
        if check_match:
            condition = check_match.group(1).lower()
            error_patterns = ['error', 'exception', 'fail', 'invalid', 'unlikely']
            
            if any(error in condition for error in error_patterns):
                hints[current_func].append(('unlikely', 'Consider using unlikely for error conditions'))
            elif '== 0' in condition or '!= 0' in condition:
                hints[current_func].append(('likely', 'Consider using likely for common conditions'))
        
        # Memory access patterns
        if array_access_pattern.search(line):
            hints[current_func].append(('aligned', 'Consider alignment hints for array accesses'))
            if any(x in line for x in ['i+', 'i + ', '[i]']):
                hints[current_func].append(('prefetch_locality', 'Consider prefetch hints for sequential access'))
        
        # Memory operations and stack protection
        if any(x in line for x in ['memcpy', 'memset', 'memmove']):
            hints[current_func].append(('nontemporal', 'Consider non-temporal hints for large memory operations'))
        
        if any(x in line for x in ['buffer', 'strcpy', 'sprintf', 'gets']):
            hints[current_func].append(('stack_protect', 'Consider stack protection for buffer operations'))
    
    def generate_hints(self, file_path: Path) -> str:
        """Generate compiler hints for a file."""
        hints = self.analyze_file(file_path)
        
        if not hints:
            return f"No optimization hints generated for {file_path}"
        
        result = [f"Compiler hints for {file_path}:"]
        for func_name, func_hints in hints.items():
            result.append(f"\nFunction: {func_name}")
            for hint_type, hint_desc in func_hints:
                result.append(f"  - {hint_desc}")
                if self.output_format == "inline" and hint_type in self.HINT_IMPLEMENTATIONS:
                    result.append(f"    Suggested implementation: {self.HINT_IMPLEMENTATIONS[hint_type]}")
        
        return "\n".join(result)
    
    def process_path(self, path: Path) -> None:
        """Process a file or directory to generate hints."""
        if path.is_file():
            if self.is_supported_file(path):
                print(self.generate_hints(path))
            elif self.verbose:
                print(f"Skipping unsupported file: {path}")
        elif path.is_dir():
            for file_path in path.glob('**/*'):
                if file_path.is_file() and self.is_supported_file(file_path):
                    print(self.generate_hints(file_path))


def main():
    """Main entry point for the CLI tool."""
    parser = argparse.ArgumentParser(description="Generate compiler hints for C/C++ code")
    parser.add_argument("path", help="Path to a file or directory to analyze")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-f", "--format", choices=["inline", "comment", "separate"], 
                       default="inline", help="Format of the generated hints")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    
    args = parser.parse_args()
    
    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path does not exist: {path}")
        sys.exit(1)
    
    generator = HintGenerator(verbose=args.verbose, output_format=args.format)
    
    if args.output:
        with open(args.output, 'w') as f:
            original_stdout = sys.stdout
            sys.stdout = f
            generator.process_path(path)
            sys.stdout = original_stdout
    else:
        generator.process_path(path)


if __name__ == "__main__":
    main()
