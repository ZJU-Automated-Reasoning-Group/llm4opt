"""
Validation using Alive2 for LLVM IR transformations.

This module provides functions to validate LLVM IR transformations using the Alive2 tool,
which can formally verify the correctness of compiler optimizations.

Alive2 GitHub: https://github.com/AliveToolkit/alive2
"""

import subprocess
import tempfile
import os
import re
from typing import List, Dict, Any, Optional, Union
from pathlib import Path


def check_alive2_installed() -> bool:
    """
    Check if Alive2 is installed and available in the system.
    
    Returns:
        True if Alive2 is installed, False otherwise
    """
    try:
        result = subprocess.run(['alive2', '--version'], 
                               capture_output=True, 
                               text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def generate_llvm_ir(source_code: str, language: str, output_file: str) -> bool:
    """
    Generate LLVM IR from source code.
    
    Args:
        source_code: Source code of the program
        language: Programming language ('c', 'cpp')
        output_file: Path to the output LLVM IR file
        
    Returns:
        True if IR generation succeeded, False otherwise
    """
    if language not in ('c', 'cpp'):
        raise ValueError(f"Unsupported language for LLVM IR generation: {language}")
    
    # Create a temporary file for the source code
    with tempfile.NamedTemporaryFile(suffix=f'.{language}', delete=False) as f:
        f.write(source_code.encode('utf-8'))
        source_file = f.name
    
    try:
        # Generate LLVM IR using clang
        compiler = 'clang' if language == 'c' else 'clang++'
        cmd = [compiler, '-S', '-emit-llvm', '-O0', source_file, '-o', output_file]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"LLVM IR generation error: {result.stderr}")
            return False
            
        return True
    finally:
        # Clean up the temporary source file
        if os.path.exists(source_file):
            os.remove(source_file)


def extract_function_from_ir(ir_file: str, function_name: str) -> Optional[str]:
    """
    Extract a specific function from an LLVM IR file.
    
    Args:
        ir_file: Path to the LLVM IR file
        function_name: Name of the function to extract
        
    Returns:
        String containing the extracted function IR, or None if not found
    """
    try:
        with open(ir_file, 'r') as f:
            ir_content = f.read()
            
        # Regular expression to match a function definition
        pattern = re.compile(r'define.*@' + re.escape(function_name) + r'\s*\([^{]*\{[^}]*\}', re.DOTALL)
        match = pattern.search(ir_content)
        
        if match:
            return match.group(0)
        else:
            return None
    except Exception as e:
        print(f"Error extracting function from IR: {e}")
        return None


def run_alive2_verification(src_ir: str, tgt_ir: str) -> Dict[str, Any]:
    """
    Run Alive2 to verify the correctness of an optimization.
    
    Args:
        src_ir: LLVM IR of the source (original) function
        tgt_ir: LLVM IR of the target (optimized) function
        
    Returns:
        Dict containing verification results:
        {
            'valid': bool,
            'output': str,
            'error': Optional[str]
        }
    """
    results = {
        'valid': False,
        'output': '',
        'error': None
    }
    
    if not check_alive2_installed():
        results['error'] = "Alive2 is not installed or not in PATH"
        return results
    
    # Create temporary files for the IR
    with tempfile.NamedTemporaryFile(suffix='.ll', delete=False) as src_file, \
         tempfile.NamedTemporaryFile(suffix='.ll', delete=False) as tgt_file:
        
        src_file.write(src_ir.encode('utf-8'))
        tgt_file.write(tgt_ir.encode('utf-8'))
        
        src_path = src_file.name
        tgt_path = tgt_file.name
    
    try:
        # Run Alive2 to verify the optimization
        cmd = ['alive2', src_path, tgt_path]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        results['output'] = result.stdout
        
        # Check if verification was successful
        if "Transformation seems to be correct!" in result.stdout:
            results['valid'] = True
        elif "Transformation doesn't verify!" in result.stdout:
            results['valid'] = False
            results['error'] = "Optimization is not semantically equivalent"
        else:
            results['error'] = f"Alive2 verification failed: {result.stderr}"
            
    except Exception as e:
        results['error'] = str(e)
    finally:
        # Clean up temporary files
        if os.path.exists(src_path):
            os.remove(src_path)
        if os.path.exists(tgt_path):
            os.remove(tgt_path)
    
    return results


def validate_with_alive2(
    original_program: str,
    optimized_program: str,
    language: str = 'c',
    function_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validate an optimization using Alive2.
    
    Args:
        original_program: Source code of the original program
        optimized_program: Source code of the optimized program
        language: Programming language ('c', 'cpp')
        function_name: Name of the function to validate (if None, tries to detect automatically)
        
    Returns:
        Dict containing validation results:
        {
            'valid': bool,
            'output': str,
            'error': Optional[str]
        }
    """
    results = {
        'valid': False,
        'output': '',
        'error': None
    }
    
    if not check_alive2_installed():
        results['error'] = "Alive2 is not installed or not in PATH"
        return results
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate LLVM IR for both programs
            original_ir_file = os.path.join(temp_dir, 'original.ll')
            optimized_ir_file = os.path.join(temp_dir, 'optimized.ll')
            
            if not generate_llvm_ir(original_program, language, original_ir_file):
                results['error'] = "Failed to generate LLVM IR for original program"
                return results
                
            if not generate_llvm_ir(optimized_program, language, optimized_ir_file):
                results['error'] = "Failed to generate LLVM IR for optimized program"
                return results
            
            # If function name is not provided, try to detect the main function
            if function_name is None:
                function_name = 'main'
            
            # Extract the specified function from both IR files
            original_func_ir = extract_function_from_ir(original_ir_file, function_name)
            optimized_func_ir = extract_function_from_ir(optimized_ir_file, function_name)
            
            if original_func_ir is None:
                results['error'] = f"Function '{function_name}' not found in original program"
                return results
                
            if optimized_func_ir is None:
                results['error'] = f"Function '{function_name}' not found in optimized program"
                return results
            
            # Run Alive2 verification
            verification_results = run_alive2_verification(original_func_ir, optimized_func_ir)
            
            results['valid'] = verification_results['valid']
            results['output'] = verification_results['output']
            results['error'] = verification_results['error']
            
    except Exception as e:
        results['error'] = str(e)
        
    return results 
