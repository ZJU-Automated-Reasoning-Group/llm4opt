"""
Validate the functional equivalence of two programs using test cases.

"""

import subprocess
import tempfile
import os
import random
import hashlib
import time
import json
import re
from typing import List, Dict, Any, Optional, Union, Callable
from pathlib import Path

# Import LLM modules if available
try:
    from ..llm import get_llm_client
except ImportError:
    def get_llm_client(*args, **kwargs):
        raise ImportError("LLM module not available. Please install the required dependencies.")


def generate_random_inputs(num_inputs: int = 10, input_type: str = 'int', 
                          min_val: int = -1000, max_val: int = 1000,
                          min_len: int = 1, max_len: int = 100) -> List[Any]:
    """
    Generate random inputs for testing program equivalence.
    
    Args:
        num_inputs: Number of random inputs to generate
        input_type: Type of inputs ('int', 'float', 'str', 'list', 'dict')
        min_val: Minimum value for numeric inputs
        max_val: Maximum value for numeric inputs
        min_len: Minimum length for container inputs
        max_len: Maximum length for container inputs
        
    Returns:
        List of randomly generated inputs
    """
    inputs = []
    
    for _ in range(num_inputs):
        if input_type == 'int':
            inputs.append(random.randint(min_val, max_val))
        elif input_type == 'float':
            inputs.append(random.uniform(min_val, max_val))
        elif input_type == 'str':
            length = random.randint(min_len, max_len)
            inputs.append(''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(length)))
        elif input_type == 'list':
            length = random.randint(min_len, max_len)
            inputs.append([random.randint(min_val, max_val) for _ in range(length)])
        elif input_type == 'dict':
            length = random.randint(min_len, max_len)
            inputs.append({f'key{i}': random.randint(min_val, max_val) for i in range(length)})
    
    return inputs


def generate_llm_inputs(
    program: str,
    language: str,
    num_inputs: int = 10,
    llm_model: str = "gpt-4",
    temperature: float = 0.7
) -> List[Any]:
    """
    Generate test inputs using a Large Language Model.
    
    Args:
        program: Source code of the program
        language: Programming language ('c', 'cpp', 'python')
        num_inputs: Number of test inputs to generate
        llm_model: Name of the LLM model to use
        temperature: Temperature parameter for the LLM (higher = more creative)
        
    Returns:
        List of generated inputs
    """
    inputs = []
    
    try:
        # Get LLM client
        llm_client = get_llm_client(model=llm_model)
        
        # Create a prompt for the LLM
        prompt = f"""
You are an expert in software testing. Given the following {language} program, 
please generate {num_inputs} diverse and effective test inputs that would thoroughly 
test the program's functionality and edge cases.

Program:
```{language}
{program}
```

Please provide the test inputs in JSON format as a list. Each test input should be a string 
that could be passed to the program. Focus on inputs that might reveal bugs or edge cases.

Example response format:
```json
[
  "input1",
  "input2",
  "42",
  "-1",
  "very long string input...",
  ...
]
```
"""
        
        # Call the LLM
        response = llm_client.generate(prompt, temperature=temperature)
        
        # Extract the JSON from the response
        json_pattern = r'```json\s*(.*?)\s*```'
        json_match = re.search(json_pattern, response, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1)
            inputs = json.loads(json_str)
        else:
            # Try to find any JSON array in the response
            array_pattern = r'\[\s*".*?"\s*(,\s*".*?"\s*)*\]'
            array_match = re.search(array_pattern, response, re.DOTALL)
            
            if array_match:
                json_str = array_match.group(0)
                inputs = json.loads(json_str)
            else:
                # If we can't find JSON, extract strings from the response
                string_pattern = r'"([^"]*)"'
                string_matches = re.findall(string_pattern, response)
                inputs = string_matches
        
        # Ensure we have the requested number of inputs
        if len(inputs) < num_inputs:
            additional_inputs = generate_random_inputs(num_inputs - len(inputs))
            inputs.extend(additional_inputs)
        elif len(inputs) > num_inputs:
            inputs = inputs[:num_inputs]
            
    except Exception as e:
        print(f"LLM input generation failed: {e}")
        # Fall back to random inputs
        inputs = generate_random_inputs(num_inputs)
    
    return inputs


def generate_test_inputs(
    program: str,
    language: str,
    num_inputs: int = 10,
    generation_method: str = 'random',
    **kwargs
) -> List[Any]:
    """
    Generate test inputs using the specified method.
    
    Args:
        program: Source code of the program
        language: Programming language ('c', 'cpp', 'python')
        num_inputs: Number of test inputs to generate
        generation_method: Method to use for generating inputs ('random', 'llm')
        **kwargs: Additional arguments for the specific generation method
        
    Returns:
        List of generated inputs
    """
    if generation_method == 'random':
        return generate_random_inputs(
            num_inputs=num_inputs,
            input_type=kwargs.get('input_type', 'int'),
            min_val=kwargs.get('min_val', -1000),
            max_val=kwargs.get('max_val', 1000),
            min_len=kwargs.get('min_len', 1),
            max_len=kwargs.get('max_len', 100)
        )
    elif generation_method == 'llm':
        return generate_llm_inputs(
            program=program,
            language=language,
            num_inputs=num_inputs,
            llm_model=kwargs.get('llm_model', 'gpt-4'),
            temperature=kwargs.get('temperature', 0.7)
        )
    else:
        raise ValueError(f"Unknown test input generation method: {generation_method}")


def compile_program(source_code: str, language: str, output_file: str) -> bool:
    """
    Compile a program from source code.
    
    Args:
        source_code: Source code of the program
        language: Programming language ('c', 'cpp')
        output_file: Path to the output executable
        
    Returns:
        True if compilation succeeded, False otherwise
    """
    if language not in ('c', 'cpp'):
        raise ValueError(f"Unsupported language for compilation: {language}")
    
    # Create a temporary file for the source code
    with tempfile.NamedTemporaryFile(suffix=f'.{language}', delete=False) as f:
        f.write(source_code.encode('utf-8'))
        source_file = f.name
    
    try:
        # Compile the source code
        compiler = 'gcc' if language == 'c' else 'g++'
        cmd = [compiler, source_file, '-o', output_file, '-O0']  # Use -O0 to disable optimizations
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Compilation error: {result.stderr}")
            return False
            
        return True
    finally:
        # Clean up the temporary source file
        if os.path.exists(source_file):
            os.remove(source_file)


def run_program(executable: str, input_data: Any, timeout: int = 30) -> Dict[str, Any]:
    """
    Run a compiled program with the given input.
    
    Args:
        executable: Path to the executable
        input_data: Input data for the program
        timeout: Maximum execution time in seconds
        
    Returns:
        Dict containing execution results:
        {
            'success': bool,
            'output': str,
            'error': str,
            'execution_time': float
        }
    """
    result = {
        'success': False,
        'output': '',
        'error': '',
        'execution_time': 0.0
    }
    
    # Convert input data to string
    input_str = str(input_data)
    
    try:
        start_time = time.time()
        
        # Run the program with the input
        process = subprocess.run(
            [executable],
            input=input_str.encode('utf-8'),
            capture_output=True,
            timeout=timeout
        )
        
        end_time = time.time()
        result['execution_time'] = end_time - start_time
        
        if process.returncode == 0:
            result['success'] = True
            result['output'] = process.stdout.decode('utf-8').strip()
        else:
            result['error'] = process.stderr.decode('utf-8').strip()
            
    except subprocess.TimeoutExpired:
        result['error'] = f"Execution timed out after {timeout} seconds"
    except Exception as e:
        result['error'] = str(e)
        
    return result


def run_python_program(source_code: str, input_data: Any, timeout: int = 30) -> Dict[str, Any]:
    """
    Run a Python program with the given input.
    
    Args:
        source_code: Source code of the Python program
        input_data: Input data for the program
        timeout: Maximum execution time in seconds
        
    Returns:
        Dict containing execution results:
        {
            'success': bool,
            'output': str,
            'error': str,
            'execution_time': float
        }
    """
    result = {
        'success': False,
        'output': '',
        'error': '',
        'execution_time': 0.0
    }
    
    # Create a temporary file for the Python code
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as f:
        f.write(source_code.encode('utf-8'))
        py_file = f.name
    
    try:
        # Convert input data to string
        input_str = str(input_data)
        
        start_time = time.time()
        
        # Run the Python program with the input
        process = subprocess.run(
            ['python', py_file],
            input=input_str.encode('utf-8'),
            capture_output=True,
            timeout=timeout
        )
        
        end_time = time.time()
        result['execution_time'] = end_time - start_time
        
        if process.returncode == 0:
            result['success'] = True
            result['output'] = process.stdout.decode('utf-8').strip()
        else:
            result['error'] = process.stderr.decode('utf-8').strip()
            
    except subprocess.TimeoutExpired:
        result['error'] = f"Execution timed out after {timeout} seconds"
    except Exception as e:
        result['error'] = str(e)
    finally:
        # Clean up the temporary Python file
        if os.path.exists(py_file):
            os.remove(py_file)
        
    return result


def validate_functional_equivalence(
    original_program: str,
    optimized_program: str,
    language: str = 'c',
    test_inputs: Optional[List[Any]] = None,
    timeout: int = 60,
    num_test_inputs: int = 10,
    test_generation_method: str = 'random',
    **test_generation_kwargs
) -> Dict[str, Any]:
    """
    Validate if two programs are functionally equivalent by comparing their outputs
    for the same set of inputs.
    
    Args:
        original_program: Source code of the original program
        optimized_program: Source code of the optimized program
        language: Programming language ('c', 'cpp', 'python')
        test_inputs: List of test inputs to validate functional equivalence
        timeout: Maximum time in seconds to run each program
        num_test_inputs: Number of test inputs to generate if none provided
        test_generation_method: Method to use for generating inputs ('random', 'llm')
        **test_generation_kwargs: Additional arguments for the test input generation method
        
    Returns:
        Dict containing validation results:
        {
            'equivalent': bool,
            'mismatched_inputs': List[Any],
            'error': Optional[str]
        }
    """
    results = {
        'equivalent': False,
        'mismatched_inputs': [],
        'error': None
    }
    
    # Generate test inputs if none provided
    if test_inputs is None or len(test_inputs) == 0:
        test_inputs = generate_test_inputs(
            program=original_program,
            language=language,
            num_inputs=num_test_inputs,
            generation_method=test_generation_method,
            **test_generation_kwargs
        )
    
    try:
        if language in ('c', 'cpp'):
            # Compile both programs
            with tempfile.TemporaryDirectory() as temp_dir:
                original_exe = os.path.join(temp_dir, 'original')
                optimized_exe = os.path.join(temp_dir, 'optimized')
                
                if not compile_program(original_program, language, original_exe):
                    results['error'] = "Failed to compile original program"
                    return results
                    
                if not compile_program(optimized_program, language, optimized_exe):
                    results['error'] = "Failed to compile optimized program"
                    return results
                
                # Run both programs with each test input and compare outputs
                for input_data in test_inputs:
                    original_result = run_program(original_exe, input_data, timeout)
                    optimized_result = run_program(optimized_exe, input_data, timeout)
                    
                    if not original_result['success']:
                        results['error'] = f"Original program failed: {original_result['error']}"
                        results['mismatched_inputs'].append(input_data)
                        continue
                        
                    if not optimized_result['success']:
                        results['error'] = f"Optimized program failed: {optimized_result['error']}"
                        results['mismatched_inputs'].append(input_data)
                        continue
                        
                    if original_result['output'] != optimized_result['output']:
                        results['mismatched_inputs'].append(input_data)
        
        elif language == 'python':
            # Run both Python programs with each test input and compare outputs
            for input_data in test_inputs:
                original_result = run_python_program(original_program, input_data, timeout)
                optimized_result = run_python_program(optimized_program, input_data, timeout)
                
                if not original_result['success']:
                    results['error'] = f"Original program failed: {original_result['error']}"
                    results['mismatched_inputs'].append(input_data)
                    continue
                    
                if not optimized_result['success']:
                    results['error'] = f"Optimized program failed: {optimized_result['error']}"
                    results['mismatched_inputs'].append(input_data)
                    continue
                    
                if original_result['output'] != optimized_result['output']:
                    results['mismatched_inputs'].append(input_data)
        
        else:
            results['error'] = f"Unsupported language: {language}"
            return results
            
        # If no mismatched inputs, the programs are functionally equivalent
        results['equivalent'] = len(results['mismatched_inputs']) == 0
        
    except Exception as e:
        results['error'] = str(e)
        
    return results 