"""Example demonstrating how to use the validation module to validate program optimizations."""

import os
import sys
import tempfile
from llm4opt.validate import validate_optimization, validate_functional_equivalence, validate_performance_improvement, generate_test_inputs


def main():
    """Main function to demonstrate the validation module."""
    
    # Example 1: Validating a simple C program optimization
    print("Example 1: Validating a simple C program optimization")
    print("-" * 80)
    
    # Original program: Naive matrix multiplication
    original_program = """
    #include <stdio.h>
    #include <stdlib.h>
    #include <time.h>
    
    void matrix_multiply(int* A, int* B, int* C, int N) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                int sum = 0;
                for (int k = 0; k < N; k++) {
                    sum += A[i * N + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }
    
    int main(int argc, char** argv) {
        int N = 100;
        if (argc > 1) {
            N = atoi(argv[1]);
        }
        
        int* A = (int*)malloc(N * N * sizeof(int));
        int* B = (int*)malloc(N * N * sizeof(int));
        int* C = (int*)malloc(N * N * sizeof(int));
        
        // Initialize matrices
        for (int i = 0; i < N * N; i++) {
            A[i] = 1;
            B[i] = 2;
            C[i] = 0;
        }
        
        // Perform matrix multiplication
        matrix_multiply(A, B, C, N);
        
        // Print a sample of the result
        printf("%d\\n", C[0]);
        
        free(A);
        free(B);
        free(C);
        
        return 0;
    }
    """
    
    # Optimized program: Cache-friendly matrix multiplication
    optimized_program = """
    #include <stdio.h>
    #include <stdlib.h>
    #include <time.h>
    
    void matrix_multiply(int* A, int* B, int* C, int N) {
        // Cache blocking optimization
        int BLOCK_SIZE = 16;
        
        for (int i = 0; i < N; i += BLOCK_SIZE) {
            for (int j = 0; j < N; j += BLOCK_SIZE) {
                for (int k = 0; k < N; k += BLOCK_SIZE) {
                    // Process a block
                    for (int ii = i; ii < i + BLOCK_SIZE && ii < N; ii++) {
                        for (int jj = j; jj < j + BLOCK_SIZE && jj < N; jj++) {
                            int sum = C[ii * N + jj];
                            for (int kk = k; kk < k + BLOCK_SIZE && kk < N; kk++) {
                                sum += A[ii * N + kk] * B[kk * N + jj];
                            }
                            C[ii * N + jj] = sum;
                        }
                    }
                }
            }
        }
    }
    
    int main(int argc, char** argv) {
        int N = 100;
        if (argc > 1) {
            N = atoi(argv[1]);
        }
        
        int* A = (int*)malloc(N * N * sizeof(int));
        int* B = (int*)malloc(N * N * sizeof(int));
        int* C = (int*)malloc(N * N * sizeof(int));
        
        // Initialize matrices
        for (int i = 0; i < N * N; i++) {
            A[i] = 1;
            B[i] = 2;
            C[i] = 0;
        }
        
        // Perform matrix multiplication
        matrix_multiply(A, B, C, N);
        
        // Print a sample of the result
        printf("%d\\n", C[0]);
        
        free(A);
        free(B);
        free(C);
        
        return 0;
    }
    """
    
    # Validate the optimization
    results = validate_optimization(
        original_program=original_program,
        optimized_program=optimized_program,
        language='c',
        performance_threshold=1.1,  # Require at least 10% improvement
        measure_size=True,
        size_weight=0.3,
        time_weight=0.7,
        num_test_inputs=5,  # Generate 5 test inputs
        test_generation_method='random'  # Use random test input generation
    )
    
    print("Validation results:")
    print(f"  Is valid optimization: {results['is_valid']}")
    print(f"  Functional equivalence: {results['functional_equivalence']}")
    print(f"  Performance improvement: {results['performance_improvement']:.2f}x")
    
    # Print code size metrics if available
    if 'size_reduction' in results:
        print(f"  Size reduction: {results['size_reduction']:.2f}x")
        print(f"  Original binary size: {results['original_size']} bytes")
        print(f"  Optimized binary size: {results['optimized_size']} bytes")
        
        if 'text_size_reduction' in results:
            print(f"  Text section size reduction: {results['text_size_reduction']:.2f}x")
            print(f"  Original text section size: {results['original_text_size']} bytes")
            print(f"  Optimized text section size: {results['optimized_text_size']} bytes")
    
    print(f"  Overall improvement: {results.get('overall_improvement', results['performance_improvement']):.2f}x")
    
    if results['error']:
        print(f"  Error: {results['error']}")
    print()
    
    # Example 2: Validating a Python program optimization
    print("Example 2: Validating a Python program optimization")
    print("-" * 80)
    
    # Original program: Naive Python function
    original_python = """
import sys
import time

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def main():
    n = 20
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    
    result = fibonacci(n)
    print(result)

if __name__ == "__main__":
    main()
    """
    
    # Optimized program: Memoized Python function
    optimized_python = """
import sys
import time

def fibonacci(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci(n-1, memo) + fibonacci(n-2, memo)
    return memo[n]

def main():
    n = 20
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    
    result = fibonacci(n)
    print(result)

if __name__ == "__main__":
    main()
    """
    
    # Validate the optimization
    results = validate_optimization(
        original_program=original_python,
        optimized_program=optimized_python,
        language='python',
        performance_threshold=1.5,  # Require at least 50% improvement
        num_test_inputs=3,  # Generate 3 test inputs
        test_generation_method='random',  # Use random test input generation
        input_type='int',  # Generate integer inputs
        min_val=5,  # Minimum value for inputs
        max_val=25  # Maximum value for inputs
    )
    
    print("Validation results:")
    print(f"  Is valid optimization: {results['is_valid']}")
    print(f"  Functional equivalence: {results['functional_equivalence']}")
    print(f"  Performance improvement: {results['performance_improvement']:.2f}x")
    print(f"  Overall improvement: {results.get('overall_improvement', results['performance_improvement']):.2f}x")
    print(f"  Test inputs used: {results['test_inputs_used']}")
    if results['error']:
        print(f"  Error: {results['error']}")
    print()
    
    # Example 3: Validating just functional equivalence
    print("Example 3: Validating just functional equivalence")
    print("-" * 80)
    
    # Original program: Simple sorting function
    original_sort = """
#include <stdio.h>
#include <stdlib.h>

void bubble_sort(int arr[], int n) {
    for (int i = 0; i < n-1; i++) {
        for (int j = 0; j < n-i-1; j++) {
            if (arr[j] > arr[j+1]) {
                // Swap
                int temp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = temp;
            }
        }
    }
}

int main(int argc, char** argv) {
    int arr[] = {64, 34, 25, 12, 22, 11, 90};
    int n = sizeof(arr)/sizeof(arr[0]);
    
    bubble_sort(arr, n);
    
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\\n");
    
    return 0;
}
    """
    
    # Optimized program: Quick sort
    optimized_sort = """
#include <stdio.h>
#include <stdlib.h>

void swap(int* a, int* b) {
    int t = *a;
    *a = *b;
    *b = t;
}

int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = (low - 1);
    
    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

void quick_sort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quick_sort(arr, low, pi - 1);
        quick_sort(arr, pi + 1, high);
    }
}

int main(int argc, char** argv) {
    int arr[] = {64, 34, 25, 12, 22, 11, 90};
    int n = sizeof(arr)/sizeof(arr[0]);
    
    quick_sort(arr, 0, n-1);
    
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\\n");
    
    return 0;
}
    """
    
    # Generate random test inputs
    random_inputs = generate_test_inputs(
        program=original_sort,
        language='c',
        num_inputs=5,
        generation_method='random'
    )
    print(f"Generated {len(random_inputs)} random test inputs")
    
    # Validate just functional equivalence
    results = validate_functional_equivalence(
        original_program=original_sort,
        optimized_program=optimized_sort,
        language='c',
        test_inputs=random_inputs,
        num_test_inputs=5,
        test_generation_method='random'
    )
    
    print("Functional equivalence results:")
    print(f"  Equivalent: {results['equivalent']}")
    if results['mismatched_inputs']:
        print(f"  Mismatched inputs: {results['mismatched_inputs']}")
    if results['error']:
        print(f"  Error: {results['error']}")
    print()
    
    # Example 4: Comparing code size for different optimization levels
    print("Example 4: Comparing code size for different optimization levels")
    print("-" * 80)
    
    # Simple program with loop unrolling
    original_loop = """
#include <stdio.h>

int main() {
    int sum = 0;
    for (int i = 0; i < 100; i++) {
        sum += i;
    }
    printf("Sum: %d\\n", sum);
    return 0;
}
    """
    
    # Optimized with manual loop unrolling
    unrolled_loop = """
#include <stdio.h>

int main() {
    int sum = 0;
    for (int i = 0; i < 100; i += 4) {
        sum += i;
        if (i + 1 < 100) sum += (i + 1);
        if (i + 2 < 100) sum += (i + 2);
        if (i + 3 < 100) sum += (i + 3);
    }
    printf("Sum: %d\\n", sum);
    return 0;
}
    """
    
    # Validate with focus on code size
    results = validate_performance_improvement(
        original_program=original_loop,
        optimized_program=unrolled_loop,
        language='c',
        measure_size=True,
        size_weight=0.7,  # Give more weight to size
        time_weight=0.3   # Give less weight to time
    )
    
    print("Performance and size comparison:")
    print(f"  Speedup: {results['speedup']:.2f}x")
    
    if 'size_reduction' in results:
        print(f"  Size reduction: {results['size_reduction']:.2f}x")
        print(f"  Original binary size: {results['original_size']} bytes")
        print(f"  Optimized binary size: {results['optimized_size']} bytes")
        
        if 'text_size_reduction' in results:
            print(f"  Text section size reduction: {results['text_size_reduction']:.2f}x")
            print(f"  Original text section size: {results['original_text_size']} bytes")
            print(f"  Optimized text section size: {results['optimized_text_size']} bytes")
    
    print(f"  Overall improvement (weighted): {results.get('overall_improvement', results['speedup']):.2f}x")
    
    if results['error']:
        print(f"  Error: {results['error']}")
    print()
    
    # Example 5: Using LLM-generated test inputs
    print("Example 5: Using LLM-generated test inputs (if available)")
    print("-" * 80)
    
    # Simple calculator program
    calculator_program = """
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Simple calculator that supports +, -, *, /
int main(int argc, char** argv) {
    if (argc != 4) {
        printf("Usage: %s <number> <operator> <number>\\n", argv[0]);
        return 1;
    }
    
    double a = atof(argv[1]);
    char op = argv[2][0];
    double b = atof(argv[3]);
    double result = 0;
    
    switch (op) {
        case '+':
            result = a + b;
            break;
        case '-':
            result = a - b;
            break;
        case '*':
            result = a * b;
            break;
        case '/':
            if (b == 0) {
                printf("Error: Division by zero\\n");
                return 1;
            }
            result = a / b;
            break;
        default:
            printf("Error: Invalid operator '%c'\\n", op);
            return 1;
    }
    
    printf("%.2f %c %.2f = %.2f\\n", a, op, b, result);
    return 0;
}
    """
    
    # Try to generate test inputs using an LLM
    try:
        # Try to generate inputs using an LLM
        llm_inputs = generate_test_inputs(
            program=calculator_program,
            language='c',
            num_inputs=5,
            generation_method='llm',
            llm_model='gpt-4',  # Specify the LLM model to use
            temperature=0.7     # Higher temperature for more diverse inputs
        )
        print("Generated test inputs using LLM:")
        for i, input_data in enumerate(llm_inputs):
            print(f"  Input {i+1}: {input_data}")
    except Exception as e:
        print(f"LLM input generation failed: {e}")
        print("Falling back to random inputs")
        llm_inputs = generate_test_inputs(
            program=calculator_program,
            language='c',
            num_inputs=5,
            generation_method='random'
        )
        print("Generated random test inputs:")
        for i, input_data in enumerate(llm_inputs):
            print(f"  Input {i+1}: {input_data}")


if __name__ == "__main__":
    main() 