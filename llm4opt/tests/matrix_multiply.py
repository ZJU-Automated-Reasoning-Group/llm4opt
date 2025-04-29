"""Example demonstrating how to use compiler hints to optimize matrix multiplication in C++."""

import os
import sys
import tempfile
import subprocess
from llm4opt.hints import (
    restrict, aligned, vectorize, parallel, unroll, loop_vectorize,
    hot, pure, inline, prefetch_locality, independent_iterations
)


def generate_optimized_matrix_multiply():
    """Generate an optimized matrix multiplication implementation using compiler hints."""
    
    # This is a template for a naive matrix multiplication in C++
    naive_cpp_code = """
    #include <vector>
    #include <chrono>
    #include <iostream>
    
    // Naive matrix multiplication implementation
    void matrix_multiply_naive(const double* A, const double* B, double* C, int N) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                double sum = 0.0;
                for (int k = 0; k < N; ++k) {
                    sum += A[i * N + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }
    
    int main() {
        const int N = 1000;
        std::vector<double> A(N * N, 1.0);
        std::vector<double> B(N * N, 2.0);
        std::vector<double> C(N * N, 0.0);
        
        auto start = std::chrono::high_resolution_clock::now();
        matrix_multiply_naive(A.data(), B.data(), C.data(), N);
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Naive implementation took " << elapsed.count() << " seconds\\n";
        
        return 0;
    }
    """
    
    # Now let's apply our compiler hints to optimize the matrix multiplication
    # In a real-world scenario, these hints would be applied by a compiler pass
    # Here we're manually translating them to C++ attributes and pragmas
    
    optimized_cpp_code = """
    #include <vector>
    #include <chrono>
    #include <iostream>
    #include <immintrin.h>  // For SIMD intrinsics
    
    // Naive matrix multiplication implementation for comparison
    void matrix_multiply_naive(const double* A, const double* B, double* C, int N) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                double sum = 0.0;
                for (int k = 0; k < N; ++k) {
                    sum += A[i * N + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }
    
    // Optimized matrix multiplication with compiler hints
    // Equivalent to using the following Python hints:
    // @hot
    // @pure
    // def matrix_multiply_optimized(A, B, C, N):
    //     with parallel(num_threads=4):
    //         for i in range(N):
    //             for j in range(N):
    //                 with loop_vectorize():
    //                     for k in range(N):
    //                         C[i*N + j] += A[i*N + k] * B[k*N + j]
    __attribute__((hot)) __attribute__((pure))
    void matrix_multiply_optimized(const double* __restrict A, 
                                  const double* __restrict B, 
                                  double* __restrict C, int N) {
        #pragma omp parallel for num_threads(4)
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                double sum = 0.0;
                #pragma GCC ivdep
                #pragma GCC vector_size(32)
                for (int k = 0; k < N; ++k) {
                    sum += A[i * N + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }
    
    // Cache-friendly tiled matrix multiplication
    // Equivalent to using the following Python hints:
    // @hot
    // @inline
    // def matrix_multiply_tiled(A, B, C, N, TILE_SIZE):
    //     with parallel(num_threads=4):
    //         for i in range(0, N, TILE_SIZE):
    //             for j in range(0, N, TILE_SIZE):
    //                 for k in range(0, N, TILE_SIZE):
    //                     with prefetch_locality(3):
    //                         for ii in range(i, min(i+TILE_SIZE, N)):
    //                             for jj in range(j, min(j+TILE_SIZE, N)):
    //                                 with loop_vectorize():
    //                                     for kk in range(k, min(k+TILE_SIZE, N)):
    //                                         C[ii*N + jj] += A[ii*N + kk] * B[kk*N + jj]
    __attribute__((hot)) inline
    void matrix_multiply_tiled(const double* __restrict A, 
                              const double* __restrict B, 
                              double* __restrict C, int N, int TILE_SIZE) {
        #pragma omp parallel for num_threads(4)
        for (int i = 0; i < N; i += TILE_SIZE) {
            for (int j = 0; j < N; j += TILE_SIZE) {
                for (int k = 0; k < N; k += TILE_SIZE) {
                    // Prefetch the tiles
                    __builtin_prefetch(&A[i * N + k], 0, 3);
                    __builtin_prefetch(&B[k * N + j], 0, 3);
                    
                    for (int ii = i; ii < std::min(i + TILE_SIZE, N); ++ii) {
                        for (int jj = j; jj < std::min(j + TILE_SIZE, N); ++jj) {
                            double sum = C[ii * N + jj];
                            #pragma GCC ivdep
                            #pragma GCC vector_size(32)
                            for (int kk = k; kk < std::min(k + TILE_SIZE, N); ++kk) {
                                sum += A[ii * N + kk] * B[kk * N + jj];
                            }
                            C[ii * N + jj] = sum;
                        }
                    }
                }
            }
        }
    }
    
    int main() {
        const int N = 1000;
        const int TILE_SIZE = 32;  // Adjust based on cache size
        
        // Aligned memory allocation for better performance
        // Equivalent to using the @aligned(64) hint in Python
        double* A = (double*)aligned_alloc(64, N * N * sizeof(double));
        double* B = (double*)aligned_alloc(64, N * N * sizeof(double));
        double* C1 = (double*)aligned_alloc(64, N * N * sizeof(double));
        double* C2 = (double*)aligned_alloc(64, N * N * sizeof(double));
        double* C3 = (double*)aligned_alloc(64, N * N * sizeof(double));
        
        // Initialize matrices
        for (int i = 0; i < N * N; ++i) {
            A[i] = 1.0;
            B[i] = 2.0;
            C1[i] = 0.0;
            C2[i] = 0.0;
            C3[i] = 0.0;
        }
        
        // Test naive implementation
        auto start1 = std::chrono::high_resolution_clock::now();
        matrix_multiply_naive(A, B, C1, N);
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed1 = end1 - start1;
        
        // Test optimized implementation
        auto start2 = std::chrono::high_resolution_clock::now();
        matrix_multiply_optimized(A, B, C2, N);
        auto end2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed2 = end2 - start2;
        
        // Test tiled implementation
        auto start3 = std::chrono::high_resolution_clock::now();
        matrix_multiply_tiled(A, B, C3, N, TILE_SIZE);
        auto end3 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed3 = end3 - start3;
        
        // Print results
        std::cout << "Naive implementation took " << elapsed1.count() << " seconds\\n";
        std::cout << "Optimized implementation took " << elapsed2.count() << " seconds\\n";
        std::cout << "Tiled implementation took " << elapsed3.count() << " seconds\\n";
        
        // Verify results
        bool correct = true;
        for (int i = 0; i < N * N; ++i) {
            if (std::abs(C1[i] - C2[i]) > 1e-6 || std::abs(C1[i] - C3[i]) > 1e-6) {
                correct = false;
                break;
            }
        }
        std::cout << "Results are " << (correct ? "correct" : "incorrect") << "\\n";
        
        // Free memory
        free(A);
        free(B);
        free(C1);
        free(C2);
        free(C3);
        
        return 0;
    }
    """
    
    return optimized_cpp_code


def compile_and_run_example():
    """Compile and run the optimized matrix multiplication example."""
    cpp_code = generate_optimized_matrix_multiply()
    
    # Create a temporary file for the C++ code
    with tempfile.NamedTemporaryFile(suffix='.cpp', delete=False) as f:
        f.write(cpp_code.encode('utf-8'))
        cpp_file = f.name
    
    # Compile the C++ code with optimization flags
    output_file = cpp_file.replace('.cpp', '')
    compile_cmd = [
        'g++', cpp_file, '-o', output_file,
        '-O3', '-march=native', '-fopenmp', '-std=c++17'
    ]
    
    try:
        subprocess.run(compile_cmd, check=True)
        print(f"Successfully compiled {cpp_file}")
        
        # Run the compiled program
        run_cmd = [output_file]
        subprocess.run(run_cmd, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
    finally:
        # Clean up temporary files
        if os.path.exists(cpp_file):
            os.remove(cpp_file)
        if os.path.exists(output_file):
            os.remove(output_file)


def main():
    """Main function to demonstrate the example."""
    print("Generating optimized matrix multiplication code...")
    cpp_code = generate_optimized_matrix_multiply()
    print("\nGenerated C++ code:")
    print("-" * 80)
    print(cpp_code)
    print("-" * 80)
    
    # Optionally compile and run if the user has the necessary compiler
    if len(sys.argv) > 1 and sys.argv[1] == '--run':
        print("\nCompiling and running the example...")
        compile_and_run_example()
    else:
        print("\nTo compile and run the example, use: python matrix_multiply.py --run")


if __name__ == "__main__":
    main() 