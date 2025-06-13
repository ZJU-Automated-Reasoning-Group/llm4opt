# LLM4OPT: LLM-guided C/C++ Code Optimization

Optimizing C/C++ code using Large Language Models (LLMs). This project leverages LLM capabilities to perform source-level optimizations and guide the LLVM/Clang compiler through optimization decisions.

## Compiler Optimizations

**1. Compiler Hints Generation**

- clang/gcc: Attributes: __restrict__, __no_alias__, etc.
- OpenMP

**2. Optimization Passes Selection**

Select the optimizaion passes for certain goals. See the `llm4opt/passes` module for details on how to select and order compiler optimization passes for specific goals (code size, performance, vectorization).



## Validation of Optimizations
 
- **Validation via formal verification** using Alive2 for C/C++ optimizations
- **Validation via testing** using random and LLM-based methods


## Requirements

- Python 3.9+
- LLVM/Clang
- LLM: GPU, Claude, DeepSeek, etc.
- LLM offline
- For validation: 
  - GCC or Clang compiler (for C/C++ validation)
  - GNU binutils (for code size measurement)
  - Alive2 (for formal verification)


## Related Work

- [ECCO](https://github.com/CodeEff/ECCO/tree/main) - Compiler optimization using deep learning
- MLGO: a Machine Learning Guided Compiler Optimizations Framework.
https://github.com/google/ml-compiler-opt
