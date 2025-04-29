# LLM4OPT: LLM-guided C/C++ Code Optimization

Optimizing C/C++ code using Large Language Models (LLMs). This project leverages LLM capabilities to perform source-level optimizations and guide the LLVM/Clang compiler through optimization decisions.

## Features

**1. Compiler hint generation (__restrict__, __no_alias__, etc.)**

For some cases, the "hins" are already "compiler optimizations", such as annotations in OpenMP.

Maybe we need to clarify the scope more clear.

**2. Optimization passes selection**

Select the optimizaion passes for certain goals. See the `llm4opt/passes` module for details on how to select and order compiler optimization passes for specific goals (code size, performance, vectorization).

You can run experiments using the `run_experiment.py` script:

```bash
python run_experiment.py --source llm4opt/passes/sample_benchmark.c --goal performance --strategy evolutionary
```

**3. Compiler-friendly code transformations**

Refactor the code so that the compiler can do better.

**3. Fine-tuning LLM**


**5. Validation of optimizations**
 
- **Validation via formal verification** using Alive2 for C/C++ optimizations
- **Validation via testing** using random and LLM-based methods

**5. Evaluation** 

## Usages

Install
~~~~

~~~~

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
