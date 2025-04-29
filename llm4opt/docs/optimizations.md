# Introduction to Compiler Optimizations


## Compiler Optimizations

By goal
- Improving performance
- Improving code size
- ...

By strategies
- Pre-compute: constant folding, etc.
- Low-cost operations: algebraic simplification, strength reduction, etc.
- Data locality: cache blocking, etc.
- Redundancy elimination: common subexpression elimination, partial redundancy elimination, global value numbering, etc.
- Cache optimizations: prefetching, etc.
- Code Motion: loop unrolling, etc.
- Parallelization: vectorization, pipeline, etc.
- Reduce call overhead: inline, tail call elimination, etc.
...

By scope
- Whole program
- Function level ("Global")
- Loop level
- Statement level
- ...

By stage
- Frontend: source-level, AST-level, etc.
- Middleend: IR-based optimizations
- Backend: LTO?, instruction selection, instruction scheduling, etc.
- Profiling-based optimizations

## Our Goal

See README.md