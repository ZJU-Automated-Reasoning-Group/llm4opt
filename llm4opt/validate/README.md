# Validation

## Testing

## Alive2


## Usage (TBD)

```python
from llm4opt.validate import validate_optimization

# Validate an optimization
results = validate_optimization(
    original_program="""
    #include <stdio.h>
    
    // Original C program
    int main() {
        int sum = 0;
        for (int i = 0; i < 1000; i++) {
            sum += i;
        }
        printf("Sum: %d\\n", sum);
        return 0;
    }
    """,
    optimized_program="""
    #include <stdio.h>
    
    // Optimized C program using closed-form formula
    int main() {
        int n = 999;  // Last number in the sum
        int sum = n * (n + 1) / 2;
        printf("Sum: %d\\n", sum);
        return 0;
    }
    """,
    language="c",
    performance_threshold=1.1,  # Require at least 10% improvement
    measure_size=True,  # Measure code size
    size_weight=0.3,  # Weight for size in overall score
    time_weight=0.7,  # Weight for time in overall score
    num_test_inputs=10,  # Number of test inputs to generate
    test_generation_method="random"  # Method to generate test inputs
)

# Check the results
print(f"Is valid optimization: {results['is_valid']}")
print(f"Functional equivalence: {results['functional_equivalence']}")
print(f"Performance improvement: {results['performance_improvement']}x")
print(f"Size reduction: {results['size_reduction']}x")
print(f"Overall improvement: {results['overall_improvement']}x")

# For C/C++ programs, you can also use Alive2 for formal verification
if results['alive2_valid'] is not None:
    print(f"Formally verified: {results['alive2_valid']}")
```
