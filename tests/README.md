# cTensor Test Suite

This directory contains a comprehensive test suite for the cTensor project, ensuring determinism between cTensor and PyTorch implementations.

## Overview

The test suite validates that cTensor operators produce identical results to PyTorch for the same inputs, ensuring mathematical correctness and determinism across platforms.

## Structure

```
tests/
├── test_framework.h/c      # Core test framework with assertion macros
├── test_operators_implemented.c  # Tests for currently implemented operators
├── test_main.c             # Test runner entry point
├── generate_reference_data.py     # PyTorch reference data generator
├── reference_data/         # JSON files with PyTorch reference results
├── CMakeLists.txt          # Build configuration
└── README.md               # This file
```

## Building and Running Tests

### Prerequisites
- CMake 3.10+
- C11 compatible compiler
- Python 3.7+ with PyTorch (for generating reference data)

### Build Tests
```bash
cd tests
mkdir -p build
cd build
cmake ..
make
```

### Run Tests
```bash
./cten_tests
```

### Generate New Reference Data
```bash
python generate_reference_data.py
```

## Current Test Coverage

The test suite currently covers all implemented operators:

- ✅ **Tensor_add** - Element-wise addition
- ✅ **Tensor_mul** - Element-wise multiplication  
- ✅ **Tensor_mulf** - Scalar multiplication
- ✅ **Tensor_sub** - Element-wise subtraction
- ✅ **Tensor_mean** - Mean reduction
- ✅ **Tensor_sum** - Sum reduction
- ✅ **Tensor_matmul** - Matrix multiplication

## Adding New Tests

To add tests for new operators:

1. **Add reference data generation** in `generate_reference_data.py`:
   ```python
   # Add your operator test case
   result = torch.your_operator(input_tensor)
   test_data["your_operator"] = {
       "input": input_tensor.tolist(),
       "expected": result.tolist()
   }
   ```

2. **Implement test function** in `test_operators_implemented.c`:
   ```c
   bool test_your_operator(void) {
       printf("Testing Your_operator...\n");
       
       // Load reference data
       // Create input tensor
       // Call cTensor operator
       // Compare with expected result
       
       return true;  // or false if test fails
   }
   ```

3. **Add test to runner** in `run_implemented_operator_tests()`:
   ```c
   RUN_TEST(test_your_operator);
   ```

4. **Regenerate reference data**:
   ```bash
   python generate_reference_data.py
   ```

5. **Build and run tests**:
   ```bash
   cmake --build build
   ./build/cten_tests
   ```

## Test Framework Features

### Assertion Macros
- `ASSERT_TRUE(condition)` - Assert condition is true
- `ASSERT_FALSE(condition)` - Assert condition is false  
- `ASSERT_FLOAT_EQ(a, b, epsilon)` - Assert floats are equal within epsilon
- `ASSERT_TENSOR_EQ(tensor, expected_data, size, epsilon)` - Assert tensor matches expected data

### Utilities
- `create_tensor_from_data()` - Create tensor from float array
- `compare_tensors()` - Compare two tensors for equality
- `print_tensor_debug()` - Debug print tensor contents

### Memory Management
The test framework integrates with cTensor's pool allocator system:
- Automatically initializes and cleans up memory pools
- Tracks test statistics and memory usage
- Provides detailed error reporting

## CI/CD Integration

Tests run automatically on:
- **Push/PR** to main branch
- **Multiple platforms**: Ubuntu, macOS, Windows
- **Dependency management**: Automatic PyTorch and CMake installation

## Expected Output

```
=== cTensor Test Framework ===
Testing determinism against PyTorch reference data

=== Running Implemented Operator Tests ===
Running test_tensor_add...
Testing Tensor_add...
PASS: test_tensor_add

[... more tests ...]

=== Test Summary ===
Total tests: 7
Passed: 7
Failed: 0
All tests PASSED! ✓
Success rate: 100.0%
```

## Troubleshooting

### Build Issues
- Ensure CMake can find the cTensor source directory
- Check that all dependencies are installed
- Verify C11 compiler support

### Test Failures
- Check if reference data is up to date
- Verify epsilon tolerance for floating-point comparisons
- Ensure cTensor operators are implemented correctly

### Memory Issues
- Tests use cTensor's pool allocator - ensure proper initialization
- Check for memory leaks with valgrind if available

## Future Enhancements

- Add tests for backward pass (gradient computation)
- Implement performance benchmarking
- Add fuzzing tests for edge cases
- Support for different data types (int, double)
- Integration with external test frameworks (CTest, Unity)