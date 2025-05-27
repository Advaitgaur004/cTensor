#include "test_framework.h"
#include "cten.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Helper function is defined in test_framework.c

/**
 * @brief Tests element-wise addition of two tensors.
 *
 * Creates two tensors with predefined data, performs element-wise addition, and asserts that the result matches the expected output.
 *
 * @return true if the test passes.
 */
bool test_tensor_add(void) {
    printf("Testing Tensor_add...\n");
    
    // Reference data from PyTorch
    float input_a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float input_b[] = {0.5f, 1.5f, 2.5f, 3.5f};
    float expected[] = {1.5f, 3.5f, 5.5f, 7.5f};
    TensorShape shape = {4, 0, 0, 0};
    
    Tensor a = create_tensor_from_data(shape, input_a, false);
    Tensor b = create_tensor_from_data(shape, input_b, false);
    
    Tensor result = Tensor_add(a, b);
    
    ASSERT_TENSOR_EQ(expected, result, 4, "Tensor addition failed");
    
    return true;
}

/**
 * @brief Tests element-wise multiplication of two tensors.
 *
 * Creates two tensors with predefined data, multiplies them element-wise using `Tensor_mul`, and asserts that the result matches the expected output.
 *
 * @return true if the test passes.
 */
bool test_tensor_mul(void) {
    printf("Testing Tensor_mul...\n");
    
    // Reference data from PyTorch
    float input_a[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float input_b[] = {2.0f, 3.0f, 4.0f, 5.0f};
    float expected[] = {2.0f, 6.0f, 12.0f, 20.0f};
    TensorShape shape = {4, 0, 0, 0};
    
    Tensor a = create_tensor_from_data(shape, input_a, false);
    Tensor b = create_tensor_from_data(shape, input_b, false);
    
    Tensor result = Tensor_mul(a, b);
    
    ASSERT_TENSOR_EQ(expected, result, 4, "Tensor multiplication failed");
    
    return true;
}

/**
 * @brief Tests element-wise multiplication of a tensor by a scalar.
 *
 * Creates a tensor from predefined data, multiplies it by a scalar value, and asserts that the result matches the expected output.
 *
 * @return true if the test passes.
 */
bool test_tensor_mul_scalar(void) {
    printf("Testing Tensor_mulf...\n");
    
    // Reference data from PyTorch
    float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float scalar = 2.5f;
    float expected[] = {2.5f, 5.0f, 7.5f, 10.0f};
    TensorShape shape = {4, 0, 0, 0};
    
    Tensor a = create_tensor_from_data(shape, input, false);
    
    Tensor result = Tensor_mulf(a, scalar);
    
    ASSERT_TENSOR_EQ(expected, result, 4, "Tensor scalar multiplication failed");
    
    return true;
}

/**
 * @brief Tests element-wise subtraction of two tensors.
 *
 * Creates two tensors with predefined data, performs element-wise subtraction, and asserts that the result matches the expected output.
 *
 * @return true if the subtraction result is correct.
 */
bool test_tensor_sub(void) {
    printf("Testing Tensor_sub...\n");
    
    // Reference data from PyTorch
    float input_a[] = {5.0f, 4.0f, 3.0f, 2.0f};
    float input_b[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float expected[] = {4.0f, 3.0f, 2.0f, 1.0f};
    TensorShape shape = {4, 0, 0, 0};
    
    Tensor a = create_tensor_from_data(shape, input_a, false);
    Tensor b = create_tensor_from_data(shape, input_b, false);
    
    Tensor result = Tensor_sub(a, b);
    
    ASSERT_TENSOR_EQ(expected, result, 4, "Tensor subtraction failed");
    
    return true;
}

/**
 * @brief Tests the computation of the mean value of tensor elements.
 *
 * Creates a tensor from a static array, computes its mean using `Tensor_mean`, and asserts that the result matches the expected mean value.
 *
 * @return true if the test passes.
 */
bool test_tensor_mean(void) {
    printf("Testing Tensor_mean...\n");
    
    // Reference data from PyTorch
    float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float expected = 2.5f;  // Mean of [1,2,3,4] = 2.5
    TensorShape shape = {4, 0, 0, 0};
    
    Tensor a = create_tensor_from_data(shape, input, false);
    
    Tensor result = Tensor_mean(a);
    
    ASSERT_FLOAT_EQ(expected, result.data->flex[0], "Tensor mean failed");
    
    return true;
}

/**
 * @brief Tests the sum operation on a tensor.
 *
 * Creates a tensor from a predefined array, computes the sum of its elements using `Tensor_sum`, and asserts that the result matches the expected value.
 *
 * @return true if the test passes.
 */
bool test_tensor_sum(void) {
    printf("Testing Tensor_sum...\n");
    
    // Reference data from PyTorch
    float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float expected = 10.0f;  // Sum of [1,2,3,4] = 10
    TensorShape shape = {4, 0, 0, 0};
    
    Tensor a = create_tensor_from_data(shape, input, false);
    
    Tensor result = Tensor_sum(a);
    
    ASSERT_FLOAT_EQ(expected, result.data->flex[0], "Tensor sum failed");
    
    return true;
}

/**
 * @brief Tests matrix multiplication of two 2x2 tensors.
 *
 * Creates two 2x2 tensors with predefined values, performs matrix multiplication using `Tensor_matmul`, and asserts that the result matches the expected output.
 *
 * @return true if the matrix multiplication produces the correct result.
 */
bool test_tensor_matmul(void) {
    printf("Testing Tensor_matmul...\n");
    
    // Reference data from PyTorch: 2x2 matrices
    float input_a[] = {1.0f, 2.0f, 3.0f, 4.0f};  // [[1,2], [3,4]]
    float input_b[] = {5.0f, 6.0f, 7.0f, 8.0f};  // [[5,6], [7,8]]
    float expected[] = {19.0f, 22.0f, 43.0f, 50.0f};  // [[19,22], [43,50]]
    
    TensorShape shape_a = {2, 2, 0, 0};
    TensorShape shape_b = {2, 2, 0, 0};
    
    Tensor a = create_tensor_from_data(shape_a, input_a, false);
    Tensor b = create_tensor_from_data(shape_b, input_b, false);
    
    Tensor result = Tensor_matmul(a, b);
    
    ASSERT_TENSOR_EQ(expected, result, 4, "Tensor matrix multiplication failed");
    
    return true;
}

/**
 * @brief Executes all unit tests for implemented tensor operators.
 *
 * Runs a suite of tests covering addition, multiplication, scalar multiplication, subtraction, mean, sum, and matrix multiplication for tensors, printing progress and completion messages.
 */
void run_implemented_operator_tests(void) {
    printf("\n=== Running Implemented Operator Tests ===\n");
    
    RUN_TEST(test_tensor_add);
    RUN_TEST(test_tensor_mul);
    RUN_TEST(test_tensor_mul_scalar);
    RUN_TEST(test_tensor_sub);
    RUN_TEST(test_tensor_mean);
    RUN_TEST(test_tensor_sum);
    RUN_TEST(test_tensor_matmul);
    
    printf("=== Implemented Operator Tests Complete ===\n\n");
}