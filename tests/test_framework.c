#include "test_framework.h"

// Global test statistics
TestStats g_test_stats = {0, 0, 0};

/**
 * @brief Initializes the cTensor test framework and resets test statistics.
 *
 * Resets global test counters, initializes the cTensor library, and starts a dedicated memory pool for testing. Prints a header indicating the start of the test session.
 */
void init_test_framework(void) {
    g_test_stats.total_tests = 0;
    g_test_stats.passed_tests = 0;
    g_test_stats.failed_tests = 0;
    
    // Initialize cTensor
    cten_initilize();
    
    // Start memory pool for tests
    cten_begin_malloc(0);  // Use pool ID 0 for tests
    
    printf("=== cTensor Test Framework ===\n");
    printf("Testing determinism against PyTorch reference data\n\n");
}

/**
 * @brief Prints a summary of test results and finalizes the test environment.
 *
 * Displays the total number of tests, passed and failed counts, and the overall success rate. Cleans up the cTensor library and releases the test memory pool.
 */
void print_test_summary(void) {
    printf("=== Test Summary ===\n");
    printf("Total tests: %d\n", g_test_stats.total_tests);
    printf("Passed: %d\n", g_test_stats.passed_tests);
    printf("Failed: %d\n", g_test_stats.failed_tests);
    
    if (g_test_stats.failed_tests == 0) {
        printf("All tests PASSED! ✓\n");
    } else {
        printf("Some tests FAILED! ✗\n");
    }
    
    float success_rate = g_test_stats.total_tests > 0 ? 
        (float)g_test_stats.passed_tests / g_test_stats.total_tests * 100.0f : 0.0f;
    printf("Success rate: %.1f%%\n", success_rate);
    
    // End memory pool and cleanup cTensor
    cten_end_malloc();
    cten_free(0);  // Free pool ID 0
    cten_finalize();
}

/**
 * @brief Creates a tensor with the specified shape and copies provided data into it.
 *
 * @param shape The shape of the tensor to create.
 * @param data Pointer to the float array containing the data to copy into the tensor.
 * @param requires_grad Indicates whether the tensor should track gradients.
 * @return Tensor A new tensor containing the copied data.
 */
Tensor create_tensor_from_data(TensorShape shape, float* data, bool requires_grad) {
    Tensor tensor = Tensor_new(shape, requires_grad);
    int numel = TensorShape_numel(shape);
    for (int i = 0; i < numel; i++) {
        tensor.data->flex[i] = data[i];
    }
    return tensor;
}

/**
 * @brief Prints debug information for a tensor, including its name, shape, and up to the first 10 data elements.
 *
 * Useful for inspecting tensor contents and verifying shape and data during testing or debugging.
 */
void print_tensor_debug(Tensor tensor, const char* name) {
    printf("Tensor %s: shape=[", name);
    int dim = TensorShape_dim(tensor.shape);
    for (int i = 0; i < dim; i++) {
        printf("%d", tensor.shape[i]);
        if (i < dim - 1) printf(", ");
    }
    printf("], data=[");
    
    int numel = TensorShape_numel(tensor.shape);
    for (int i = 0; i < numel && i < 10; i++) {  // Print max 10 elements
        printf("%.6f", tensor.data->flex[i]);
        if (i < numel - 1 && i < 9) printf(", ");
    }
    if (numel > 10) printf("...");
    printf("]\n");
}

/**
 * @brief Compares two tensors for equality within a specified tolerance.
 *
 * Checks if both tensors have identical shapes and if all corresponding elements differ by no more than the given epsilon value.
 *
 * @param a First tensor to compare.
 * @param b Second tensor to compare.
 * @param epsilon Maximum allowed difference between corresponding elements.
 * @return true if tensors are equal within the specified tolerance and have the same shape, false otherwise.
 */
bool compare_tensors(Tensor a, Tensor b, float epsilon) {
    // Check shapes
    for (int i = 0; i < 4; i++) {
        if (a.shape[i] != b.shape[i]) {
            return false;
        }
    }
    
    // Check data
    int numel = TensorShape_numel(a.shape);
    for (int i = 0; i < numel; i++) {
        if (fabsf(a.data->flex[i] - b.data->flex[i]) > epsilon) {
            return false;
        }
    }
    
    return true;
}

/**
 * @brief Compares a tensor's data to a reference array within a tolerance.
 *
 * Checks if the tensor's data matches the provided float array element-wise, allowing for differences up to the specified epsilon. Returns true if all elements match within the tolerance and the sizes are equal; otherwise, returns false.
 *
 * @param tensor The tensor to compare.
 * @param expected_data Pointer to the reference float array.
 * @param size Number of elements in the reference array.
 * @param epsilon Maximum allowed difference between corresponding elements.
 * @return true if the tensor matches the reference data within epsilon, false otherwise.
 */
bool compare_tensor_with_data(Tensor tensor, float* expected_data, int size, float epsilon) {
    if (TensorShape_numel(tensor.shape) != size) {
        return false;
    }
    
    for (int i = 0; i < size; i++) {
        if (fabsf(tensor.data->flex[i] - expected_data[i]) > epsilon) {
            return false;
        }
    }
    
    return true;
}

/**
 * @brief Initializes the test framework for cTensor tests.
 *
 * This function is an alias for `init_test_framework()` and prepares the test environment by resetting statistics, initializing the cTensor library, and setting up the memory pool.
 */
void test_framework_init(void) {
    init_test_framework();
}

/**
 * @brief Cleans up the test framework and prints a summary of test results.
 *
 * Finalizes the cTensor test environment by reporting test statistics and releasing resources.
 */
void test_framework_cleanup(void) {
    print_test_summary();
}

/**
 * @brief Returns the number of tests that have passed.
 *
 * @return int The count of passed tests.
 */
int get_tests_passed(void) {
    return g_test_stats.passed_tests;
}

/**
 * @brief Returns the total number of tests executed by the test framework.
 *
 * @return int Total test count.
 */
int get_total_tests(void) {
    return g_test_stats.total_tests;
}