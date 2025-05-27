#include "test_framework.h"

// Global test statistics
TestStats g_test_stats = {0, 0, 0};

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

Tensor create_tensor_from_data(TensorShape shape, float* data, bool requires_grad) {
    Tensor tensor = Tensor_new(shape, requires_grad);
    int numel = TensorShape_numel(shape);
    for (int i = 0; i < numel; i++) {
        tensor.data->flex[i] = data[i];
    }
    return tensor;
}

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

// Alias functions for compatibility
void test_framework_init(void) {
    init_test_framework();
}

void test_framework_cleanup(void) {
    print_test_summary();
}

int get_tests_passed(void) {
    return g_test_stats.passed_tests;
}

int get_total_tests(void) {
    return g_test_stats.total_tests;
}