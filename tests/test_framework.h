#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include "../include/cten.h"

// Test framework macros and utilities
#define EPSILON 1e-6f
#define MAX_TEST_NAME_LEN 256
#define MAX_ERROR_MSG_LEN 512

// Test statistics
typedef struct {
    int total_tests;
    int passed_tests;
    int failed_tests;
} TestStats;

// Global test statistics
extern TestStats g_test_stats;

// Test assertion macros
#define ASSERT_TRUE(condition, message) \
    do { \
        g_test_stats.total_tests++; \
        if (!(condition)) { \
            printf("FAIL: %s - %s\n", __func__, message); \
            g_test_stats.failed_tests++; \
            return false; \
        } else { \
            g_test_stats.passed_tests++; \
        } \
    } while(0)

#define ASSERT_FALSE(condition, message) \
    ASSERT_TRUE(!(condition), message)

#define ASSERT_FLOAT_EQ(expected, actual, message) \
    do { \
        g_test_stats.total_tests++; \
        float diff = fabsf((expected) - (actual)); \
        if (diff > EPSILON) { \
            printf("FAIL: %s - %s (expected: %.6f, actual: %.6f, diff: %.6f)\n", \
                   __func__, message, expected, actual, diff); \
            g_test_stats.failed_tests++; \
            return false; \
        } else { \
            g_test_stats.passed_tests++; \
        } \
    } while(0)

#define ASSERT_TENSOR_EQ(expected_data, actual_tensor, size, message) \
    do { \
        bool tensors_equal = true; \
        for (int i = 0; i < size; i++) { \
            float diff = fabsf(expected_data[i] - actual_tensor.data->flex[i]); \
            if (diff > EPSILON) { \
                printf("FAIL: %s - %s at index %d (expected: %.6f, actual: %.6f, diff: %.6f)\n", \
                       __func__, message, i, expected_data[i], actual_tensor.data->flex[i], diff); \
                tensors_equal = false; \
                break; \
            } \
        } \
        g_test_stats.total_tests++; \
        if (!tensors_equal) { \
            g_test_stats.failed_tests++; \
            return false; \
        } else { \
            g_test_stats.passed_tests++; \
        } \
    } while(0)

#define ASSERT_SHAPE_EQ(expected_shape, actual_shape, message) \
    do { \
        bool shapes_equal = true; \
        for (int i = 0; i < 4; i++) { \
            if (expected_shape[i] != actual_shape[i]) { \
                printf("FAIL: %s - %s at dimension %d (expected: %d, actual: %d)\n", \
                       __func__, message, i, expected_shape[i], actual_shape[i]); \
                shapes_equal = false; \
                break; \
            } \
        } \
        g_test_stats.total_tests++; \
        if (!shapes_equal) { \
            g_test_stats.failed_tests++; \
            return false; \
        } else { \
            g_test_stats.passed_tests++; \
        } \
    } while(0)

// Test runner macros
#define RUN_TEST(test_func) \
    do { \
        printf("Running %s...\n", #test_func); \
        if (test_func()) { \
            printf("PASS: %s\n", #test_func); \
        } else { \
            printf("FAIL: %s\n", #test_func); \
        } \
        printf("\n"); \
    } while(0)

// Utility functions
void init_test_framework(void);
void test_framework_init(void);
void test_framework_cleanup(void);
int get_tests_passed(void);
int get_total_tests(void);
void print_test_summary(void);
Tensor create_tensor_from_data(TensorShape shape, float* data, bool requires_grad);
void print_tensor_debug(Tensor tensor, const char* name);
bool compare_tensors(Tensor a, Tensor b, float epsilon);
bool compare_tensor_with_data(Tensor tensor, float* expected_data, int size, float epsilon);

// Test function declarations
bool test_arithmetic_operations(void);
bool test_matrix_operations(void);
bool test_reduction_operations(void);
bool test_activation_functions(void);
bool test_loss_functions(void);
bool test_neural_network_operations(void);

// Individual operator tests
bool test_tensor_add(void);
bool test_tensor_sub(void);
bool test_tensor_mul(void);
bool test_tensor_div(void);
bool test_tensor_pow(void);
bool test_tensor_add_scalar(void);
bool test_tensor_mul_scalar(void);
bool test_tensor_matmul(void);
bool test_tensor_transpose(void);
bool test_tensor_sum(void);
bool test_tensor_mean(void);
bool test_tensor_max(void);
bool test_tensor_min(void);
bool test_tensor_argmax(void);
bool test_nn_relu(void);
bool test_nn_sigmoid(void);
bool test_nn_tanh(void);
bool test_nn_softmax(void);
bool test_nn_log(void);
bool test_nn_exp(void);
bool test_nn_sin(void);
bool test_nn_cos(void);
bool test_nn_tan(void);
bool test_nn_crossentropy(void);
bool test_nn_softmax_crossentropy(void);
bool test_nn_linear(void);