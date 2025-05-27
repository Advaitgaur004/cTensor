#include "test_framework.h"

// Forward declarations
void run_implemented_operator_tests(void);

/**
 * @brief Entry point for the cTensor test suite.
 *
 * Initializes the test framework, runs all implemented operator tests, performs cleanup, and prints a summary of test results. Returns 0 if all tests pass, otherwise returns 1.
 *
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line argument strings.
 * @return int Exit code: 0 if all tests pass, 1 otherwise.
 */
int main(int argc, char* argv[]) {
    printf("Starting cTensor Test Suite\n");
    printf("===========================\n\n");
    
    test_framework_init();
    
    // Run test suites
    run_implemented_operator_tests();
    
    test_framework_cleanup();
    
    printf("\n===========================\n");
    printf("Test Suite Complete\n");
    printf("Tests passed: %d/%d\n", get_tests_passed(), get_total_tests());
    
    return get_tests_passed() == get_total_tests() ? 0 : 1;
}