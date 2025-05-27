#include "test_framework.h"

// Forward declarations
void run_implemented_operator_tests(void);

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