cmake_minimum_required(VERSION 3.10)
project(cTensorTests LANGUAGES C)

# Include project headers
include_directories(../include)

# Collect source files for the library (excluding main.c from src2)
file(GLOB_RECURSE LIB_SOURCES
    "../src/*.c"
)

# Collect test source files
file(GLOB TEST_SOURCES
    "test_framework.c"
    "test_operators_implemented.c"
    "test_main.c"
)

# Create test executable
add_executable(cten_tests ${LIB_SOURCES} ${TEST_SOURCES})

# Set C standard
set_target_properties(cten_tests PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
)

# Link math library only on non-Windows systems
if(NOT WIN32)
    target_link_libraries(cten_tests PRIVATE m)
endif()

# Enable testing
enable_testing()

# Add test
add_test(NAME cten_operator_tests COMMAND cten_tests)

# Set test properties
set_tests_properties(cten_operator_tests PROPERTIES
    TIMEOUT 60
    PASS_REGULAR_EXPRESSION "All tests PASSED"
)