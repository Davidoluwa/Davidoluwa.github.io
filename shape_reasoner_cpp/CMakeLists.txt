cmake_minimum_required(VERSION 3.20)
project(sera_shape_reasoner VERSION 0.1.0 LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(sera_core
    src/shape_engine.cpp
    src/categorical_engine.cpp
    src/arithmetic_learner.cpp
    src/boolean_program_learner.cpp
    src/neural_shape_field.cpp
    src/neural_proof_kernel.cpp
    src/interaction_workspace_kernel.cpp
)
target_include_directories(sera_core PUBLIC include)
target_compile_options(sera_core PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -march=native -Wall -Wextra -Wpedantic>
)

add_executable(sera_stress src/stress_main.cpp)
target_link_libraries(sera_stress PRIVATE sera_core)
target_compile_options(sera_stress PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -march=native -Wall -Wextra -Wpedantic>
)

add_executable(sera_s1_stress src/s1_stress_main.cpp)
target_link_libraries(sera_s1_stress PRIVATE sera_core)
target_compile_options(sera_s1_stress PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -march=native -Wall -Wextra -Wpedantic>
)

add_executable(sera_arithmetic_research src/arithmetic_research_main.cpp)
target_link_libraries(sera_arithmetic_research PRIVATE sera_core)
target_compile_options(sera_arithmetic_research PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -march=native -Wall -Wextra -Wpedantic>
)

add_executable(sera_no_cue_research src/no_cue_research_main.cpp)
target_link_libraries(sera_no_cue_research PRIVATE sera_core)
target_compile_options(sera_no_cue_research PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -march=native -Wall -Wextra -Wpedantic>
)

add_executable(sera_neural_shape_research src/neural_shape_research_main.cpp)
target_link_libraries(sera_neural_shape_research PRIVATE sera_core)
target_compile_options(sera_neural_shape_research PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -march=native -Wall -Wextra -Wpedantic>
)

add_executable(sera_neural_proof_research src/neural_proof_research_main.cpp)
target_link_libraries(sera_neural_proof_research PRIVATE sera_core)
target_compile_options(sera_neural_proof_research PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -march=native -Wall -Wextra -Wpedantic>
)

if(BUILD_TESTING)
    add_executable(
        sera_neural_proof_kernel_test
        tests/neural_proof_kernel_test.cpp
    )
    target_link_libraries(sera_neural_proof_kernel_test PRIVATE sera_core)
    target_compile_options(sera_neural_proof_kernel_test PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:
            -O2 -Wall -Wextra -Wpedantic
        >
    )
    add_test(
        NAME sera_neural_proof_kernel_test
        COMMAND sera_neural_proof_kernel_test
    )

    add_executable(
        sera_interaction_workspace_kernel_test
        tests/interaction_workspace_kernel_test.cpp
    )
    target_link_libraries(
        sera_interaction_workspace_kernel_test PRIVATE sera_core
    )
    target_compile_options(sera_interaction_workspace_kernel_test PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:
            -O2 -Wall -Wextra -Wpedantic
        >
    )
    add_test(
        NAME sera_interaction_workspace_kernel_test
        COMMAND sera_interaction_workspace_kernel_test
    )
endif()

add_executable(sera_s5_comparison src/s5_comparison_main.cpp)
target_link_libraries(sera_s5_comparison PRIVATE sera_core)
target_compile_options(sera_s5_comparison PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -march=native -Wall -Wextra -Wpedantic>
)

add_executable(sera_s5_proof src/s5_proof_main.cpp)
target_link_libraries(sera_s5_proof PRIVATE sera_core)
target_compile_options(sera_s5_proof PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -march=native -Wall -Wextra -Wpedantic>
)
