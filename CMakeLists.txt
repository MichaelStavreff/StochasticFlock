cmake_minimum_required(VERSION 3.18)
project(stochastic_flock LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_LEAVE_TEST_IN_ALL_TARGET OFF CACHE BOOL "" FORCE) 
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)            

FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
)

# Use the "manual" population instead of MakeAvailable 
# so we can use EXCLUDE_FROM_ALL
FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
    FetchContent_Populate(eigen)
    # The EXCLUDE_FROM_ALL flag here is the magic bit.
    # It tells CMake: "Build this, but don't install its files."
    add_subdirectory(${eigen_SOURCE_DIR} ${eigen_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

find_package(pybind11 QUIET)

# If not found, fall back to your manual search (works for your local build)
if(NOT pybind11_FOUND)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    find_package(pybind11 REQUIRED PATHS ${pybind11_DIR} NO_DEFAULT_PATH)
endif()

# --- 2. Build Options (sim_opts) ---
add_library(sim_opts INTERFACE)
target_compile_definitions(sim_opts INTERFACE EIGEN_RUNTIME_NO_MALLOC)

option(USE_SANITIZERS "Enable Address and Undefined Behavior Sanitizers" OFF)

if(NOT MSVC)
    # CONFIG SENSITIVE FLAGS:
    # -O0 for Debug (make dev), -O3 -ffast-math for Release (make portable/native)
    target_compile_options(sim_opts INTERFACE 
        $<$<CONFIG:Debug>:-O0 -g> 
        $<$<CONFIG:Release>:-O3 -ffast-math>
        -Wall
    )
    if(USE_SANITIZERS)
        target_compile_options(sim_opts INTERFACE -fsanitize=address,undefined)
        target_link_options(sim_opts INTERFACE -fsanitize=address,undefined)
        message(STATUS "Sanitizers enabled for all targets")
    endif()

    # NATIVE OPTIMIZATION: Only added if explicitly requested in Makefile
    if(NATIVE_OPTIM)
        target_compile_options(sim_opts INTERFACE -march=native)
    endif()
endif()

# --- 3. LTO Configuration ---
include(CheckIPOSupported)
check_ipo_supported(RESULT lto_supported)

# --- 4. Python Module (stochastic_flock) ---
pybind11_add_module(stochastic_flock src/python_bindings.cpp)
target_include_directories(stochastic_flock PRIVATE src)
target_link_libraries(stochastic_flock PRIVATE Eigen3::Eigen sim_opts)
target_precompile_headers(stochastic_flock PRIVATE src/pch_python.hpp)

# Always use LTO for the Python module in Release for max performance
if(lto_supported)
    set_target_properties(stochastic_flock PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

# --- 5. Executable (bird_solver) ---
if(NOT SKBUILD)
    find_package(SFML 2.5 COMPONENTS graphics window system QUIET)
    if(SFML_FOUND)
        add_executable(bird_solver src/bird_solver.cpp)
        target_include_directories(bird_solver PRIVATE src)
        target_link_libraries(bird_solver PRIVATE Eigen3::Eigen sfml-graphics sfml-window sfml-system sim_opts)
        target_precompile_headers(bird_solver PRIVATE src/pch_exec.hpp)

        # LINKER SETTINGS: 
        # LTO is ON for Release (portable/native), but OFF for Debug (dev) for instant linking.
        if(lto_supported)
            set_target_properties(bird_solver PROPERTIES 
                INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
                INTERPROCEDURAL_OPTIMIZATION_DEBUG FALSE
            )
        endif()
    endif()
endif()