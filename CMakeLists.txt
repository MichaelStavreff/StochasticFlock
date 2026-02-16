cmake_minimum_required(VERSION 3.18)
project(stochastic_flock)

# 1. Global Settings
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_definitions(-DEIGEN_RUNTIME_NO_MALLOC)

# 2. Dependency: Eigen (FetchContent)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "..." FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "..." FORCE)
set(BUILD_TESTING OFF CACHE BOOL "..." FORCE) # Stops Eigen from running 50+ tests

include(FetchContent)
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
)
FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
    FetchContent_Populate(eigen)
    add_subdirectory(${eigen_SOURCE_DIR} ${eigen_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# 3. Dependency: Python & Pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED)

# 4. Target: Python Module
pybind11_add_module(stochastic_flock src/python_bindings.cpp)
target_include_directories(stochastic_flock PRIVATE src)
target_link_libraries(stochastic_flock PRIVATE Eigen3::Eigen)

# 5. Target: Optional Standalone C++ Solver (SFML)
find_package(SFML 2.5 COMPONENTS graphics window system QUIET)
if(SFML_FOUND)
    message(STATUS "SFML found! Building bird_solver...")
    add_executable(bird_solver src/bird_solver.cpp)
    target_include_directories(bird_solver PRIVATE src)
    target_link_libraries(bird_solver PRIVATE Eigen3::Eigen sfml-graphics sfml-window sfml-system)
else()
    message(STATUS "SFML NOT found. Skipping bird_solver.")
endif()

# 6. Optimizations & Flags (Applied to existing targets)
# We use a list so we don't have to repeat code for both targets
set(ALL_TARGETS stochastic_flock)
if(TARGET bird_solver)
    list(APPEND ALL_TARGETS bird_solver)
endif()

foreach(target ${ALL_TARGETS})
    if(NOT MSVC)
        target_compile_options(${target} PRIVATE -O3 -ffast-math -Wall)
        if(NATIVE_OPTIM)
            target_compile_options(${target} PRIVATE -march=native)
            include(CheckIPOSupported)
            check_ipo_supported(RESULT lto_supported OUTPUT lto_error)
            if(lto_supported)
                set_target_properties(${target} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
            endif()
        endif()
    endif()
endforeach()