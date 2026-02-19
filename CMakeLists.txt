cmake_minimum_required(VERSION 3.18)
project(stochastic_flock LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Dependencies (Eigen)
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

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED) #added .Module

# if(NOT pybind11_FOUND)
#     execute_process(
#         COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
#         OUTPUT_VARIABLE pybind11_DIR
#         OUTPUT_STRIP_TRAILING_WHITESPACE
#         ERROR_QUIET
#     )
# endif()

# # Now find it (using the path discovered above if necessary)
# find_package(pybind11 REQUIRED)

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

# 3. Define Optimization Settings (Interface Library)
# This replaces your 'foreach' loop logic
add_library(sim_opts INTERFACE)
target_compile_definitions(sim_opts INTERFACE EIGEN_RUNTIME_NO_MALLOC)

if(NOT MSVC)
    target_compile_options(sim_opts INTERFACE -O3 -ffast-math -Wall)
    if(NATIVE_OPTIM)
        target_compile_options(sim_opts INTERFACE -march=native)
        include(CheckIPOSupported)
        check_ipo_supported(RESULT lto_supported)
        if(lto_supported)
            set_property(TARGET sim_opts PROPERTY INTERFACE_INTERPROCEDURAL_OPTIMIZATION TRUE)
        endif()
    endif()
endif()

pybind11_add_module(stochastic_flock src/python_bindings.cpp)
target_include_directories(stochastic_flock PRIVATE src)
target_link_libraries(stochastic_flock PRIVATE Eigen3::Eigen sim_opts)

install(TARGETS stochastic_flock DESTINATION stochastic_flock)

#TODO : ^^^ doesnt install to stochastic_flock folder. perhaps completely remove coupled makefile with bird_solver and python?


# 5. Target: Optional SFML (Skipped during Python build)
if(NOT SKBUILD)
    find_package(SFML 2.5 COMPONENTS graphics window system QUIET)
    if(SFML_FOUND)
        add_executable(bird_solver src/bird_solver.cpp)
        target_include_directories(bird_solver PRIVATE src)
        target_link_libraries(bird_solver PRIVATE Eigen3::Eigen sfml-graphics sfml-window sfml-system sim_opts)
    endif()
endif()