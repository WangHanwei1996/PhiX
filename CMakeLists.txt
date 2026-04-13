cmake_minimum_required(VERSION 3.16)
project(PhiX LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------------
# C++ / CUDA standards
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CXX_STANDARD_REQUIRED  ON)
# CMAKE_CUDA_STANDARD 17 requires CMake 3.18+; set the flag directly for 3.16 compatibility
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17")
endif()

# ---------------------------------------------------------------------------
# CUDA architecture
# Defaults to sm_75 (Turing).  Override at configure time:
#   cmake -DPHIX_CUDA_ARCH=86 ..   (Ampere)
# ---------------------------------------------------------------------------
set(PHIX_CUDA_ARCH "75" CACHE STRING "CUDA compute capability (e.g. 75, 80, 86, 89)")
# CMAKE_CUDA_ARCHITECTURES requires CMake 3.18+; fall back to -arch flag for 3.16 compatibility
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
    set(CMAKE_CUDA_ARCHITECTURES ${PHIX_CUDA_ARCH})
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_${PHIX_CUDA_ARCH}")
endif()

# ---------------------------------------------------------------------------
# Extended lambdas — required for [] __device__ (...){} in Equation DSL
# ---------------------------------------------------------------------------
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")

# ---------------------------------------------------------------------------
# Include paths
# ---------------------------------------------------------------------------
include_directories(${CMAKE_SOURCE_DIR}/include)

# ---------------------------------------------------------------------------
# PhiX core library  (static)
# ---------------------------------------------------------------------------
add_library(phix STATIC
    src/mesh/Mesh.cpp
    src/field/ScalarField.cu
    src/field/VectorField.cu
    src/boundary/Boundary.cu
    src/equation/Equation.cu
    src/equation/VectorEquation.cu
    src/solver/Solver.cu
    src/solver/VectorSolver.cu
)

target_include_directories(phix PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
)

# Ensure CUDA device-code linking is enabled for the library
set_target_properties(phix PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ---------------------------------------------------------------------------
# Tutorials
# ---------------------------------------------------------------------------
add_subdirectory(tutorials/quickstart)

# ---------------------------------------------------------------------------
# Develop
# ---------------------------------------------------------------------------
add_subdirectory("develop/Spinodal Decomposition")
