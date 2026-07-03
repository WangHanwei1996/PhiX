cmake_minimum_required(VERSION 3.16)
project(PhiX LANGUAGES CXX CUDA)

enable_testing()

# ---------------------------------------------------------------------------
# Build type — default to Release (an empty CMAKE_BUILD_TYPE compiles the
# host side with no optimisation at all)
# ---------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message(STATUS "PhiX: CMAKE_BUILD_TYPE not set — defaulting to Release")
endif()

# ---------------------------------------------------------------------------
# Profiling helpers
#   PHIX_LINEINFO=ON     nvcc -lineinfo (source-correlated kernel profiling)
#   PHIX_ENABLE_NVTX=ON  compile PHIX_NVTX_RANGE() into real NVTX ranges
# ---------------------------------------------------------------------------
option(PHIX_LINEINFO    "Add -lineinfo to CUDA compiles (profiling)" OFF)
option(PHIX_ENABLE_NVTX "Enable NVTX ranges (perf/Perf.h)"           OFF)

# ---------------------------------------------------------------------------
# Precision policy (core/Real.h)
#   DOUBLE (default): Real = double — full test suite + applications
#   FLOAT           : Real = float  — library + float smoke tests + benchmark
#                     (apps and 1e-12-tolerance suites are not configured)
# ---------------------------------------------------------------------------
set(PHIX_PRECISION "DOUBLE" CACHE STRING "Field scalar type: DOUBLE or FLOAT")
if(PHIX_PRECISION STREQUAL "FLOAT")
    add_compile_definitions(PHIX_REAL_FLOAT)
    message(STATUS "PhiX: FLOAT precision build (Real = float)")
elseif(NOT PHIX_PRECISION STREQUAL "DOUBLE")
    message(FATAL_ERROR "PHIX_PRECISION must be DOUBLE or FLOAT (got '${PHIX_PRECISION}')")
endif()
if(PHIX_LINEINFO)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
endif()
if(PHIX_ENABLE_NVTX)
    add_compile_definitions(PHIX_ENABLE_NVTX)
endif()

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

# cuFFT (spectral elasticity / future spectral solvers)
find_package(CUDAToolkit REQUIRED)

# ---------------------------------------------------------------------------
# PhiX core library  (static)
# ---------------------------------------------------------------------------
add_library(phix STATIC
    src/mesh/Patch.cpp
    src/mesh/Mesh.cpp
    src/field/FieldLayout.cpp
    src/field/ScalarField.cu
    src/field/VectorField.cu
    src/field/FaceField.cu
    src/field/GibbsSimplex.cu
    src/field/NoiseGenerator.cu
    src/field/Reduce.cu
    src/operators/Laplacian.cu
    src/operators/Gradient.cu
    src/operators/Advection.cu
    src/operators/Anisotropy.cu
    src/operators/FaceOps.cu
    src/boundary/Boundary.cu
    src/boundary/BCFactory.cpp
    src/equation/Equation.cu
    src/equation/EvalPlan.cu
    src/equation/Expr.cu
    src/equation/EquationSystem.cu
    src/equation/VectorEquation.cu
    src/solver/Solver.cu
    src/solver/LinearSolver.cu
    src/solver/SemiImplicitSolver.cu
    src/solver/VectorSolver.cu
    src/IO/ConfigFile.cpp
    src/IO/FieldIO.cpp
    src/IO/OutputWriter.cpp
    src/IO/PFHubWriter.cpp
    src/material/FreeEnergyTable.cu
    src/material/KKS.cu
    src/material/KKSAntiTrapping.cu
    src/material/BinaryAlloy.cu
    src/lbm/LBM.cu
    src/diagnostics/Interface.cu
    src/mechanics/ElasticityFFT.cu
)

target_include_directories(phix PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
)

target_link_libraries(phix PUBLIC CUDA::cufft)

# Ensure CUDA device-code linking is enabled for the library
set_target_properties(phix PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ---------------------------------------------------------------------------
# Module tests + convergence suite (DOUBLE builds only — tolerances assume
# double precision); float smoke tests replace them in FLOAT builds
# ---------------------------------------------------------------------------
if(PHIX_PRECISION STREQUAL "DOUBLE")
    add_subdirectory(test/moduleTest)
    add_subdirectory(test/convergence)
else()
    add_subdirectory(test/floatSmoke)
endif()

# ---------------------------------------------------------------------------
# Performance benchmarks (both precisions)
# ---------------------------------------------------------------------------
add_subdirectory(test/benchmark)

# ---------------------------------------------------------------------------
# Tutorials
# ---------------------------------------------------------------------------
# TODO(v2.2.0): re-enable after solver refactor
# add_subdirectory(tutorials/quickstart)

# ---------------------------------------------------------------------------
# Sample tests + Applications (DOUBLE builds only — solver sources use
# double-typed host code and double-calibrated tolerances)
# ---------------------------------------------------------------------------
if(PHIX_PRECISION STREQUAL "DOUBLE")

add_subdirectory(test/sampleTest)

# add_subdirectory("applications/solvers/glass_formation/2D")
# add_subdirectory("applications/solvers/GFA/verification0")
add_subdirectory("applications/solvers/dendrite_growth/2D")
add_subdirectory("applications/solvers/dendrite_growth/dendrite_growth_flag")
# add_subdirectory("develop/Spinodal Decomposition")
# add_subdirectory("develop/Ostwald_Ripening")
# add_subdirectory("develop/fieldOperation")
# add_subdirectory("applications/tools/test")
add_subdirectory("applications/solvers/glass_formation_4_phases/2D")
add_subdirectory("applications/solvers/MPF_AC_DW")
add_subdirectory("applications/solvers/Cahn-Hillard+Allen-Cahn_double-well/2D")
add_subdirectory("applications/solvers/Cahn-Hillard_double-well/2D")
add_subdirectory("applications/solvers/GFA_binary")
add_subdirectory("applications/solvers/GFA_FeB")
add_subdirectory("develop/implicit_examples")

endif()
