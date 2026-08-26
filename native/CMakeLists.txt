cmake_minimum_required(VERSION 3.18)
project(scl_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# IEEE-754 strict math, deliberately: no -ffast-math, no auto-vectorization
# flags that could reorder floating-point reductions. This is the same
# "same binary, same machine" determinism claim execution/gromacs.py makes
# for GROMACS, and it is only true if the compiler is not allowed to
# reassociate floating-point sums. See docs/SCL_CONTRACT.md's
# reproducibility section.
add_compile_options(-Wall -Wextra -O2 -fno-fast-math)

option(SCL_WITH_CUDA "Build the CUDA backend (requires the CUDA toolkit)" OFF)
option(SCL_BUILD_TESTS "Build native unit tests" ON)

find_package(nlohmann_json 3.9 REQUIRED)

add_library(scl_core STATIC
  src/lj_pairwise.cpp
  src/backend.cpp
  src/fourier.cpp
  src/operation_registry.cpp
  src/op_lj_pairwise.cpp
  src/op_fourier.cpp
  src/least_squares.cpp
  src/op_least_squares.cpp
)
target_include_directories(scl_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# Operation modules and the CUDA backends include "backends/cuda/..."
# relative to native/ itself; that directory is intentionally outside
# include/scl/ because it is not part of the CPU-only public interface.
target_include_directories(scl_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(SCL_WITH_CUDA)
  # Modern CMake (CUDA_ARCHITECTURES / policy CMP0104) requires an explicit
  # target architecture list before enabling the CUDA language -- it will
  # not silently fall back to "detect the installed GPU" the way older
  # CMake/nvcc invocations did, and this Phase 3 build environment has NO
  # GPU AT ALL (verified: no /dev/nvidia*, no PCI VGA/3D device, `nvidia-
  # smi` absent) for CMake to detect one from even if it tried. The default
  # below is therefore a REASONABLE, DOCUMENTED CHOICE of common compute
  # capabilities (Volta through Ada), not a value verified against real
  # deployment hardware -- override with -DSCL_CUDA_ARCHITECTURES=... on a
  # machine where the real target GPU is known.
  if(NOT DEFINED SCL_CUDA_ARCHITECTURES)
    set(SCL_CUDA_ARCHITECTURES "70;75;80;86;89" CACHE STRING
        "Target CUDA compute capabilities (no GPU present in this build environment to detect from)")
  endif()
  set(CMAKE_CUDA_ARCHITECTURES ${SCL_CUDA_ARCHITECTURES})

  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    target_sources(scl_core PRIVATE
      backends/cuda/lj_pairwise_cuda.cu
      backends/cuda/fourier_cuda.cu
    )
    target_compile_definitions(scl_core PUBLIC SCL_WITH_CUDA)
    set_target_properties(scl_core PROPERTIES
      CUDA_STANDARD 17
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_ARCHITECTURES "${SCL_CUDA_ARCHITECTURES}")
    find_package(CUDAToolkit REQUIRED)
    # cuFFT backs fourier_transform_1d's CUDA path -- an established
    # vendor primitive rather than a hand-written GPU FFT.
    target_link_libraries(scl_core PUBLIC CUDA::cudart CUDA::cufft)
  else()
    message(WARNING "SCL_WITH_CUDA=ON but no CUDA compiler was found; building CPU-only")
  endif()
endif()

add_executable(scl_cli src/main.cpp)
target_link_libraries(scl_cli PRIVATE scl_core nlohmann_json::nlohmann_json)

if(SCL_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
