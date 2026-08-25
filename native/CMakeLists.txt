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
)
target_include_directories(scl_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(SCL_WITH_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    target_sources(scl_core PRIVATE backends/cuda/lj_pairwise_cuda.cu)
    target_compile_definitions(scl_core PUBLIC SCL_WITH_CUDA)
    set_target_properties(scl_core PROPERTIES CUDA_STANDARD 17 CUDA_SEPARABLE_COMPILATION ON)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(scl_core PUBLIC CUDA::cudart)
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
