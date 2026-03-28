# CMakeList.txt : CMake project for Traveling Sales Person Cuda, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.10)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("Traveling Sales Person Cuda")

# Enable CUDA language
enable_language(CUDA)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Set CUDA architectures (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES native)

# Add source to this project's executable.
add_executable (CMakeTarget "Traveling Sales Person Cuda.cpp" "runners/common.cpp" "runners/cuda_runner.cu")
set_target_properties(CMakeTarget PROPERTIES OUTPUT_NAME "tsp")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET CMakeTarget PROPERTY CXX_STANDARD 20)
endif()

# Set CUDA standard
set_property(TARGET CMakeTarget PROPERTY CUDA_STANDARD 17)

# Match MSVC runtime library between CUDA and CXX objects
if(MSVC)
  target_compile_options(CMakeTarget PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/MD>)
endif()

# Link CUDA runtime library
target_link_libraries(CMakeTarget CUDA::cudart)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET CMakeTarget PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
