cmake_minimum_required(VERSION 3.0.0)
project(lammpsPluginTest VERSION 0.1.0)

# set op prec
set(HIGH_PREC_DEF "HIGH_PREC")
# this defination doesn't work, but leaving it empty will cause error
set(LOW_PREC_DEF "LOW_PREC")
set(HIGH_PREC_VARIANT "")
set(LOW_PREC_VARIANT "_low")

# if (BUILD_CPP_IF) set(LIB_TORCHMOLNET_OP "torchmolnet_op")
set(LIB_TORCHMOLNET_CC "torchmolnet_cc")
# endif()

find_package(Git)
if(GIT_FOUND)
  execute_process(
    COMMAND git describe --tags --dirty
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_SUMM
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(
    COMMAND git log -1 --format=%h
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(
    COMMAND git rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(
    COMMAND git show -s --format=%ci ${GIT_HASH}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_DATE
    OUTPUT_STRIP_TRAILING_WHITESPACE)
endif(GIT_FOUND)

if(Torch_DIR
   AND LAMMPS_BINARY_ROOT)

  set(Torch_DIR ${Torch_DIR}/share/cmake/Torch)
  set(LAMMPS_DIR ${LAMMPS_BINARY_ROOT}/lib/cmake/LAMMPS)
  set(LAMMPS_HEADER_DIR ${LAMMPS_BINARY_ROOT}/include/lammps)

  find_package(Torch REQUIRED)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
  if(TORCH_FOUND)
    if(CUDA_FOUND)
      message(STATUS "CUDA is found, use CUDA features of Torch")
    else()
      message(STATUS "CUDA is not found, use CPU features of Torch")
    endif()
  endif(TORCH_FOUND)
else()
  if(NOT Torch_DIR)
    message(FATAL_ERROR "Torch_DIR is not set")
  endif()
  if(NOT LAMMPS_BINARY_ROOT)
    message(FATAL_ERROR "LAMMPS_BINARY_ROOT is not set")
  endif()
endif()

get_cmake_property(_variableNames VARIABLES)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach()

message(STATUS "Torch version: ${Torch_VERSION}")
message(STATUS "Git summary: ${GIT_SUMM}")

add_subdirectory(src)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
