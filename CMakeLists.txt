cmake_minimum_required(VERSION 3.15...3.27)

project(plinalg LANGUAGES CXX CUDA) # Replace 'my_project' with the name of your project

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

option(PLINALG_ADD_TESTS "Add tests for the plinalg library" ON)
# This option is to be used only to make a full compile_commands.json for LSP autocomplete & co
option(COMPILE_FOR_LSP "Compile everything and get compile_commands for LSP" OFF)

if (COMPILE_FOR_LSP)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()
  

set(NVHPC_CUDA_VERSION 11.8)
find_package(NVHPC REQUIRED COMPONENTS MATH MPI NCCL)

string(REPLACE "/lib64" "/include" NVHPC_MATH_INCLUDE_DIR ${NVHPC_MATH_LIBRARY_DIR})
string(REPLACE "/lib" "/include" NVHPC_NCCL_INCLUDE_DIR ${NVHPC_NCCL_LIBRARY_DIR})
string(REPLACE "/lib64" "/include" NVHPC_CUDA_INCLUDE_DIR ${NVHPC_CUDA_LIBRARY_DIR})
string(REPLACE "/lib64" "/bin/nvcc" NVHPC_CUDA_COMPILER ${NVHPC_CUDA_LIBRARY_DIR})

if (CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -noswitcherror")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -noswitcherror")
endif()

set(CMAKE_CUDA_COMPILER ${NVHPC_CUDA_COMPILER})

if (NOT SKBUILD OR COMPILE_FOR_LSP)
  message("\
          =====================================================================
          Building as a standalone C++ Library
          =====================================================================
          ")

  # **********************************************************************
  # Add pHermitian library
  # **********************************************************************
  add_library(pHermitian SHARED 
                  ${CMAKE_CURRENT_LIST_DIR}/lib/src/hermitian_kernel.cc 
                  ${CMAKE_CURRENT_LIST_DIR}/lib/src/hermitian_kernel.cc.cu
                  )

  target_include_directories(pHermitian PUBLIC
                  ${CMAKE_CURRENT_LIST_DIR}/lib/include
                  ${NVHPC_CUDA_INCLUDE_DIR}
                  ${MPI_CXX_INCLUDE_DIRS}
                  ${NVHPC_MATH_INCLUDE_DIR}
                  )
  target_link_libraries(pHermitian PRIVATE MPI::MPI_CXX NVHPC::CUDA)
  set_target_properties(pHermitian PROPERTIES LINKER_LANGUAGE CXX)
    
  #install(TARGETS pHermitian LIBRARY DESTINATION ${CMAKE_CURRENT_LIST_DIR}/out/lib)
     
  if (PLINALG_ADD_TESTS)
    message("\
          =====================================================================
          Add a testing executable
          =====================================================================
          ")
    add_executable(plinalg_test ${CMAKE_CURRENT_LIST_DIR}/lib/src/test.cc.cu)
    target_include_directories(plinalg_test
                              PUBLIC
                                ${CMAKE_CURRENT_LIST_DIR}/lib/include
                                ${NVHPC_CUDA_INCLUDE_DIR}
                                ${MPI_CXX_INCLUDE_DIRS}
                                ${NVHPC_MATH_INCLUDE_DIR}
    )
    target_link_libraries(plinalg_test PRIVATE MPI::MPI_CXX NVHPC::CUDA pHermitian)
    set_target_properties(plinalg_test PROPERTIES LINKER_LANGUAGE CXX)

    #install(TARGETS plinalg_test RUNTIME DESTINATION ${CMAKE_CURRENT_LIST_DIR}/out/bin)

  endif()
endif()

if(SKBUILD OR COMPILE_FOR_LSP)

  message("\
          =====================================================================
          Building as a Python module to be imported as a JAX primitive
          =====================================================================
          ")
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ext/nanobind)


  nanobind_add_module(_pHermitian STABLE_ABI  
      ${CMAKE_CURRENT_LIST_DIR}/lib/src/plinalg.cc
      ${CMAKE_CURRENT_LIST_DIR}/lib/src/hermitian_kernel.cc 
      ${CMAKE_CURRENT_LIST_DIR}/lib/src/hermitian_kernel.cc.cu)

  target_include_directories(_pHermitian
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/lib/include
    ${NVHPC_CUDA_INCLUDE_DIR}
    ${MPI_CXX_INCLUDE_DIRS}
    ${NVHPC_MATH_INCLUDE_DIR}
  )

  target_link_libraries(_pHermitian PRIVATE MPI::MPI_CXX NVHPC::CUDA) 
  set_target_properties(_pHermitian PROPERTIES LINKER_LANGUAGE CXX)

  install(TARGETS _pHermitian LIBRARY DESTINATION plinalglib)
endif()
