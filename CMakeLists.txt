CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(spreadInterp CXX)

# lib and bin will be in top source
SET(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}")

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "7.5.0")
    message(FATAL_ERROR "Insufficient gcc version")
  endif()
endif()

# make sure we can link to openmp
find_package(OpenMP REQUIRED)

# general compiler flags
option(debug "Build with debugger" OFF)
# setting include dir and general compile flags
if(NOT debug)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Iinclude")
elseif(debug)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Iinclude -g -DDEBUG")
endif()

# include dirs
INCLUDE_DIRECTORIES(include)
FILE(GLOB h_FILES include/*.h)
SET(spreadInterp_INC_FILES ${h_FILES})
INSTALL(FILES ${spreadInterp_INC_FILES} DESTINATION include)

# setting lib src files
SET(ioSRC src/io.cpp)
SET(initSRC src/init.cpp)
SET(spreadInterpSRC src/spread_interp.cpp)
SET(benchSRC benchmark/spread_interp_bench.cpp)
SET(testSRC testing/test_spread.cpp)

# set build output paths
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# building lib
ADD_LIBRARY(spreadInterpLib STATIC ${spreadInterpSRC})
set_source_files_properties(${spreadInterpSRC} PROPERTIES COMPILE_FLAGS "-O2 -ftree-vectorize -fopt-info -fopt-info-missed -mavx -fopenmp")
ADD_LIBRARY(ioLib STATIC ${ioSRC})
set_source_files_properties(${ioSRC} PROPERTIES COMPILE_FLAGS "-O2 -ftree-vectorize -fopt-info")
ADD_LIBRARY(initLib STATIC ${initSRC})
set_source_files_properties(${initSRC} PROPERTIES COMPILE_FLAGS "-O2 -fopenmp -ftree-vectorize -fopt-info")

# install libs
INSTALL(TARGETS spreadInterpLib ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)
INSTALL(TARGETS initLib LIBRARY ARCHIVE DESTINATION lib DESTINATION lib)
INSTALL(TARGETS ioLib LIBRARY ARCHIVE DESTINATION lib DESTINATION lib)

## build executables ##

# building benchmark
ADD_EXECUTABLE(spread_interp_bench ${benchSRC})
set_source_files_properties(${benchSRC} PROPERTIES COMPILE_FLAGS "-fopenmp")
TARGET_LINK_LIBRARIES(spread_interp_bench spreadInterpLib ioLib initLib gomp)

# building test
ADD_EXECUTABLE(test_spread ${testSRC})
set_source_files_properties(${testSRC} PROPERTIES COMPILE_FLAGS "-fopenmp")
TARGET_LINK_LIBRARIES(test_spread spreadInterpLib ioLib initLib gomp)

# install benchmark
INSTALL(TARGETS spread_interp_bench RUNTIME DESTINATION bin/benchmark)

# install test
INSTALL(TARGETS test_spread RUNTIME DESTINATION bin/testing)
