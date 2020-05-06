cmake_minimum_required(VERSION 2.8)
project(spreadInterp CXX)

# lib and bin will be in top source
set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}")

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "7.5.0")
    message(FATAL_ERROR "Insufficient gcc version")
  endif()
endif()

# make sure we can link to openmp
find_package(OpenMP REQUIRED)

# general compiler flags
option(debug "Build with debugger" ON)
option(test "Build all tests" ON)
option(sepxyz "Separate XYZ" ON)


# setting include dir and general compile flags
if(NOT debug)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Iinclude")
elseif(debug)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Iinclude -g -DDEBUG")
endif()

if(sepxyz)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSEPXYZ")
else()
  message(STATUS "Not using separate x y z")
endif()

# include dirs
include_directories(include)
file(GLOB h_FILES include/*.h)
set(spreadInterp_INC_FILES ${h_FILES})
install(FILES ${spreadInterp_INC_FILES} DESTINATION include)

# setting lib src files
set(ioSRC src/io.cpp)
set(initSRC src/init.cpp)
set(spreadInterpSRC src/spread_interp.cpp)
set(benchSRC benchmark/spread_interp_bench.cpp)
set(benchweakSRC benchmark/spread_interp_bench_weak.cpp)
set(testSRC testing/test_spread.cpp)

# set build output paths
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# building lib
add_library(spreadInterpLib STATIC ${spreadInterpSRC})
set_source_files_properties(${spreadInterpSRC} PROPERTIES COMPILE_FLAGS "-Ofast -ftree-vectorize -fopt-info -mavx -fopenmp")
add_library(ioLib STATIC ${ioSRC})
set_source_files_properties(${ioSRC} PROPERTIES COMPILE_FLAGS "-Ofast -ftree-vectorize")
add_library(initLib STATIC ${initSRC})
set_source_files_properties(${initSRC} PROPERTIES COMPILE_FLAGS "-Ofast -fopenmp -ftree-vectorize")

# install libs
install(TARGETS spreadInterpLib ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)
install(TARGETS initLib LIBRARY ARCHIVE DESTINATION lib DESTINATION lib)
install(TARGETS ioLib LIBRARY ARCHIVE DESTINATION lib DESTINATION lib)

## build executables ##

# building benchmark
add_executable(spread_interp_bench ${benchSRC})
set_source_files_properties(${benchSRC} PROPERTIES COMPILE_FLAGS "-fopenmp")
target_link_libraries(spread_interp_bench spreadInterpLib ioLib initLib gomp)

add_executable(spread_interp_bench_weak ${benchweakSRC})
set_source_files_properties(${benchweakSRC} PROPERTIES COMPILE_FLAGS "-fopenmp")
target_link_libraries(spread_interp_bench_weak spreadInterpLib ioLib initLib gomp)
  
# building test
add_executable(test_spread ${testSRC})
set_source_files_properties(${testSRC} PROPERTIES COMPILE_FLAGS "-fopenmp")
target_link_libraries(test_spread spreadInterpLib ioLib initLib gomp)


# install benchmark
install(TARGETS spread_interp_bench RUNTIME DESTINATION bin/benchmark)
install(TARGETS spread_interp_bench_weak RUNTIME DESTINATION bin/benchmark)

# install exec for test data creation
install(TARGETS test_spread RUNTIME DESTINATION bin/testing)


if (test)
  # Download and unpack googletest at configure time
  configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
  execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
  if(result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
  endif()
  execute_process(COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
  if(result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
  endif()
  
  # Prevent overriding the parent project's compiler/linker
  # settings on Windows
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  
  # Add googletest directly to our build. This defines
  # the gtest and gtest_main targets.
  add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                   ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
                   EXCLUDE_FROM_ALL)
  
  # The gtest/gtest_main targets carry header search path
  # dependencies automatically when using CMake 2.8.11 or
  # later. Otherwise we have to add them here ourselves.
  if (CMAKE_VERSION VERSION_LESS 2.8.11)
    include_directories(${gtest_SOURCE_DIR}/include)
  endif()

  enable_testing()
  
  set(OLD_RUNTIME_OUTPUT_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/testing)
  add_executable(testSpreadInterpNoPBC ${CMAKE_SOURCE_DIR}/testing/spreadInterpNoPBC.cpp)
  add_executable(testSpreadInterpPBC ${CMAKE_SOURCE_DIR}/testing/spreadInterpPBC.cpp)
  target_link_libraries(testSpreadInterpNoPBC spreadInterpLib ioLib initLib gomp gtest_main)
  target_link_libraries(testSpreadInterpPBC spreadInterpLib ioLib initLib gomp gtest_main)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OLD_RUNTIME_OUTPUT_DIR})
  add_subdirectory(testing)
  install(TARGETS testSpreadInterpNoPBC RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/testing)
  install(TARGETS testSpreadInterpPBC RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/testing)
endif()
