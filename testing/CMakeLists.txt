cmake_minimum_required(VERSION 2.8)
project(spreadInterp)

message(STATUS "ACCESSING TEST DIRECTORY: ${PROJECT_SOURCE_DIR}")

set(TEST_DATA ${CMAKE_CURRENT_LIST_DIR}/test_data)

add_test(NAME spreadInterpNoPBC COMMAND testSpreadInterp ${TEST_DATA}/nopbc/particles.txt ${TEST_DATA}/nopbc/forces.txt ${TEST_DATA}/nopbc/spread.txt ${TEST_DATA}/nopbc/interp.txt)
