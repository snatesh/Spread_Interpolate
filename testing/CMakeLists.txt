cmake_minimum_required(VERSION 2.8)
project(spreadInterp)

message(STATUS "ACCESSING TEST DIRECTORY: ${PROJECT_SOURCE_DIR}")

set(TEST_DATA ${CMAKE_CURRENT_LIST_DIR}/test_data)

add_test(NAME spreadInterpNoPBC COMMAND testSpreadInterpNoPBC ${TEST_DATA}/nopbc/particles.txt ${TEST_DATA}/nopbc/forces.txt ${TEST_DATA}/nopbc/spread.txt ${TEST_DATA}/nopbc/interp.txt)
add_test(NAME spreadInterpPBC COMMAND testSpreadInterpPBC ${TEST_DATA}/pbc/particles.txt ${TEST_DATA}/pbc/forces.txt ${TEST_DATA}/pbc/spread.txt ${TEST_DATA}/pbc/interp.txt)
