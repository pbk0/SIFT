cmake_minimum_required(VERSION 2.8)
project(SIFT)

message(STATUS "start running cmake...")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp helper_vigra.cpp helper_vigra.h)


message(STATUS "Search OpenCV...")

find_package( OpenCV REQUIRED )

if(OpenCV_FOUND)

    message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV_LIBRARIES: ${OpenCV_LIBRARIES}")
    message(STATUS "OpenCV_VERSION: ${OpenCV_VERSION}")

    include_directories(${OpenCV_INCLUDE_DIRS})

endif()

find_package(Doxygen)

if(DOXYGEN_FOUND)

    set(DOXYGEN_INPUT ${SOURCE_FILES})
    set(DOXYGEN_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    add_custom_command(
            OUTPUT ${DOXYGEN_OUTPUT}
            COMMAND ${CMAKE_COMMAND} -E echo_append "Building API Documentation..."
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_INPUT}
            COMMAND ${CMAKE_COMMAND} -E echo "Done."
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            DEPENDS ${DOXYGEN_INPUT}
    )

    add_custom_target(apidoc ALL DEPENDS ${DOXYGEN_OUTPUT})

    add_custom_target(apidoc_forced
            COMMAND ${CMAKE_COMMAND} -E echo_append "Building API Documentation..."
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_INPUT}
            COMMAND ${CMAKE_COMMAND} -E echo "Done."
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            )

endif(DOXYGEN_FOUND)

add_executable(SIFT ${SOURCE_FILES} )

if(OpenCV_FOUND)

    target_link_libraries(SIFT ${OpenCV_LIBRARIES})

endif()