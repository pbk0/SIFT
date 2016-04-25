cmake_minimum_required(VERSION 2.8)
project(SIFT)

message(STATUS "start running cmake...")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp helper_vigra.cpp helper_vigra.h helper_opencv.cpp helper_opencv.h)


message(STATUS "Search OpenCV...")

find_package( OpenCV REQUIRED )

if(OpenCV_FOUND)

    message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV_LIBRARIES: ${OpenCV_LIBRARIES}")
    message(STATUS "OpenCV_VERSION: ${OpenCV_VERSION}")

    include_directories(${OpenCV_INCLUDE_DIRS})

endif()

add_executable(SIFT ${SOURCE_FILES} helper_vigra.cpp helper_vigra.h helper_opencv.cpp helper_opencv.h)

if(OpenCV_FOUND)

    target_link_libraries(SIFT ${OpenCV_LIBRARIES})

endif()