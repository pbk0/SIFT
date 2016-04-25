cmake_minimum_required(VERSION 2.8)
project(SIFT)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp helper_vigra.cpp helper_vigra.h helper_opencv.cpp helper_opencv.h)
add_executable(SIFT ${SOURCE_FILES} helper_vigra.cpp helper_vigra.h helper_opencv.cpp helper_opencv.h)