cmake_minimum_required(VERSION 3.5)
project(devkit_object)

set(CMAKE_BUILD_TYPE "Release")
find_package(Boost)
IF (Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIR})
endif()

set(Boost_USE_MULTITHREAD ON)

add_subdirectory(pybind11)
pybind11_add_module(lib_cpp lib_cpp.cpp)

set_target_properties(lib_cpp PROPERTIES PREFIX "")
set_target_properties(lib_cpp PROPERTIES OUTPUT_NAME "lib_cpp")
set_target_properties(lib_cpp PROPERTIES SUFFIX ".so")




