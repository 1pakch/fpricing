cmake_minimum_required(VERSION 2.8)
project(fpricing)

set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "") # disables -rdynamic

# enables init list constructors for Eigen objects
# the header file below is a symlink to
#    extlibs/eigen-initializer_list/src/Matrix_initializer_list.h
include_directories(include)
add_definitions(-DEIGEN_MATRIX_PLUGIN="fpricing/misc/eigen-init-list-plugin.h")

# library dependencies
find_package(Boost COMPONENTS unit_test_framework REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# compiler flags
set(CMAKE_CXX_COMPILER g++-5)
if(UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++14")
endif()

add_subdirectory(lib)
add_subdirectory(test)
enable_testing ()

