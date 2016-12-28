cmake_minimum_required(VERSION 2.8)
project(fpricing)

set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "") # disables -rdynamic

set(CMAKE_CXX_COMPILER g++-5)
if(UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++14")
endif()

find_package(Boost COMPONENTS unit_test_framework REQUIRED)

add_subdirectory(lib)
add_subdirectory(test)
enable_testing ()

