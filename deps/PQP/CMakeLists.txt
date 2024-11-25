cmake_minimum_required(VERSION 3.0)
set(CMAKE_BUILD_TYPE_INIT Release)
project(PQP)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if (COMPILER_SUPPORTS_MARCH_NATIVE)
add_compile_options("-march=native")
endif ()

add_library(pqp STATIC src/BV.cpp src/PQP.cpp src/Build.cpp src/TriDist.cpp)
target_include_directories(pqp PUBLIC include)

