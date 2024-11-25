cmake_minimum_required(VERSION 3.0)
# project name
project(RepulsiveShapeSpaceII)

################################################################################
# general settings
################################################################################

# use C++20 standard to build this project
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# produce position independent code for ALL targets (often necessary for shared libraries)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

message(STATUS "CMAKE_CXX_COMPILER: " ${BoldBlue} ${CMAKE_CXX_COMPILER_ID} ${ColourReset})
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "AppleClang" OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++17 -stdlib=libc++ -D_LIBCPP_ENABLE_CXX17_REMOVED_FEATURES")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()
add_compile_options(-Warray-bounds)

################################################################################
# Architecture optimizations
################################################################################
option(OPTIMIZE_FOR_NATIVE "Build with -march=native" OFF)
if (OPTIMIZE_FOR_NATIVE)
  include(CheckCXXCompilerFlag)
  CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
  if (COMPILER_SUPPORTS_MARCH_NATIVE)
    add_compile_options("-march=native")
  endif ()
endif ()


################################################################################
# Load GOAST
################################################################################
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/deps/GOAST/cmake)
include(GOAST)

################################################################################
# Use MKL in the examples etc
################################################################################
option(GOAST_WITH_MKL "Use Intel's Math Kernel Library" OFF)
add_library(MKL::MKL INTERFACE IMPORTED)
if (GOAST_WITH_MKL)
    target_link_options(MKL::MKL INTERFACE -m64 -Wl,--no-as-needed)

    target_compile_options(MKL::MKL INTERFACE -m64)
    target_include_directories(MKL::MKL INTERFACE $ENV{MKLROOT}/include)

    target_link_directories(MKL::MKL INTERFACE $ENV{MKLROOT}/lib/intel64)
    target_link_libraries(MKL::MKL INTERFACE mkl_intel_lp64 mkl_gnu_thread mkl_core gomp pthread m dl)
    target_compile_definitions(MKL::MKL INTERFACE GOAST_WITH_MKL)
endif (GOAST_WITH_MKL)

################################################################################
# own modules
################################################################################
set(Boost_USE_STATIC_LIBS OFF) # must be OFF if one wants to build shared libraries
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost REQUIRED COMPONENTS system filesystem)

# YAML
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "Enable testing")
add_subdirectory(deps/yaml-cpp)

# Henrik's previous repulsion library
add_library(Repulsion::Repulsion INTERFACE IMPORTED)
target_include_directories(Repulsion::Repulsion INTERFACE deps/repulsion)
target_link_libraries(Repulsion::Repulsion INTERFACE MKL::MKL)

# OpenMP
find_package(OpenMP)

#for cmake version less than 3.9 there is no find module, so we need to define the target ourselves
if (NOT TARGET OpenMP::OpenMP_CXX)
    if (${CMAKE_CXX_COMPILER_ID} STREQUAL "AppleClang" OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
        message(WARNING "OpenMP detection might now work with AppleClang, Intel")
    endif ()
    find_package(Threads REQUIRED)
    add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
    set_property(TARGET OpenMP::OpenMP_CXX
            PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
    set_property(TARGET OpenMP::OpenMP_CXX
            PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS} Threads::Threads)
endif ()

target_link_libraries(GOAST INTERFACE OpenMP::OpenMP_CXX)
target_compile_definitions(GOAST INTERFACE GOAST_WITH_OPENMP)

# PQP
add_subdirectory(deps/PQP)

################################################################################
# own directories
################################################################################
include_directories(include)

################################################################################
# own targets
################################################################################
add_executable(EvaluateTPE src/EvaluateTPE.cpp)
target_link_libraries(EvaluateTPE PUBLIC GOAST::All)
target_link_libraries(EvaluateTPE PUBLIC Boost::boost Boost::filesystem)
target_link_libraries(EvaluateTPE PUBLIC yaml-cpp)
target_link_libraries(EvaluateTPE PUBLIC MKL::MKL)
target_link_libraries(EvaluateTPE PUBLIC Repulsion::Repulsion)
target_link_libraries(EvaluateTPE PUBLIC pqp)

add_executable(AugmentedSurfaceInterpolation src/AugmentedSurfaceInterpolation.cpp)
target_link_libraries(AugmentedSurfaceInterpolation PUBLIC GOAST::All)
target_link_libraries(AugmentedSurfaceInterpolation PUBLIC Boost::boost Boost::filesystem)
target_link_libraries(AugmentedSurfaceInterpolation PUBLIC yaml-cpp)
target_link_libraries(AugmentedSurfaceInterpolation PUBLIC MKL::MKL)
target_link_libraries(AugmentedSurfaceInterpolation PUBLIC Repulsion::Repulsion)
target_link_libraries(AugmentedSurfaceInterpolation PUBLIC pqp)
target_compile_definitions(AugmentedSurfaceInterpolation PUBLIC EIGEN_DONT_PARALLELIZE)


add_executable(ElasticSurfaceExtrapolation src/ElasticSurfaceExtrapolation.cpp)
target_link_libraries(ElasticSurfaceExtrapolation PUBLIC GOAST::All)
target_link_libraries(ElasticSurfaceExtrapolation PUBLIC Boost::boost Boost::filesystem)
target_link_libraries(ElasticSurfaceExtrapolation PUBLIC yaml-cpp)
target_link_libraries(ElasticSurfaceExtrapolation PUBLIC MKL::MKL)
target_link_libraries(ElasticSurfaceExtrapolation PUBLIC Repulsion::Repulsion)
target_link_libraries(ElasticSurfaceExtrapolation PUBLIC pqp)
target_compile_definitions(ElasticSurfaceExtrapolation PUBLIC EIGEN_DONT_PARALLELIZE)

add_executable(AugmentedSurfaceExtrapolation src/AugmentedSurfaceExtrapolation.cpp)
target_link_libraries(AugmentedSurfaceExtrapolation PUBLIC GOAST::All)
target_link_libraries(AugmentedSurfaceExtrapolation PUBLIC Boost::boost Boost::filesystem)
target_link_libraries(AugmentedSurfaceExtrapolation PUBLIC yaml-cpp)
target_link_libraries(AugmentedSurfaceExtrapolation PUBLIC MKL::MKL)
target_link_libraries(AugmentedSurfaceExtrapolation PUBLIC Repulsion::Repulsion)
target_link_libraries(AugmentedSurfaceExtrapolation PUBLIC pqp)
target_compile_definitions(AugmentedSurfaceExtrapolation PUBLIC EIGEN_DONT_PARALLELIZE)

add_executable(ShapeFromMetric src/ShapeFromMetric.cpp)
target_link_libraries(ShapeFromMetric PUBLIC GOAST::All)
target_link_libraries(ShapeFromMetric PUBLIC Boost::boost Boost::filesystem)
target_link_libraries(ShapeFromMetric PUBLIC yaml-cpp)
target_link_libraries(ShapeFromMetric PUBLIC MKL::MKL)
target_link_libraries(ShapeFromMetric PUBLIC Repulsion::Repulsion)
target_link_libraries(ShapeFromMetric PUBLIC pqp)

add_executable(RepulsiveMeans src/RepulsiveMeans.cpp)
target_link_libraries(RepulsiveMeans PUBLIC GOAST::All)
target_link_libraries(RepulsiveMeans PUBLIC Boost::boost Boost::filesystem)
target_link_libraries(RepulsiveMeans PUBLIC yaml-cpp)
target_link_libraries(RepulsiveMeans PUBLIC MKL::MKL)
target_link_libraries(RepulsiveMeans PUBLIC Repulsion::Repulsion)
target_link_libraries(RepulsiveMeans PUBLIC pqp)

add_executable(Prolongation src/Prolongation.cpp)
target_link_libraries(Prolongation PUBLIC GOAST::All)
