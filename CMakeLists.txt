cmake_minimum_required(VERSION 3.16)
project(LinearAlgebra)

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_COMPILER g++-10)

set(algebra_src
        ${LinearAlgebra_SOURCE_DIR}/LinearAlgebra/Matrix/Matrix.h
        ${LinearAlgebra_SOURCE_DIR}/LinearAlgebra/SynkhornProcess/SynkhornProcess.h)

add_executable(LinearAlgebra main.cpp ${algebra_src})

add_subdirectory(Tests)