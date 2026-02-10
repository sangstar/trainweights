cmake_minimum_required(VERSION 3.15)
project(trainweights LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python COMPONENTS Interpreter Development REQUIRED)

execute_process(
        COMMAND ${Python_EXECUTABLE} -m nanobind --cmake_dir
        OUTPUT_VARIABLE nanobind_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(
        trainweights
        src/trainweights.cpp
        src/tensor.cpp
        src/serialize.cpp
        src/types.cpp
        src/logger.cpp
)

add_executable(test_ndarray
        src/tests/test_ndarray.cpp
        src/tensor.h
        src/tensor.cpp
        src/serialize.cpp
        src/serialize.h
        src/types.h
        src/types.cpp
        src/logger.cpp
        src/logger.h
)


target_link_libraries(test_ndarray
        PRIVATE
        nanobind-static
        Python::Python
)
