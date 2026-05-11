# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(python_extension)

set(CMAKE_SKIP_RPATH TRUE)

set(CATLASS_SHARED_LIB_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../shared_lib)
set(CATLASS_SHARED_LIB_INCLUDE_DIR ${CATLASS_SHARED_LIB_SRC_DIR}/include)
set(CATLASS_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/../../include)
add_subdirectory(${CATLASS_SHARED_LIB_SRC_DIR} shared_lib)

if(NOT DEFINED Python3_EXECUTABLE)
    message(WARNING "Python3_EXECUTABLE is not defined, using default python3")
else()
    message("Using python: ${Python3_EXECUTABLE}")
endif()

find_package(Python3 COMPONENTS Interpreter Development)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please run set_env.sh.")
else()
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

list(APPEND CMAKE_PREFIX_PATH "${Python3_SITELIB}")

find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)

include_directories(${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${Python3_SITELIB}/torch_npu/include
    ${ASCEND_HOME_PATH}/include
    ${CATLASS_SHARED_LIB_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/include)

link_directories(${ASCEND_HOME_PATH}/lib64
    ${CMAKE_BINARY_DIR}
    ${Python3_SITELIB}/torch_npu/lib)

link_libraries(libcatlass_kernel.a "${TORCH_LIBRARIES}" torch_npu tiling_api)

add_compile_options(
    -fstack-protector-strong)
add_link_options(-Wl,-z,relro,-z,now -s)

add_library(catlass_kernel_wrapper OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/src/wrapper/catlass_kernel_wrapper.cpp)
target_compile_options(catlass_kernel_wrapper PRIVATE -fPIC)
add_dependencies(catlass_kernel_wrapper catlass_kernel)

if(BUILD_PYBIND)
    set(PYBIND_BINDINGS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/bindings/pybind_bindings.cpp)
    pybind11_add_module(_C
        SHARED ${PYBIND_BINDINGS_SRC} $<TARGET_OBJECTS:catlass_kernel_wrapper>)
    add_dependencies(_C catlass_kernel_wrapper)
    install(TARGETS _C DESTINATION .)
endif()

if(BUILD_TORCH_LIB)
    set(TORCH_BINDINGS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/bindings/torch_bindings.cpp)
    add_library(catlass_torch
        SHARED ${TORCH_BINDINGS_SRC} $<TARGET_OBJECTS:catlass_kernel_wrapper>)
    add_dependencies(catlass_torch catlass_kernel_wrapper)
    install(TARGETS catlass_torch DESTINATION .)
endif()
