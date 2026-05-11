# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set(CATLASS_EXAMPLES_COMMON_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)
set(BISHENG_COMPILER_OPTIONS
    -O2 -std=c++17 -xcce
    -Xhost-start
    -Wl,-z,relro,-z,now
    -fstack-protector-strong
    -s 
    -Xhost-end
    -mllvm -cce-aicore-stack-size=0x8000
    -mllvm -cce-aicore-function-stack-size=0x8000
    -mllvm -cce-aicore-record-overflow=true
    -mllvm -cce-aicore-addr-transform
    -mllvm -cce-aicore-dcci-insert-for-scalar=false
    -DL2_CACHE_HINT
    -I${ASCEND_HOME_PATH}/compiler/tikcpp
    -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw
    -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl
    -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface
    -I${ASCEND_HOME_PATH}/include
    -I${ASCEND_HOME_PATH}/include/experiment/runtime
    -I${ASCEND_HOME_PATH}/include/experiment/msprof
    -I${CATLASS_EXAMPLES_COMMON_SOURCE_DIR}
    -I${CMAKE_SOURCE_DIR}/include
    -L${ASCEND_HOME_PATH}/lib64
    -Wno-macro-redefined -Wno-ignored-attributes
    -lruntime -lstdc++ -lascendcl -lm -ltiling_api -lplatform -lc_sec -ldl -lnnopbase
)

if(DEFINED PROF)
    list(APPEND ${BISHENG_COMPILER_OPTIONS} -lprofapi)
endif()

file(GLOB_RECURSE CATLASS_INCLUDE_FILES ${CMAKE_SOURCE_DIR}/include/*.hpp)
file(GLOB_RECURSE CATLASS_EXAMPLES_COMMON_INCLUDE_FILES ${CATLASS_EXAMPLES_COMMON_SOURCE_DIR}/*.hpp)
add_custom_target(catlass_examples)

function(catlass_example_add_executable NAME)
    if (${NAME} STREQUAL "17_gemv_aiv")
        set(ARCH "dav-c220-vec")
    else()
        set(ARCH "dav-c220")
    endif()

    add_custom_command(
        OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME}
        COMMAND ${CMAKE_BISHENG_COMPILER} --cce-aicore-arch=${ARCH} ${BISHENG_COMPILER_OPTIONS} ${ARGN} -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME}
        DEPENDS ${ARGN} ${CATLASS_INCLUDE_FILES} ${CATLASS_EXAMPLES_COMMON_INCLUDE_FILES}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Compiling executable kernel: ${NAME}"
    )

    add_custom_target(${NAME} ALL DEPENDS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME})
    add_dependencies(catlass_examples ${NAME})
endfunction()

if(ENABLE_LIB)
    add_subdirectory(lib_cmake)
else()

foreach(EXAMPLE
    00_basic_matmul
    01_batched_matmul
    02_grouped_matmul_slice_m
    03_matmul_add
    04_padding_matmul
    05_grouped_matmul_slice_k
    06_optimized_matmul
    07_grouped_matmul_slice_m_per_token_dequant_moe
    08_grouped_matmul
    09_splitk_matmul
    10_grouped_matmul_slice_m_per_token_dequant
    11_grouped_matmul_slice_k_per_token_dequant
    12_quant_matmul
    13_basic_matmul_tla
    14_optimized_matmul_tla
    15_gemm
    16_group_gemm
    17_gemv_aiv
    18_gemv_aic
    19_mla
)
    add_subdirectory(${EXAMPLE})
endforeach()

endif()
