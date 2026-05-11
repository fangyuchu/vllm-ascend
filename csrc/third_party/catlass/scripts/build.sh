#!/bin/bash

# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

SCRIPT_PATH=$(dirname "$(realpath "$0")")
CMAKE_SOURCE_PATH=$(realpath "$SCRIPT_PATH"/..)

CMAKE_BUILD_PATH=$CMAKE_SOURCE_PATH/build

OUTPUT_PATH=$CMAKE_SOURCE_PATH/output

if [[ $# -eq 0 ]]; then
    echo "Usage: bash build.sh [--clean] [target]"
    exit 0
fi

TARGET=${!#}
echo "Target is: $TARGET"
CMAKE_BUILD_TYPE=Release

mkdir -p "$CMAKE_BUILD_PATH"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)
            rm -rf build
            rm -rf output
            ;;
        --debug)
            echo "Hint: only python extension support debug mode."
            CMAKE_BUILD_TYPE=Debug
            ;;
        --*)
            echo "Unknown option: $1"
            ;;
    esac
    shift
done

function build_shared_lib() {
    cd "$CMAKE_SOURCE_PATH"/examples/shared_lib || exit
    rm -rf build
    cmake --no-warn-unused-cli -B build -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" -DCMAKE_INSTALL_PREFIX="$OUTPUT_PATH"/shared_lib -DCATLASS_INCLUDE_DIR="$CMAKE_SOURCE_PATH"/include
    cmake --build build -j
    cmake --install build
    cd "$CMAKE_SOURCE_PATH" || exit
}

function build_torch_library() {
    cd "$CMAKE_SOURCE_PATH"/examples/python_extension || exit
    rm -rf build
    cmake --no-warn-unused-cli -B build -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" -DCMAKE_INSTALL_PREFIX="$OUTPUT_PATH"/python_extension -DCATLASS_INCLUDE_DIR="$CMAKE_SOURCE_PATH"/include -DPython3_EXECUTABLE="$(which python3)" -DBUILD_TORCH_LIB=True
    cmake --build build -j
    cmake --install build
    cd "$CMAKE_SOURCE_PATH" || exit
}

function build_python_extension() {
    cd "$CMAKE_SOURCE_PATH"/examples/python_extension || exit
    rm -rf build
    python3 setup.py bdist_wheel --dist-dir "$OUTPUT_PATH"/python_extension
    cd "$CMAKE_SOURCE_PATH" || exit
}

if [[ "$TARGET" == "shared_lib" ]]; then
    build_shared_lib
elif [[  "$TARGET" == "lib_cmake" ]]; then
    cmake -DENABLE_LIB=ON -S "$CMAKE_SOURCE_PATH" -B "$CMAKE_BUILD_PATH"
    cmake --build "$CMAKE_BUILD_PATH"
elif [[ "$TARGET" == "python_extension" ]]; then
    build_python_extension
elif [[ "$TARGET" == "torch_library" ]]; then
    build_torch_library
else
    cmake --no-warn-unused-cli -S"$CMAKE_SOURCE_PATH" -B"$CMAKE_BUILD_PATH"
    cmake --build "$CMAKE_BUILD_PATH" --target "$TARGET" -j
fi