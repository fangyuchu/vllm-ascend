SCRIPT_PATH=$(dirname "$(realpath "$0")")
BUILD_SCRIPT_PATH=$(realpath "$SCRIPT_PATH"/../scripts/build.sh)

# example test
bash "$BUILD_SCRIPT_PATH" --clean catlass_examples || exit 1
python3 "$SCRIPT_PATH/test_example.py"

# python extension
bash "$BUILD_SCRIPT_PATH" --clean python_extension || exit 1
pip install "$SCRIPT_PATH/../output/python_extension/*.whl"
python3 "$SCRIPT_PATH/test_python_extension.py"
pip uninstall torch_catlass

# torch lib
bash "$BUILD_SCRIPT_PATH" --clean torch_library || exit 1
python3 "$SCRIPT_PATH/test_torch_lib.py"