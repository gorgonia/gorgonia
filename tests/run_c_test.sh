#!/bin/bash

# Compile and run standalone C test

cd "$(dirname "$0")"

# Use system CUDA configuration if set, otherwise use default symlink
if [[ -z "$CUDA_HOME" ]]; then
    export CUDA_HOME=/usr/local/cuda
fi

# Detect CUDA version
CUDA_VERSION=$($CUDA_HOME/bin/nvcc --version 2>/dev/null | grep "release" | awk '{print $5}' | cut -d',' -f1 || echo "unknown")

echo "Setting up CUDA environment..."
echo "  CUDA_HOME: $CUDA_HOME"
echo "  CUDA Version: $CUDA_VERSION"

export PATH=$CUDA_HOME/bin:$PATH
# Set clean LD_LIBRARY_PATH to avoid conflicts with other CUDA versions
export LD_LIBRARY_PATH=$CUDA_HOME/lib64

echo "Compiling standalone C test..."
gcc -o cudnn_test_exe cudnn_test.c \
    -I$CUDA_HOME/include \
    -L$CUDA_HOME/lib64 \
    -lcuda -lcudart -lcudnn \
    -Wl,-rpath,$CUDA_HOME/lib64

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

echo "Compilation successful!"
echo ""
echo "Running test..."
echo ""

./cudnn_test_exe
exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "Test completed successfully!"
else
    echo "Test failed with exit code $exit_code"
fi

# Clean up
rm -f cudnn_test_exe

exit $exit_code
