#!/bin/bash

# Direct CUDA/cuDNN Test Runner

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
export CGO_CFLAGS="-I$CUDA_HOME/include"
export CGO_LDFLAGS="-L$CUDA_HOME/lib64 -lcuda -lcudart -lcudnn"
export GODEBUG=gotypesalias=0
export ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.23

echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo ""

echo "Running direct CUDA/cuDNN test (bypassing gorgonia/cu)..."
echo ""

cd "$(dirname "$0")"
go test -tags=cuda -v -run TestDirectCUDAandCuDNN .
