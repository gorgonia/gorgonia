#!/bin/bash

# CUDA 11.4 Test Script

echo "Setting up CUDA 11.4 environment..."
export CUDA_HOME=/usr/local/cuda-11.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CGO_CFLAGS="-I$CUDA_HOME/include"
export CGO_LDFLAGS="-L$CUDA_HOME/lib64 -lcuda -lcudart -lcublas -lcudnn"
export GODEBUG=gotypesalias=0
export ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.23

echo "Environment:"
echo "  CUDA_HOME=$CUDA_HOME"
echo "  LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo ""

echo "Running CUDA tests..."
go test -tags=cuda -v . 2>&1
