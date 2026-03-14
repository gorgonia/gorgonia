// +build cuda

package tests

/*
// CGO flags are set by the test runner script (run_direct_test.sh)
// via CGO_CFLAGS and CGO_LDFLAGS environment variables

#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>
#include <stdlib.h>

// Test basic CUDA initialization
int testCudaInit() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        printf("CUDA Error: cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    printf("✓ CUDA Runtime initialized successfully\n");
    printf("  Found %d CUDA device(s)\n", deviceCount);
    return 0;
}

// Test cuDNN context creation
int testCudnnInit() {
    cudnnHandle_t cudnn;
    cudnnStatus_t status;

    printf("Creating cuDNN handle...\n");
    status = cudnnCreate(&cudnn);

    if (status != CUDNN_STATUS_SUCCESS) {
        printf("✗ cuDNN Error: cudnnCreate failed with status %d\n", status);
        printf("  Error string: %s\n", cudnnGetErrorString(status));
        return -1;
    }

    printf("✓ cuDNN handle created successfully\n");

    // Get cuDNN version
    size_t version = cudnnGetVersion();
    printf("  cuDNN version: %zu\n", version);

    // Clean up
    status = cudnnDestroy(cudnn);
    if (status != CUDNN_STATUS_SUCCESS) {
        printf("✗ cuDNN Error: cudnnDestroy failed with status %d\n", status);
        return -1;
    }

    printf("✓ cuDNN handle destroyed successfully\n");
    return 0;
}

// Test CUDA device properties
int testDeviceProps() {
    struct cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);

    if (err != cudaSuccess) {
        printf("CUDA Error: cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    printf("✓ CUDA Device 0: %s\n", prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Total global memory: %.2f GB\n", prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
    return 0;
}

// Run all tests
int runCudaTests() {
    printf("========================================\n");
    printf("Direct CUDA/cuDNN Test (bypassing gorgonia/cu)\n");
    printf("========================================\n\n");

    printf("Test 1: CUDA Runtime Initialization\n");
    if (testCudaInit() != 0) {
        return 1;
    }
    printf("\n");

    printf("Test 2: CUDA Device Properties\n");
    if (testDeviceProps() != 0) {
        return 1;
    }
    printf("\n");

    printf("Test 3: cuDNN Context Creation\n");
    if (testCudnnInit() != 0) {
        return 1;
    }
    printf("\n");

    printf("========================================\n");
    printf("All tests passed!\n");
    printf("========================================\n");
    return 0;
}
*/
import "C"
import (
	"testing"
)

func TestDirectCUDAandCuDNN(t *testing.T) {
	result := C.runCudaTests()
	if result != 0 {
		t.Fatalf("Direct CUDA/cuDNN tests failed with code %d", result)
	}
}
