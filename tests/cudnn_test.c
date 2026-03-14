/*
 * Standalone CUDA/cuDNN Test (Pure C)
 *
 * This tests CUDA and cuDNN initialization directly without any Go code.
 * Compile: gcc -o cudnn_test cudnn_test.c -I/usr/local/cuda-11.4/include -L/usr/local/cuda-11.4/lib64 -lcuda -lcudart -lcudnn
 * Run: ./cudnn_test
 */

#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("========================================\n");
    printf("Standalone CUDA/cuDNN Test (Pure C)\n");
    printf("========================================\n\n");

    // Test 1: CUDA Runtime
    printf("Test 1: CUDA Runtime Initialization\n");
    int deviceCount = 0;
    cudaError_t cudaErr = cudaGetDeviceCount(&deviceCount);

    if (cudaErr != cudaSuccess) {
        printf("✗ CUDA Error: cudaGetDeviceCount failed: %s\n", cudaGetErrorString(cudaErr));
        return 1;
    }

    printf("✓ CUDA Runtime initialized successfully\n");
    printf("  Found %d CUDA device(s)\n", deviceCount);
    printf("\n");

    // Test 2: Device Properties
    printf("Test 2: CUDA Device Properties\n");
    struct cudaDeviceProp prop;
    cudaErr = cudaGetDeviceProperties(&prop, 0);

    if (cudaErr != cudaSuccess) {
        printf("✗ CUDA Error: cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(cudaErr));
        return 1;
    }

    printf("✓ Device 0: %s\n", prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Total global memory: %.2f GB\n", prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
    printf("\n");

    // Test 3: cuDNN Context
    printf("Test 3: cuDNN Context Creation\n");
    cudnnHandle_t cudnn;
    cudnnStatus_t cudnnStatus;

    printf("  Creating cuDNN handle...\n");
    cudnnStatus = cudnnCreate(&cudnn);

    if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
        printf("✗ cuDNN Error: cudnnCreate failed with status %d\n", cudnnStatus);
        printf("  Error: %s\n", cudnnGetErrorString(cudnnStatus));
        printf("\n");
        printf("This indicates cuDNN cannot initialize on your system.\n");
        printf("Possible causes:\n");
        printf("  - Driver too old (you have 470, may need 510+)\n");
        printf("  - cuDNN version incompatible with driver\n");
        printf("  - Missing CUDA context initialization\n");
        return 1;
    }

    printf("✓ cuDNN handle created successfully\n");

    // Get cuDNN version
    size_t version = cudnnGetVersion();
    printf("  cuDNN version: %zu (major.minor.patch: %zu.%zu.%zu)\n",
           version, version/1000, (version%1000)/100, version%100);

    // Clean up
    cudnnStatus = cudnnDestroy(cudnn);
    if (cudnnStatus != CUDNN_STATUS_SUCCESS) {
        printf("✗ cuDNN Error: cudnnDestroy failed with status %d\n", cudnnStatus);
        return 1;
    }

    printf("✓ cuDNN handle destroyed successfully\n");
    printf("\n");

    printf("========================================\n");
    printf("✓ All tests passed!\n");
    printf("  CUDA and cuDNN are working correctly.\n");
    printf("  The issue is likely in gorgonia/cu bindings.\n");
    printf("========================================\n");

    return 0;
}
