#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif


__global__ void square64(double* A, int size)
{
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += blockDim.x * gridDim.x) {
		printf("A %p | idx: %d | size %d | blockIdx.x %d | blockDim.x %d | threadIdx.x %d | gridDim.x %d\n", A, idx, size,  blockIdx.x, blockDim.x, threadIdx.x, gridDim.x);
		A[idx] = (double)(idx); 
	}
}
	
#ifdef __cplusplus
}
#endif