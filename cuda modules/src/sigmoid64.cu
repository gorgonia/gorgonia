#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif


__global__ void sigmoid64(double* A, int size)
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
	int idx = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
	if (idx >= size) {
		return;
	}
	A[idx] = 1 / (1 + pow(M_E, (double)(-1 * A[idx])));
}
	
#ifdef __cplusplus
}
#endif