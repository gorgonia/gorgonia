#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif


__global__ void tanh64(float* A, int size)
{
	int idx = threadIdx.x;
	if (idx >= size) {
		return;
	}
	A[idx] = tanh(A[idx]);
}
	
#ifdef __cplusplus
}
#endif