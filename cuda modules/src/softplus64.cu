#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif


__global__ void softplus64(double* A, int size)
{
	int idx = threadIdx.x;
	if (idx >= size) {
		return;
	}
	A[idx] = log1p(exp(A[idx]));
}
	
#ifdef __cplusplus
}
#endif