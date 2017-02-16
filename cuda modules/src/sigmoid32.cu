#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif


__global__ void sigmoid32(float* A, int size)
{
	int idx = threadIdx.x;
	if (idx >= size) {
		return;
	}
	A[idx] = 1 / (1 + pow((float)(M_E), (-1 * A[idx])));
}
	
#ifdef __cplusplus
}
#endif