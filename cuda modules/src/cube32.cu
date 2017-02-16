#ifdef __cplusplus
extern "C" {
#endif


__global__ void cube32(float* A, int size)
{
	int idx = threadIdx.x;
	if (idx >= size) {
		return;
	}
	A[idx] = A[idx] * A[idx] * A[idx]; 
}
	
#ifdef __cplusplus
}
#endif