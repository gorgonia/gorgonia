#ifdef __cplusplus
extern "C" {
#endif


__global__ void mul32(float* A, float* B, int size)
{
	int idx = threadIdx.x;
	if (idx >= size) {
		return;
	}
	A[idx] = A[idx] * B[idx]; 
}
	
#ifdef __cplusplus
}
#endif