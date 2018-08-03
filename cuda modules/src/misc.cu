#define _USE_MATH_DEFINES
#include <math.h>

#define THREADID \
	int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;\
	int idx = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

#define CHECKSIZE \
	if (idx >= size) { \
		return; \
	}

extern "C" { 
	__global__ void isnan_f32(float* A, int size, bool* retVal) {
		THREADID
		CHECKSIZE
		if (isnan(A[idx])) {
			*retVal = true;
		}
		return;
	}
}

extern "C" { 
	__global__ void isnan_f64(double* A, int size, bool* retVal) {
		THREADID
		CHECKSIZE
		if (isnan(A[idx])) {
			*retVal = true;
		}
		return;
	}
}

extern "C" { 
	__global__ void isinf_f32(float* A, int size, bool* retVal) {
		THREADID
		CHECKSIZE
		if (isinf(A[idx])) {
			*retVal = true;
		}
		return;
	}
}


extern "C" { 
	__global__ void isinf_f64(double* A, int size, bool* retVal) {
		THREADID
		CHECKSIZE
		if (isinf(A[idx])) {
			*retVal = true;
		}
		return;
	}
}

/*
*/