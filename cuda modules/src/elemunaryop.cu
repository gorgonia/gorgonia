#define _USE_MATH_DEFINES
#include <math.h>

#define THREADID \
	int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;\
	int idx = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

#define CHECKSIZE \
	if (idx >= size) { \
		return; \
	}

#define UNARYOP(name, t, type, op)\
	__global__ void name ##_##t (type* A, int size) { \
		THREADID \
		CHECKSIZE \
		A[idx] = op(A[idx]); \
	}


extern "C" { UNARYOP(cos, f64, double, cos) }
extern "C" { UNARYOP(sin, f64, double, sin) }
extern "C" { UNARYOP(exp, f64, double, exp) }
extern "C" { UNARYOP(ln, f64, double, log) }
extern "C" { UNARYOP(log2, f64, double, log2) }
extern "C" { UNARYOP(sqrt, f64, double, sqrt) }
extern "C" { UNARYOP(tanh, f64, double, tanh) }
extern "C" { UNARYOP(cbrt, f64, double, cbrt) }
extern "C" { UNARYOP(log1p, f64, double, log1p) }
extern "C" { UNARYOP(expm1, f64, double, expm1) }

// un-differentiable
extern "C" { UNARYOP(abs, f64, double, abs) }
extern "C" { UNARYOP(ceil, f64, double, ceil) }
extern "C" { UNARYOP(floor, f64, double, floor) }


extern "C" {
	__global__ void sign_f64(double* A, int size) {
		THREADID
		CHECKSIZE
		A[idx] = (A[idx] > 0.0) - (A[idx] < 0.0);
	}
}

extern "C" {
	__global__ void square_f64(double* A, int size) {
		THREADID
		CHECKSIZE
		A[idx] = A[idx] * A[idx];
	}
}

extern "C" {
	__global__ void cube_f64(double* A, int size) {
		THREADID
		CHECKSIZE
		A[idx] = A[idx] * A[idx] * A[idx];
	}
}

extern "C" {
	__global__ void neg_f64(double* A, int size) {
		THREADID
		CHECKSIZE
		A[idx] = -A[idx];
	}
}

extern "C" {
	__global__ void inverse_f64(double* A, int size) {
		THREADID
		CHECKSIZE
		A[idx] = 1.0/A[idx];
	}
}

extern "C" {
	__global__ void softplus_f64(double* A, int size) {
		THREADID
		CHECKSIZE
		if (A[idx] < -708.0) {
			A[idx] = 0.0;
		} else if (A[idx] > 16.0) {
			// no op
		} else {
			A[idx] = log1p(exp(A[idx]));
		}
	}
}

extern "C" {
	__global__ void sigmoid_f64(double* A, int size) {
		THREADID
		CHECKSIZE
		if (A[idx] < -709.0) {
			A[idx] = 0.0;
		} else if (A[idx] > 19.0) {
			A[idx] = 1.0;
		} else {
			A[idx] = 1.0 / (1.0 + exp(-A[idx]));
		}
		// alternative sigmoid function:
		// A[idx] = 1 / (1 + pow(M_E, (double)(-1 * A[idx])));
	}
}

extern "C" {
	__global__ void invsqrt_f64(double* A, int size) {
		THREADID
		CHECKSIZE
		A[idx] = 1.0/sqrt(A[idx]);
	}
}

/* FLOAT32 */

extern "C" { UNARYOP(cos, f32, float, cosf) }
extern "C" { UNARYOP(sin, f32, float, sinf) }
extern "C" { UNARYOP(exp, f32, float, expf) }
extern "C" { UNARYOP(ln, f32, float, logf) }
extern "C" { UNARYOP(log2, f32, float, log2f) }
extern "C" { UNARYOP(sqrt, f32, float, sqrtf) }
extern "C" { UNARYOP(tanh, f32, float, tanhf) }
extern "C" { UNARYOP(cbrt, f32, float, cbrtf) }
extern "C" { UNARYOP(log1p, f32, float, log1pf) }
extern "C" { UNARYOP(expm1, f32, float, expm1f) }

// un-differentiable
extern "C" { UNARYOP(abs, f32, float, abs) }
extern "C" { UNARYOP(ceil, f32, float, ceilf) }
extern "C" { UNARYOP(floor, f32, float, floorf) }


extern "C" {
	__global__ void sign_f32(float* A, int size) {
		THREADID
		CHECKSIZE
		A[idx] = (A[idx] > 0.0f) - (A[idx] < 0.0f);
	}
}

extern "C" {
	__global__ void square_f32(float* A, int size) {
		THREADID
		CHECKSIZE
		A[idx] = A[idx] * A[idx];
	}
}

extern "C" {
	__global__ void cube_f32(float* A, int size) {
		THREADID
		CHECKSIZE
		A[idx] = A[idx] * A[idx] * A[idx];
	}
}

extern "C" {
	__global__ void neg_f32(float* A, int size) {
		THREADID
		CHECKSIZE
		A[idx] = -A[idx];
	}
}

extern "C" {
	__global__ void inverse_f32(float* A, int size) {
		THREADID
		CHECKSIZE
		A[idx] = 1.0f/A[idx];
	}
}

extern "C" {
	__global__ void softplus_f32(float* A, int size) {
		THREADID
		CHECKSIZE
		if (A[idx] < -103.0f) {
			A[idx] = 0.0f;
		} else if (A[idx] > 14.0f) {
			// no op
		} else {
			A[idx] = log1pf(expf(A[idx]));
		}
	}
}

extern "C" {
	__global__ void sigmoid_f32(float* A, int size) {
		THREADID
		CHECKSIZE
		if (A[idx] < -88.0f) {
			A[idx] = 0.0f;
		} else if (A[idx] > 15.0f) {
			A[idx] = 1.0f;
		} else {
			A[idx] = 1.0f / (1.0f + expf(-A[idx]));
		}
		// alternative sigmoid function:
		// A[idx] = 1 / (1 + powf((float)(M_E), (-1 * A[idx])));
	}
}

extern "C" {
	__global__ void invsqrt_f32(float* A, int size) {
		THREADID
		CHECKSIZE
		A[idx] = 1.0/sqrtf(A[idx]);
	}
}
