#include <math.h>

#define THREADID \
	int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;\
	int idx = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

#define CHECKSIZE \
	if (idx >= size) { \
		return; \
	}

#define VVBINOP(name, t, type, op)\
	__global__ void  name ##_vv_ ##t(type* A, type* B, int size) { \
		THREADID \
		CHECKSIZE \
		A[idx] = A[idx] op B[idx];}

#define VSBINOP(name, t, type, op)\
	__global__ void  name ##_vs_ ##t(type* A, type* B, int size) { \
		THREADID \
		CHECKSIZE \
		A[idx] = A[idx] op B[0];}

#define SVBINOP(name, t, type, op)\
	__global__ void  name ##_sv_ ##t(type* A, type* B, int size) { \
		THREADID \
		CHECKSIZE \
		B[idx] = A[0] op B[idx];}

#define SSBINOP(name, t, type, op)\
	__global__ void  name ##_ss_ ##t(type* A, type* B, int size) { \
		THREADID \
		CHECKSIZE \
		A[0] = A[0] op B[0];}

/* VECTOR-VECTOR BIN OP */

extern "C" { VVBINOP(add, f64, double, +) }
extern "C" { VVBINOP(add, f32, float, +) }

extern "C" { VVBINOP(sub, f64, double, -) }
extern "C" { VVBINOP(sub, f32, float, -) }

extern "C" { VVBINOP(mul, f64, double, *) }
extern "C" { VVBINOP(mul, f32, float, *) }

extern "C" { VVBINOP(div, f64, double, /) }
extern "C" { VVBINOP(div, f32, float, /) }

/* VECTOR-SCALAR BIN OP */

extern "C" { VSBINOP(add, f64, double, +) }
extern "C" { VSBINOP(add, f32, float, +) }

extern "C" { VSBINOP(sub, f64, double, -) }
extern "C" { VSBINOP(sub, f32, float, -) }

extern "C" { VSBINOP(mul, f64, double, *) }
extern "C" { VSBINOP(mul, f32, float, *) }

extern "C" { VSBINOP(div, f64, double, /) }
extern "C" { VSBINOP(div, f32, float, /) }

/* SCALAR-VECTOR BIN OP */

extern "C" { SVBINOP(add, f64, double, +) }
extern "C" { SVBINOP(add, f32, float, +) }

extern "C" { SVBINOP(sub, f64, double, -) }
extern "C" { SVBINOP(sub, f32, float, -) }

extern "C" { SVBINOP(mul, f64, double, *) }
extern "C" { SVBINOP(mul, f32, float, *) }

extern "C" { SVBINOP(div, f64, double, /) }
extern "C" { SVBINOP(div, f32, float, /) }

/* SCALAR-SCALAR BIN OP */	

extern "C" { SSBINOP(add, f64, double, +) }
extern "C" { SSBINOP(add, f32, float, +) }

extern "C" { SSBINOP(sub, f64, double, -) }
extern "C" { SSBINOP(sub, f32, float, -) }

extern "C" { SSBINOP(mul, f64, double, *) }
extern "C" { SSBINOP(mul, f32, float, *) }

extern "C" { SSBINOP(div, f64, double, /) }
extern "C" { SSBINOP(div, f32, float, /) }

/* FUNCTION BIN OP */

#define VVFNBINOP(name, t, type, op)\
	__global__ void  name ##_vv_ ##t(type* A, type* B, int size) { \
		THREADID \
		CHECKSIZE \
		A[idx] = op(A[idx], B[idx]);}

#define VSFNBINOP(name, t, type, op)\
	__global__ void  name ##_vs_ ##t(type* A, type* B, int size) { \
		THREADID \
		CHECKSIZE \
		A[idx] = op(A[idx], B[0]);}

#define SVFNBINOP(name, t, type, op)\
	__global__ void  name ##_sv_ ##t(type* A, type* B, int size) { \
		THREADID \
		CHECKSIZE \
		B[idx] = op(A[0], B[idx]);}

#define SSFNBINOP(name, t, type, op)\
	__global__ void  name ##_ss_ ##t(type* A, type* B, int size) { \
		THREADID \
		CHECKSIZE \
		A[0] = op(A[0], B[0]);}

extern "C" { VVFNBINOP(pow, f64, double, pow) }
extern "C" { VVFNBINOP(pow, f32, float, powf) }
extern "C" { VSFNBINOP(pow, f64, double, pow) }
extern "C" { VSFNBINOP(pow, f32, float, powf) }
extern "C" { SVFNBINOP(pow, f64, double, pow) }
extern "C" { SVFNBINOP(pow, f32, float, powf) }
extern "C" { SSFNBINOP(pow, f64, double, pow) }
extern "C" { SSFNBINOP(pow, f32, float, powf) }