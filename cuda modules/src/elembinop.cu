#define _USE_MATH_DEFINES
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

extern "C" { VVBINOP(gt, f64, double, >)}
extern "C" { VVBINOP(gt, f32, float, >)}

extern "C" { VVBINOP(gte, f64, double, >=)}
extern "C" { VVBINOP(gte, f32, float, >=)}

extern "C" { VVBINOP(lt, f64, double, <)}
extern "C" { VVBINOP(lt, f32, float, <)}

extern "C" { VVBINOP(lte, f64, double, <=)}
extern "C" { VVBINOP(lte, f32, float, <=)}

extern "C" { VVBINOP(eq, f64, double, ==)}
extern "C" { VVBINOP(eq, f32, float, ==)}

extern "C" { VVBINOP(ne, f64, double, !=)}
extern "C" { VVBINOP(ne, f32, float, !=)}


/* VECTOR-SCALAR BIN OP */

extern "C" { VSBINOP(add, f64, double, +) }
extern "C" { VSBINOP(add, f32, float, +) }

extern "C" { VSBINOP(sub, f64, double, -) }
extern "C" { VSBINOP(sub, f32, float, -) }

extern "C" { VSBINOP(mul, f64, double, *) }
extern "C" { VSBINOP(mul, f32, float, *) }

extern "C" { VSBINOP(div, f64, double, /) }
extern "C" { VSBINOP(div, f32, float, /) }

extern "C" { VSBINOP(gt, f64, double, >)}
extern "C" { VSBINOP(gt, f32, float, >)}

extern "C" { VSBINOP(gte, f64, double, >=)}
extern "C" { VSBINOP(gte, f32, float, >=)}

extern "C" { VSBINOP(lt, f64, double, <)}
extern "C" { VSBINOP(lt, f32, float, <)}

extern "C" { VSBINOP(lte, f64, double, <=)}
extern "C" { VSBINOP(lte, f32, float, <=)}

extern "C" { VSBINOP(eq, f64, double, ==)}
extern "C" { VSBINOP(eq, f32, float, ==)}

extern "C" { VSBINOP(ne, f64, double, !=)}
extern "C" { VSBINOP(ne, f32, float, !=)}

/* SCALAR-VECTOR BIN OP */

extern "C" { SVBINOP(add, f64, double, +) }
extern "C" { SVBINOP(add, f32, float, +) }

extern "C" { SVBINOP(sub, f64, double, -) }
extern "C" { SVBINOP(sub, f32, float, -) }

extern "C" { SVBINOP(mul, f64, double, *) }
extern "C" { SVBINOP(mul, f32, float, *) }

extern "C" { SVBINOP(div, f64, double, /) }
extern "C" { SVBINOP(div, f32, float, /) }

extern "C" { SVBINOP(gt, f64, double, >) }
extern "C" { SVBINOP(gt, f32, float, >) }

extern "C" { SVBINOP(gte, f64, double, >=) }
extern "C" { SVBINOP(gte, f32, float, >=) }

extern "C" { SVBINOP(lt, f64, double, <) }
extern "C" { SVBINOP(lt, f32, float, <) }

extern "C" { SVBINOP(lte, f64, double, <=) }
extern "C" { SVBINOP(lte, f32, float, <=) }

extern "C" { SVBINOP(eq, f64, double, ==) }
extern "C" { SVBINOP(eq, f32, float, ==) }

extern "C" { SVBINOP(ne, f64, double, !=) }
extern "C" { SVBINOP(ne, f32, float, !=) }

/* SCALAR-SCALAR BIN OP */	

extern "C" { SSBINOP(add, f64, double, +) }
extern "C" { SSBINOP(add, f32, float, +) }

extern "C" { SSBINOP(sub, f64, double, -) }
extern "C" { SSBINOP(sub, f32, float, -) }

extern "C" { SSBINOP(mul, f64, double, *) }
extern "C" { SSBINOP(mul, f32, float, *) }

extern "C" { SSBINOP(div, f64, double, /) }
extern "C" { SSBINOP(div, f32, float, /) }

extern "C" { SSBINOP(gt, f64, double, >)}
extern "C" { SSBINOP(gt, f32, float, >)}

extern "C" { SSBINOP(gte, f64, double, >=)}
extern "C" { SSBINOP(gte, f32, float, >=)}

extern "C" { SSBINOP(lt, f64, double, <)}
extern "C" { SSBINOP(lt, f32, float, <)}

extern "C" { SSBINOP(lte, f64, double, <=)}
extern "C" { SSBINOP(lte, f32, float, <=)}

extern "C" { SSBINOP(eq, f64, double, ==)}
extern "C" { SSBINOP(eq, f32, float, ==)}

extern "C" { SSBINOP(ne, f64, double, !=)}
extern "C" { SSBINOP(ne, f32, float, !=)}

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

/*
extern "C" { VVFNBINOP(mod, f64, double, modf) }
extern "C" { VVFNBINOP(mod, f32, float, modff) }
extern "C" { VSFNBINOP(mod, f64, double, modf) }
extern "C" { VSFNBINOP(mod, f32, float, modff) }
extern "C" { SVFNBINOP(mod, f64, double, modf) }
extern "C" { SVFNBINOP(mod, f32, float, modff) }
extern "C" { SSFNBINOP(mod, f64, double, modf) }
extern "C" { SSFNBINOP(mod, f32, float, modff) }
*/