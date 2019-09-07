// +build fastmath

package gorgonia

import (
	"math"
	"unsafe"
)

// SetOptimizationLevel to i
func SetOptimizationLevel(i int) { optimizationLevel = i }

func castFU32(x float32) uint32 { return *(*uint32)(unsafe.Pointer(&x)) }
func castFU64(x float64) uint64 { return *(*uint64)(unsafe.Pointer(&x)) }
func castUF64(x uint64) float64 { return *(*float64)(unsafe.Pointer(&x)) }
func castUF32(x uint32) float32 { return *(*float32)(unsafe.Pointer(&x)) }

/*
INVERSE/RECIPROCAL HACKS

Resources:
	http://stackoverflow.com/questions/31555260/fast-vectorized-rsqrt-and-reciprocal-with-sse-avx-depending-on-precision
	https://github.com/stgatilov/recip_rsqrt_benchmark

Also many thanks to the countless crazy people out there who have done their homework, written papers about it
so that I may benefit from it.
*/

// magic numbers are acquired from here:
// 		0x7FDE623822FC16E6 - https://www.pvk.ca/Blog/LowLevel/software-reciprocal.html
// 		0x7FDE6238DA3C2118 - http://www.hackersdelight.org/hdcodetxt/recip.c.txt
// Paul Khong's magic number seems to be the best performing for my use case, with 3 newton iterations
//
// On the number of refinement steps required, 4 refinement steps will yield the same
// results as the naive function for most values. However, the gains in accuracy is offset
// by the loss in speed gains:
//		BenchmarkInv64-8    	300000000	         5.99 ns/op
//		BenchmarkApp4Inv64-8	300000000	         5.09 ns/op
//		BenchmarkApp3Inv64-8	500000000	         3.70 ns/op
func _inversef64(x float64) float64 {
	u := uint64(0x7FDE623822FC16E6) - castFU64(x)
	// u := uint64(0x7FDE6238DA3C2118) - castFU64(x)

	f := castUF64(u)

	// 3 newton raphson refinement steps:
	for i := 0; i < 3; i++ {
		f = 2.0*f - f*f*x
	}
	return f
}

// magic numbers acquired from here:
//		http://bits.stephan-brumme.com/inverse.html
// On the number of refinement steps:
// 		BenchmarkInv32-8    	500000000	         3.85 ns/op
//		BenchmarkApp3Inv32-8	500000000	         3.69 ns/op
// 		BenchmarkApp2Inv32-8	1000000000	         2.47 ns/op
//
// I have also found that 2 refinement steps are more than sufficient to get decent results. No funny gradient explosions for sure
// TODO: use RCPSS when available
func _inversef32(x float32) float32 {
	u := uint32(0x7F000000) - castFU32(x)
	f := castUF32(u)

	// 2 newton raphson refinement steps:
	for i := 0; i < 2; i++ {
		f = 2.0*f - f*f*x
	}
	return castUF32(u)
}

/*
TANH HACKS
Resources:
	"Speed Improvement of the Back-Propagation on Current Generation Workstations" D. Anguita, G. Parodi and R. Zunino. Proceedings of the World Congress on Neural Networking, 1993.
	Fuzzpillz's Tanh http://www.musicdsp.org/showone.php?id=178
	varosound's lambert expansion Tanh -  https://varietyofsound.wordpress.com/2011/02/14/efficient-tanh-computation-using-lamberts-continued-fraction/

Here are the relative benchmarks:

float32
	BenchmarkTanh32UW-8       	200000000	         7.34 ns/op
	BenchmarkTanh32-8         	100000000	        10.6 ns/op
	BenchmarkFuzzpillsTanh32-8	200000000	         9.24 ns/op
	BenchmarkAnguitaTanh32-8  	500000000	         3.52 ns/op
	BenchmarkVarosoundTanh32-8	200000000	         8.96 ns/op

float64
	BenchmarkTanh64UW-8       	200000000	         7.25 ns/op
	BenchmarkTanh64-8         	200000000	         9.64 ns/op
	BenchmarkFuzzpillsTanh64-8	200000000	         5.98 ns/op
	BenchmarkAnguitaTanh64-8  	500000000	         3.26 ns/op
	BenchmarkVarosoundTanh64-8	300000000	         6.03 ns/op

There appears to be a problem when using float32 - the conversion takes extra time.
Tanh32UW and Tanh64UW is a direct call to math.Tanh(), without a wrapping function call. The results of the float32 version is inaccurate, because if you
wrap the results in float32(), the benchmark won't run

On the precisions, I found Anguita's the least precise, but surprisingly works well for the very limited scope of things I am doing.
VarietyOfSound's approximation algorithm is also very close to the actual math.Tanh() implementation.
*/

func _tanhf32(x float32) float32 {
	switch optimizationLevel {
	case 0, 1:
		return float32(math.Tanh(float64(x)))
	case 2:
		// Use Anguita
		switch {
		case x > 1.92033:
			return 0.96016
		case x > 0:
			return 0.96016 - 0.26037*(x-1.92033)*(x-1.92033)
		case x <= -1.92033:
			return -0.96016
		case x < 0:
			return 0.26037*(x+1.92033)*(x+1.92033) - 0.96016
		}
	}
	panic("unreachable")
}

func _tanhf64(x float64) float64 {
	switch optimizationLevel {
	case 0:
		return math.Tanh(x)
	case 1:
		// use Variety of Sound's
		x2 := x * x
		a := x * (135135.0 + x2*(17325.0+x2*(378.0+x2)))
		b := 135135.0 + x2*(62370.0+x2*(3150.0+x2*28.0))
		return a / b
	case 2:
		// Use Anguita
		switch {
		case x > 1.92033:
			return 0.96016
		case x > 0:
			return 0.96016 - 0.26037*(x-1.92033)*(x-1.92033)
		case x <= -1.92033:
			return -0.96016
		case x < 0:
			return 0.26037*(x+1.92033)*(x+1.92033) - 0.96016
		}
	}
	panic("unreachable")
}

func _sigmoidf64(x float64) float64 {
	return 0
}

func _sigmoidf32(x float32) float32 {
	return 0
}
