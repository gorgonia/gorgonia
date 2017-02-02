package tensori

import "math"

// this file is used in tandem with these other files:
// 		arith_incr_asm.go
//		arith_incr_go.go
//
// arith_incr_asm.go and arith_incr_go.go have functions that are exactly the same.
// arith_incr_asm.go is the header for any arithmeticfunction that has
// a asm version of it (which will all have names like arith_$FUNCTIONNAME_$ARCH.s )
//
// arith_incr_go.go basically is the default versions of all the functions listed in arith_incr_asm.go
//
// arith_incr deals with vectorized arithmetic ops that are then incremented onto another vector.
// Example (incrVecMul) is this:
//		vecA += vecB*vecC
//
// incrVecAdd and incrVecSub  are not included because of the commutativity of the operators:
//		vecA += vecB+vecC
// is equivalent to:
//		vecAdd(vecA, vecB)
//		vecAdd(vecA, vecC)

func incrVecMul(a, b, c []int) {
	for i, v := range b {
		a[i] += v * c[i]
	}
}

func incrVecScale(a, b []int, c int) {
	for i, v := range b {
		a[i] += v * c
	}
}

func incrVecDiv(a, b, c []int) {
	for i, v := range b {

		a[i] += v / c[i]
	}
}

func incrVecDivBy(a, b []int, c int) {
	for i, v := range b {

		a[i] += c / v
	}
}

func incrVecPow(a, b, c []int) {
	for i, v := range b {
		switch c[i] {
		case 0:
			a[i]++
		case 1:
			a[i] += v
		case 2:
			a[i] += v * v
		case 3:
			a[i] += v * v * v
		default:
			a[i] += int(math.Pow(float64(v), float64(c[i])))
		}
	}
}

func incrVecPower(a, b []int, c int) {
	switch c {
	case 0:
		for i := range a {
			a[i]++
		}
	case 1:
		vecAdd(a, b)
	case 2:
		for i, v := range b {
			a[i] += v * v
		}
	case 3:
		for i, v := range b {
			a[i] += v * v * v
		}
	default:
		for i, v := range b {
			a[i] += int(math.Pow(float64(v), float64(c)))
		}
	}
}

func incrVecPowerFrom(a, b []int, c int) {
	switch c {
	case 0:
		return
	case 1:
		for i := range a {
			a[i]++
		}
	default:
		for i, v := range b {
			a[i] += int(math.Pow(float64(c), float64(v)))
		}
	}
}
