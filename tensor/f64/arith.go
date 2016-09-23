package tensorf64

import "math"

// this file is used in tandem with these other files:
// 		arith_asm.go
//		arith_go.go
//
// arith_asm.go and arith_go.go have functions that are exactly the same.
// arith_asm.go is the header for any arithmeticfunction that has
// a asm version of it (which will all have names like arith_$FUNCTIONNAME_$ARCH.s )
//
// arith_go.go basically is the default versions of all the functions listed in arith_asm.go
//
// arithmetic functions that are not in either will be put here.
// Some functions are listed with a TODO: Vectorize. These are functions which will eventually have asm versions

func vecPow(a, b []float64) {
	for i, v := range a {
		switch b[i] {
		case 0:
			a[i] = float64(1)
		case 1:
			a[i] = v
		case 2:
			a[i] = v * v
		case 3:
			a[i] = v * v * v
		default:
			a[i] = math.Pow(v, b[i])
		}
	}
}

// TODO:Vectorize
func vecScale(s float64, a []float64) {
	for i, v := range a {
		a[i] = v * s
	}
}

// TODO:Vectorize
func vecDivBy(s float64, a []float64) {
	for i, v := range a {
		a[i] = s / v
	}
}

// TODO:Vectorize
func vecTrans(s float64, a []float64) {
	for i, v := range a {
		a[i] = v + s
	}
}

// TODO:Vectorize
func vecTransFrom(s float64, a []float64) {
	for i, v := range a {
		a[i] = s - v
	}
}

// TODO:Vectorize
func vecPower(s float64, a []float64) {
	for i, v := range a {
		a[i] = math.Pow(v, s)
	}
}

// TODO:Vectorize
func vecPowerFrom(s float64, a []float64) {
	for i, v := range a {
		a[i] = math.Pow(s, v)
	}
}

/* REDUCTION RELATED */

func sum(a []float64) float64 {
	return reduce(add, float64(0), a...)
}

// caveat : default value is used
func sliceMax(a []float64) (retVal float64) {
	for _, v := range a {
		if v > retVal {
			retVal = v
		}
	}
	return
}

func vecMax(a, b []float64) {
	if len(a) != len(b) {
		panic("Index error")
	}

	a = a[:len(a)]
	b = b[:len(a)]

	for i, v := range a {
		bv := b[i]
		if bv > v {
			a[i] = bv
		}
	}
}

/* FUNCTION VARIABLES */

var (
	add = func(a, b float64) float64 { return a + b }
	sub = func(a, b float64) float64 { return a - b }
	mul = func(a, b float64) float64 { return a * b }
	div = func(a, b float64) float64 { return a / b }
	mod = func(a, b float64) float64 { return math.Mod(a, b) }
)
