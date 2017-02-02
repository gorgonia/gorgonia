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

// sliceMax finds the max of a []float64. it panics if the slice is empty
func sliceMax(a []float64) (retVal float64) {
	if len(a) < 1 {
		panic("Cannot find the max of an empty slice")
	}
	return reduce(max, a[0], a[1:]...)
}

// sliceMin finds the max of a []float64. it panics if the slice is empty
func sliceMin(a []float64) (retVal float64) {
	if len(a) < 1 {
		panic("Cannot find the min of an empty slice")
	}
	return reduce(min, a[0], a[1:]...)
}

// vecMax takes two slices, and compares them elementwise. The highest value is put into a
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

// vecMax takes two slices, and compares them elementwise. The lowest value is put into a
func vecMin(a, b []float64) {
	if len(a) != len(b) {
		panic("Index error")
	}

	a = a[:len(a)]
	b = b[:len(a)]

	for i, v := range a {
		bv := b[i]
		if bv < v {
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

	min = func(a, b float64) float64 {
		if a < b {
			return a
		}
		return b
	}

	max = func(a, b float64) float64 {
		if a > b {
			return a
		}
		return b
	}
)
