package tensori

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

func vecPow(a, b []int) {
	for i, v := range a {
		switch b[i] {
		case 0:
			a[i] = int(1)
		case 1:
			a[i] = v
		case 2:
			a[i] = v * v
		case 3:
			a[i] = v * v * v
		default:
			a[i] = int(math.Pow(float64(v), float64(b[i])))
		}
	}
}

// TODO:Vectorize
func vecScale(s int, a []int) {
	for i, v := range a {
		a[i] = v * s
	}
}

// TODO:Vectorize
func vecDivBy(s int, a []int) {
	for i, v := range a {
		a[i] = s / v
	}
}

// TODO:Vectorize
func vecTrans(s int, a []int) {
	for i, v := range a {
		a[i] = v + s
	}
}

// TODO:Vectorize
func vecTransFrom(s int, a []int) {
	for i, v := range a {
		a[i] = s - v
	}
}

// TODO:Vectorize
func vecPower(s int, a []int) {
	for i, v := range a {
		a[i] = int(math.Pow(float64(v), float64(s)))
	}
}

// TODO:Vectorize
func vecPowerFrom(s int, a []int) {
	for i, v := range a {
		a[i] = int(math.Pow(float64(s), float64(v)))
	}
}

/* REDUCTION RELATED */

func sum(a []int) int {
	return reduce(add, int(0), a...)
}

/* FUNCTION VARIABLES */

var (
	add = func(a, b int) int { return a + b }
	sub = func(a, b int) int { return a - b }
	mul = func(a, b int) int { return a * b }
	div = func(a, b int) int { return a / b }
	mod = func(a, b int) int { return int(math.Mod(float64(a), float64(b))) }
)
