package tensori

import (
	"fmt"
	"math"
)

// This file is for the safe versions of any arithmetic functions listed in arith.go and (arith_asm.go or arith_go.go)
// A safe version is a version with return values, and does not mutate the underlying data

func safeVecAdd(a, b []int, optional ...[]int) (retVal []int) {
	var reuse []int
	if len(a) != len(b) {
		panic("Differing lengths!")
	}

	if len(optional) >= 1 {
		reuse = optional[0]
		if len(reuse) != len(b) {
			panic("Reused slice does not have the same size as the expected result slice")
		}
	}

	if reuse != nil {
		retVal = reuse
	} else {
		retVal = make([]int, len(a))
	}

	copy(retVal, a)
	vecAdd(retVal, b)
	return retVal
}

func safeVecSub(a, b []int, optional ...[]int) (retVal []int) {
	var reuse []int
	if len(a) != len(b) {
		panic("Differing lengths!")
	}

	if len(optional) >= 1 {
		reuse = optional[0]
		if len(reuse) != len(b) {
			panic("Reused slice does not have the same size as the expected result slice")
		}
	}

	if reuse != nil {
		retVal = reuse
	} else {
		retVal = make([]int, len(a))
	}

	copy(retVal, a)
	vecSub(retVal, b)
	return retVal
}

func safeVecMul(a, b []int, optional ...[]int) (retVal []int) {
	var reuse []int
	if len(a) != len(b) {
		panic("Differing lengths!")
	}

	if len(optional) >= 1 {
		reuse = optional[0]
		if len(reuse) != len(b) {
			err := fmt.Sprintf("Reused slice does not have the same size as the expected result slice. Expected: %v. Got %v", len(b), len(reuse))
			panic(err)
		}
	}

	if reuse != nil {
		retVal = reuse
	} else {
		retVal = make([]int, len(a))
	}

	copy(retVal, a)
	vecMul(retVal, b)
	return retVal
}

func safeVecDiv(a, b []int, optional ...[]int) (retVal []int) {
	var reuse []int
	if len(a) != len(b) {
		panic("Differing lengths!")
	}

	if len(optional) >= 1 {
		reuse = optional[0]
		if len(reuse) != len(b) {
			panic("Reused slice does not have the same size as the expected result slice")
		}
	}

	if reuse != nil {
		retVal = reuse
	} else {
		retVal = make([]int, len(a))
	}

	copy(retVal, a)
	vecDiv(retVal, b)
	return retVal
}

func safeVecPow(a, b []int, optional ...[]int) (retVal []int) {
	var reuse []int
	if len(a) != len(b) {
		panic("Differing lengths!")
	}
	if len(optional) >= 1 {
		reuse = optional[0]
		if len(reuse) != len(b) {
			panic("Reused slice does not have the same size as the expected result slice")
		}
	}

	if reuse != nil {
		retVal = reuse
	} else {
		retVal = make([]int, len(a))
	}

	copy(retVal, a)
	vecPow(retVal, b)
	return retVal
}

func safeVecTrans(s int, a []int, optional ...[]int) (retVal []int) {
	if len(optional) >= 1 {
		retVal = optional[0]
		if len(retVal) != len(a) {
			panic("Reused slice does not have the same size as the expected result slice")
		}
	} else {
		retVal = make([]int, len(a))
	}
	for i, v := range a {
		retVal[i] = v + s
	}
	return
}

func safeVecTransFrom(s int, a []int, optional ...[]int) (retVal []int) {
	if len(optional) >= 1 {
		retVal = optional[0]
		if len(retVal) != len(a) {
			panic("Reused slice does not have the same size as the expected result slice")
		}
	} else {
		retVal = make([]int, len(a))
	}

	for i, v := range a {
		retVal[i] = s - v
	}
	return
}

func safeVecScale(s int, a []int, optional ...[]int) (retVal []int) {
	if len(optional) >= 1 {
		retVal = optional[0]
		if len(retVal) != len(a) {
			panic("Reused slice does not have the same size as the expected result slice")
		}
	} else {
		retVal = make([]int, len(a))
	}

	for i, v := range a {
		retVal[i] = v * s
	}
	return
}

func safeVecDivBy(s int, a []int, optional ...[]int) (retVal []int) {
	if len(optional) >= 1 {
		retVal = optional[0]
		if len(retVal) != len(a) {
			panic("Reused slice does not have the same size as the expected result slice")
		}
	} else {
		retVal = make([]int, len(a))
	}

	for i, v := range a {
		retVal[i] = s / v
	}
	return
}

func safeVecPower(s int, a []int, optional ...[]int) (retVal []int) {
	if len(optional) >= 1 {
		retVal = optional[0]
		if len(retVal) != len(a) {
			panic("Reused slice does not have the same size as the expected result slice")
		}
	} else {
		retVal = make([]int, len(a))
	}

	for i, v := range a {
		retVal[i] = int(math.Pow(float64(v), float64(s)))
	}
	return
}

func safeVecPowerFrom(s int, a []int, optional ...[]int) (retVal []int) {
	if len(optional) >= 1 {
		retVal = optional[0]
		if len(retVal) != len(a) {
			panic("Reused slice does not have the same size as the expected result slice")
		}
	} else {
		retVal = make([]int, len(a))
	}

	for i, v := range a {
		retVal[i] = int(math.Pow(float64(s), float64(v)))
	}
	return
}

/* Unaries */

func safeVecSqrt(a []int, optional ...[]int) (retVal []int) {
	if len(optional) >= 1 {
		retVal = optional[0]
		if len(retVal) != len(a) {
			panic("Reused slice does not have the same size as the expected result slice")
		}
	} else {
		retVal = make([]int, len(a))
	}

	copy(retVal, a)
	vecSqrt(retVal)
	return
}

func safeVecInvSqrt(a []int, optional ...[]int) (retVal []int) {
	if len(optional) >= 1 {
		retVal = optional[0]
		if len(retVal) != len(a) {
			panic("Reused slice does not have the same size as the expected result slice")
		}
	} else {
		retVal = make([]int, len(a))
	}

	copy(retVal, a)
	vecInvSqrt(retVal)
	return
}
