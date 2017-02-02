package tensori

import "math"

func vecAdd(a, b []int) {
	for i, v := range a {
		a[i] = v + b[i]
	}
}

func vecSub(a, b []int) {
	for i, v := range a {
		a[i] = v - b[i]
	}
}

func vecMul(a, b []int) {
	for i, v := range a {
		a[i] = v * b[i]
	}
}

func vecDiv(a, b []int) {
	for i, v := range a {

		a[i] = v / b[i]
	}
}

func vecSqrt(a []int) {
	for i, v := range a {
		a[i] = int(math.Sqrt(float64(v)))
	}
}

func vecInvSqrt(a []int) {
	for i, v := range a {
		a[i] = 1 / int(math.Sqrt(float64(v)))
	}
}
