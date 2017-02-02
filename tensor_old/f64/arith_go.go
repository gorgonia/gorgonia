// +build !avx,!sse

package tensorf64

import "math"

func vecAdd(a, b []float64) {
	for i, v := range a {
		a[i] = v + b[i]
	}
}

func vecSub(a, b []float64) {
	for i, v := range a {
		a[i] = v - b[i]
	}
}

func vecMul(a, b []float64) {
	for i, v := range a {
		a[i] = v * b[i]
	}
}

func vecDiv(a, b []float64) {
	for i, v := range a {
		if b[i] == 0 {
			a[i] = math.Inf(0)
			continue
		}

		a[i] = v / b[i]
	}
}

func vecSqrt(a []float64) {
	for i, v := range a {
		a[i] = math.Sqrt(v)
	}
}

func vecInvSqrt(a []float64) {
	for i, v := range a {
		a[i] = float64(1) / math.Sqrt(v)
	}
}
