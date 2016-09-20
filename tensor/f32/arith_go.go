// +build !avx,!sse

package tensorf32

import "github.com/chewxy/math32"

func vecAdd(a, b []float32) {
	for i, v := range a {
		a[i] = v + b[i]
	}
}

func vecSub(a, b []float32) {
	for i, v := range a {
		a[i] = v - b[i]
	}
}

func vecMul(a, b []float32) {
	for i, v := range a {
		a[i] = v * b[i]
	}
}

func vecDiv(a, b []float32) {
	for i, v := range a {
		if b[i] == 0 {
			a[i] = math32.Inf(0)
			continue
		}

		a[i] = v / b[i]
	}
}

func vecSqrt(a []float32) {
	for i, v := range a {
		a[i] = math32.Sqrt(v)
	}
}

func vecInvSqrt(a []float32) {
	for i, v := range a {
		a[i] = float32(1) / math32.Sqrt(v)
	}
}
