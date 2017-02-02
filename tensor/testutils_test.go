package tensor

import (
	"math"
	"math/rand"
	"testing"
)

func checkErr(t *testing.T, expected bool, err error, name string, id interface{}) (cont bool) {
	switch {
	case expected:
		if err == nil {
			t.Errorf("Expected error in test %v (%v)", name, id)
		}
		return true
	case !expected && err != nil:
		t.Errorf("Test %v (%v) errored: %+v", name, id, err)
		return true
	}
	return false
}

func sliceApproxf64(a, b []float64, fn func(a, b float64) bool) bool {
	if len(a) != len(b) {
		return false
	}

	for i, v := range a {
		if math.IsNaN(v) {
			if !alikef64(v, b[i]) {
				return false
			}
		}
		if !fn(v, b[i]) {
			return false
		}
	}
	return true
}

func RandomFloat64(size int) []float64 {
	r := make([]float64, size)
	for i := range r {
		r[i] = rand.NormFloat64()
	}
	return r
}
