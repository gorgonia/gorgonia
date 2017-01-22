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

func tolerancef64(a, b, e float64) bool {
	d := a - b
	if d < 0 {
		d = -d
	}

	// note: b is correct (expected) value, a is actual value.
	// make error tolerance a fraction of b, not a.
	if b != 0 {
		e = e * b
		if e < 0 {
			e = -e
		}
	}
	return d < e
}

func closeenoughf64(a, b float64) bool { return tolerancef64(a, b, 1e-8) }
func isclosef64(a, b float64) bool     { return tolerancef64(a, b, 1e-14) }
func veryclosef64(a, b float64) bool   { return tolerancef64(a, b, 4e-16) }
func soclosef64(a, b, e float64) bool  { return tolerancef64(a, b, e) }
func alikef64(a, b float64) bool {
	switch {
	case math.IsNaN(a) && math.IsNaN(b):
		return true
	case a == b:
		return math.Signbit(a) == math.Signbit(b)
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
