package tensorf32

import "github.com/chewxy/math32"

func tolerance(a, b, e float32) bool {
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

func closeenough(a, b float32) bool { return tolerance(a, b, 1e-6) }
func isclose(a, b float32) bool     { return tolerance(a, b, 1e-6) }
func veryclose(a, b float32) bool   { return tolerance(a, b, 4e-10) }
func soclose(a, b, e float32) bool  { return tolerance(a, b, e) }
func alike(a, b float32) bool {
	switch {
	case math32.IsNaN(a) && math32.IsNaN(b):
		return true
	case a == b:
		return math32.Signbit(a) == math32.Signbit(b)
	}
	return false
}

func sliceApprox(a, b []float32, fn func(a, b float32) bool) bool {
	if len(a) != len(b) {
		return false
	}

	for i, v := range a {
		if math32.IsNaN(v) {
			if !alike(v, b[i]) {
				return false
			}
		}
		if !fn(v, b[i]) {
			return false
		}
	}
	return true
}
