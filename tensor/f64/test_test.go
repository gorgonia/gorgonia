package tensorf64

import "math"

func tolerance(a, b, e float64) bool {
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

func closeenough(a, b float64) bool { return tolerance(a, b, 1e-8) }
func isclose(a, b float64) bool     { return tolerance(a, b, 1e-14) }
func veryclose(a, b float64) bool   { return tolerance(a, b, 4e-16) }
func soclose(a, b, e float64) bool  { return tolerance(a, b, e) }
func alike(a, b float64) bool {
	switch {
	case math.IsNaN(a) && math.IsNaN(b):
		return true
	case a == b:
		return math.Signbit(a) == math.Signbit(b)
	}
	return false
}

func sliceApprox(a, b []float64, fn func(a, b float64) bool) bool {
	if len(a) != len(b) {
		return false
	}

	for i, v := range a {
		if math.IsNaN(v) {
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
