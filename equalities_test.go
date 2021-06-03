package gorgonia

import "testing"

var scalarEqualities = []struct {
	a, b Scalar
	eq   bool
}{
	{NewF64(1), NewF64(1), true},
	{NewF64(1), NewF64(0), false},
	{NewF64(1), NewF32(1), false},

	{NewF32(1), NewF32(1), true},
	{NewF32(1), NewF32(0), false},
	{NewF32(1), NewI(1), false},

	{NewI(1), NewI(1), true},
	{NewI(1), NewI(0), false},
	{NewI(1), NewI64(1), false},

	{NewI64(1), NewI64(1), true},
	{NewI64(1), NewI64(0), false},
	{NewI64(1), NewI32(1), false},

	{NewI32(1), NewI32(1), true},
	{NewI32(1), NewI32(0), false},
	{NewI32(1), NewU8(1), false},

	{NewU8(1), NewU8(1), true},
	{NewU8(1), NewU8(0), false},
	{NewU8(1), NewB(true), false},

	{NewB(true), NewB(true), true},
	{NewB(true), NewB(false), false},
	{NewB(true), NewF64(1), false},
}

func TestScalarEq(t *testing.T) {
	for _, seq := range scalarEqualities {
		if (scalarEq(seq.a, seq.b) && !seq.eq) || (!scalarEq(seq.a, seq.b) && seq.eq) {
			t.Errorf("expected %v(%v) and %v(%v) to be %v", seq.a, TypeOf(seq.a), seq.b, TypeOf(seq.b), seq.eq)
		}
	}
}
