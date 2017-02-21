package gorgonia

import "testing"

var scalarEqualities = []struct {
	a, b Scalar
	eq   bool
}{
	{newF64(1), newF64(1), true},
	{newF64(1), newF64(0), false},
	{newF64(1), newF32(1), false},

	{newF32(1), newF32(1), true},
	{newF32(1), newF32(0), false},
	{newF32(1), newI(1), false},

	{newI(1), newI(1), true},
	{newI(1), newI(0), false},
	{newI(1), newI64(1), false},

	{newI64(1), newI64(1), true},
	{newI64(1), newI64(0), false},
	{newI64(1), newI32(1), false},

	{newI32(1), newI32(1), true},
	{newI32(1), newI32(0), false},
	{newI32(1), newU8(1), false},

	{newU8(1), newU8(1), true},
	{newU8(1), newU8(0), false},
	{newU8(1), newB(true), false},

	{newB(true), newB(true), true},
	{newB(true), newB(false), false},
	{newB(true), newF64(1), false},
}

func TestScalarEq(t *testing.T) {
	for _, seq := range scalarEqualities {
		if (scalarEq(seq.a, seq.b) && !seq.eq) || (!scalarEq(seq.a, seq.b) && seq.eq) {
			t.Errorf("expected %v(%v) and %v(%v) to be %v", seq.a, TypeOf(seq.a), seq.b, TypeOf(seq.b), seq.eq)
		}
	}
}
