package gorgonia

import (
	"testing"

	"gorgonia.org/tensor"
)

var cloneValTests = []Value{
	// prims
	NewF64(10.0),
	NewF32(10.0),
	NewI(10),
	NewI64(10),
	NewI32(10),
	NewU8(10),
	NewB(true),

	tensor.New(tensor.Of(tensor.Float64), tensor.WithShape(2, 4, 6)),
	tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(2, 4, 6)),
}

func TestCloneValue(t *testing.T) {
	for _, cvts := range cloneValTests {
		v, err := CloneValue(cvts)
		if err != nil {
			t.Error(err)
		}

		if v == cvts {
			t.Errorf("Expected values to have different pointers. Got %p == %p", v, cvts)
		}

		if !ValueEq(cvts, v) {
			t.Errorf("Cloning failed")
		}
	}
}
