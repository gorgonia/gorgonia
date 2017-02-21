package gorgonia

import (
	"testing"

	"github.com/chewxy/gorgonia/tensor"
)

var cloneValTests = []Value{
	// prims
	newF64(10.0),
	newF32(10.0),
	newI(10),
	newI64(10),
	newI32(10),
	newU8(10),
	newB(true),

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
			t.Error("Expected values to have different pointers. Got %p == %p", v, cvts)
		}

		if !ValueEq(cvts, v) {
			t.Errorf("Cloning failed")
		}
	}
}
