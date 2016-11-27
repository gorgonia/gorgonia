package gorgonia

import (
	"testing"

	"github.com/chewxy/gorgonia/tensor"
	"github.com/chewxy/gorgonia/tensor/types"
)

var cloneValTests = []struct {
	v   Value
	ptr bool
}{
	// prims
	{F64(10.0), false},
	{F32(10.0), false},
	{I(10), false},
	{I64(10), false},
	{I32(10), false},
	{U8(10), false},
	{B(true), false},

	{tensor.New(types.Float64, tensor.WithShape(2, 4, 6)), true},
	{tensor.New(types.Float32, tensor.WithShape(2, 4, 6)), true},
}

func TestCloneValue(t *testing.T) {
	for _, cvts := range cloneValTests {
		v, err := CloneValue(cvts.v)
		if err != nil {
			t.Error(err)
		}

		if cvts.ptr {
			if v == cvts.v {
				t.Error("Expected values to have different pointers. Got %p == %p", v, cvts.v)
			}
		}

		if !ValueEq(cvts.v, v) {
			t.Errorf("Cloning failed")
		}
	}
}
