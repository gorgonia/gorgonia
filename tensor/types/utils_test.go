package types

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestLtoi_Itol(t *testing.T) {
	var strides []int
	var shape Shape
	assert := assert.New(t)

	shape = Shape{3, 4}
	strides = shape.CalcStrides()

	for i := 0; i < shape[0]; i++ {
		for j := 0; j < shape[1]; j++ {
			coord := []int{i, j}
			idx, err := Ltoi(shape, strides, coord...)
			if err != nil {
				t.Error(err)
			}

			got, err := Itol(idx, shape, strides)
			if err != nil {
				t.Error(err)
			}

			assert.Equal(coord, got)
		}
	}

	shape = Shape{2, 3, 4}
	strides = shape.CalcStrides()

	for i := 0; i < shape[0]; i++ {
		for j := 0; j < shape[1]; j++ {
			for k := 0; k < shape[2]; k++ {
				coord := []int{i, j, k}
				idx, err := Ltoi(shape, strides, coord...)
				if err != nil {
					t.Error(err)
				}

				got, err := Itol(idx, shape, strides)
				if err != nil {
					t.Error(err)
				}

				assert.Equal(coord, got)
			}
		}
	}
}

func TestPermute(t *testing.T) {
	assert := assert.New(t)
	var shape Shape
	var strides []int

	shape = Shape{3, 4}
	retVals, err := Permute([]int{1, 0}, shape)
	if err != nil {
		t.Error(err)
	}
	if len(retVals) != 1 {
		t.Error("one input should only have one output.")
	}
	assert.Equal([]int{4, 3}, retVals[0])

	shape = Shape{2, 3, 4}
	strides = []int{12, 4, 1}
	retVals, err = Permute([]int{1, 0, 2}, shape, strides)
	if err != nil {
		t.Error(err)
	}
	if len(retVals) != 2 {
		t.Error("two inputs should have two outputs")
	}
	assert.Equal([]int{3, 2, 4}, retVals[0])
	assert.Equal([]int{4, 12, 1}, retVals[1])

	// NOOP
	retVals, err = Permute([]int{0, 1, 2}, shape, strides)
	if err != nil {
		if _, ok := err.(NoOpError); !ok {
			t.Error(err)
		}
	}
	assert.Equal([]int(shape), retVals[0])
	assert.Equal(strides, retVals[1])

	/* Idiotsville */
	_, err = Permute([]int{1, 2, 3})
	if err == nil {
		t.Error("Expected an OpError - nothing was passed in!")
	}

	_, err = Permute([]int{0, 1}, shape, strides)
	if err == nil {
		t.Error("Expected a DimMismatch error.")
	}

	strides = []int{2, 1}
	_, err = Permute([]int{0, 1}, strides, shape)
	if err == nil {
		t.Error("Expected a DimMismatch error.")
	}

	// a pattern that is greater than the dim
	shape = Shape{2, 3}
	retVals, err = Permute([]int{0, 2}, strides, shape)
	if err == nil {
		t.Error("Expected an AxisErr")
		t.Error(retVals)
	}

	// repeated patterns
	shape = Shape{2, 3}
	_, err = Permute([]int{0, 0}, strides, shape)
	if err == nil {
		t.Error("Expected an AxisErr")
	}

}

func TestUnsafePermute(t *testing.T) {
	assert := assert.New(t)
	var shape []int
	var strides []int

	shape = []int{3, 4}
	err := UnsafePermute([]int{1, 0}, shape)
	if err != nil {
		t.Error(err)
	}
	assert.Equal([]int{4, 3}, shape)

	shape = []int{2, 3, 4}
	strides = []int{12, 4, 1}
	err = UnsafePermute([]int{1, 0, 2}, shape, strides)
	if err != nil {
		t.Error(err)
	}
	assert.Equal([]int{3, 2, 4}, shape)
	assert.Equal([]int{4, 12, 1}, strides)

	// NOOP
	err = UnsafePermute([]int{0, 1, 2}, shape, strides)
	if err != nil {
		if _, ok := err.(NoOpError); !ok {
			t.Error(err)
		}
	}
	assert.Equal([]int{3, 2, 4}, shape)
	assert.Equal([]int{4, 12, 1}, strides)

	/* Idiotsville */
	_, err = Permute([]int{1, 2, 3})
	if err == nil {
		t.Error("Expected an OpError - nothing was passed in!")
	}

	_, err = Permute([]int{0, 1}, shape, strides)
	if err == nil {
		t.Error("Expected a DimMismatch error.")
	}

	strides = []int{2, 1}
	_, err = Permute([]int{0, 1}, strides, shape)
	if err == nil {
		t.Error("Expected a DimMismatch error.")
	}

	// a pattern that is greater than the dim
	shape = []int{2, 3}
	err = UnsafePermute([]int{0, 2}, strides, shape)
	if err == nil {
		t.Error("Expected an AxisErr")
	}

	// repeated patterns
	shape = []int{2, 3}
	_, err = Permute([]int{0, 0}, strides, shape)
	if err == nil {
		t.Error("Expected an AxisErr")
	}
}

func TestSliceSanity(t *testing.T) {
	var s Slice
	var err error

	s = rs{0, 1, 1}
	if err = sliceSanity(s, 10); err != nil {
		t.Error(err)
	}

	s = rs{1, 6, 2}
	if err = sliceSanity(s, 10); err != nil {
		t.Error(err)
	}

	s = rs{}
	if err = sliceSanity(s, 10); err != nil {
		t.Error(err)
	}

	s = rs{4, 2, 1}
	if err = sliceSanity(s, 10); err == nil {
		t.Error("Expected an error")
	}

	s = rs{2, 4, 0}
	if err = sliceSanity(s, 10); err == nil {
		t.Error("Expected an error")
	}

	s = rs{-1, -1, 1}
	if err = sliceSanity(s, 10); err == nil {
		t.Error("Expected an error")
	}

	s = rs{-2, -1, 1}
	if err = sliceSanity(s, 10); err == nil {
		t.Error("Expected an error")
	}

	s = rs{12, 24, 1}
	if err = sliceSanity(s, 10); err == nil {
		t.Error("Expected an error")
	}
}
