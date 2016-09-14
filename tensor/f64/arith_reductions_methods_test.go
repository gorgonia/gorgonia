package tensorf64

import (
	"testing"

	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/stretchr/testify/assert"
)

func TestTreduce(t *testing.T) {
	assert := assert.New(t)
	var T, T2 *Tensor
	var expectedShape types.Shape
	var expectedData []float64
	var err error

	/*
		3D tensor

		0, 1
		2, 3
		4, 5

		6, 7
		8, 9
		10, 11
	*/
	T = NewTensor(WithShape(2, 3, 2), WithBacking(RangeFloat64(0, 2*3*2)))
	T2, err = T.Reduce(add, 0, 0)
	if err != nil {
		t.Error(err)
	}
	expectedShape = types.Shape{3, 2}
	expectedData = []float64{6, 8, 10, 12, 14, 16}
	assert.Equal(expectedShape, T2.Shape())
	assert.Equal(expectedData, T2.data)

	T2, err = T.Reduce(add, 0, 1)
	if err != nil {
		t.Error(err)
	}

	expectedShape = types.Shape{2, 2}
	expectedData = []float64{6, 9, 24, 27}
	assert.Equal(expectedShape, T2.Shape())
	assert.Equal(expectedData, T2.data)

	T2, err = T.Reduce(add, 0, 2)
	if err != nil {
		t.Error(err)
	}
	expectedShape = types.Shape{2, 3}
	expectedData = []float64{1, 5, 9, 13, 17, 21}
	assert.Equal(expectedShape, T2.Shape())
	assert.Equal(expectedData, T2.data)

	/*
		Matrix

		0, 1, 2
		3, 4, 5
	*/
	T = NewTensor(WithShape(2, 3), WithBacking(RangeFloat64(0, 6)))
	T2, err = T.Reduce(add, 0, 0)
	if err != nil {
		t.Error(err)
	}
	expectedShape = types.Shape{3}
	expectedData = []float64{3, 5, 7}
	assert.Equal(expectedShape, T2.Shape())
	assert.Equal(expectedData, T2.data)

	T2, err = T.Reduce(mul, 1, 0)
	if err != nil {
		t.Error(err)
	}
	expectedShape = types.Shape{3}
	expectedData = []float64{0, 4, 10}
	assert.Equal(expectedShape, T2.Shape())
	assert.Equal(expectedData, T2.data)

	T2, err = T.Reduce(div, 0, 0)
	if err != nil {
		t.Error(err)
	}
	expectedShape = types.Shape{3}
	expectedData = []float64{0, 0.25, 0.4}
	assert.Equal(expectedShape, T2.Shape())
	assert.Equal(expectedData, T2.data)

	T2, err = T.Reduce(mul, 1, 1)
	if err != nil {
		t.Error(err)
	}
	expectedShape = types.Shape{2}
	expectedData = []float64{0, 60}
	assert.Equal(expectedShape, T2.Shape())
	assert.Equal(expectedData, T2.data)
}

func TestTSum(t *testing.T) {
	assert := assert.New(t)
	var T, T2 *Tensor
	var err error
	var expectedShape types.Shape
	var expectedData []float64

	// Most common use (don't sum along any axis)
	T = NewTensor(WithShape(2, 2), WithBacking(RangeFloat64(0, 4)))
	T2, err = T.Sum()
	if err != nil {
		t.Error(err)
	}
	expectedShape = types.ScalarShape()
	expectedData = []float64{6}
	assert.Equal(expectedData, T2.data)
	assert.True(expectedShape.Eq(T2.Shape()))

	// sum along one axis (see TestTsum for more specific axis related testing)
	T2, err = T.Sum(0)
	if err != nil {
		t.Error(err)
	}

	expectedShape = types.Shape{2}
	expectedData = []float64{2, 4}
	assert.Equal(expectedData, T2.data)
	assert.Equal(expectedShape, T2.Shape())

	// Sum along multiple axis
	T = NewTensor(WithShape(2, 3, 4), WithBacking(RangeFloat64(0, 2*3*4)))
	T2, err = T.Sum(1, 2)
	if err != nil {
		t.Error(err)
	}

	expectedShape = types.Shape{2}
	expectedData = []float64{66, 210}
	assert.Equal(expectedData, T2.data)
	assert.Equal(expectedShape, T2.Shape())

	// Sum along multiple axes, but larger axis first. Should have the same result as prev
	T2, err = T.Sum(2, 1)
	if err != nil {
		t.Error(err)
	}

	assert.Equal(expectedData, T2.data)
	assert.Equal(expectedShape, T2.Shape())

	/* IDIOT TESTING TIME */
	_, err = T.Sum(3)
	assert.NotNil(err)
}

func TestTsum(t *testing.T) {
	assert := assert.New(t)
	var T, T2 *Tensor
	var expectedShape types.Shape
	var expectedData []float64

	T = NewTensor(WithShape(2, 2), WithBacking(RangeFloat64(0, 4)))

	T2 = T.sum(0)
	expectedShape = types.Shape{2}
	expectedData = []float64{2, 4}

	assert.Equal(expectedData, T2.data)
	assert.Equal(expectedShape, T2.Shape())

	T2 = T.sum(1)
	expectedShape = types.Shape{2}
	expectedData = []float64{1, 5}

	assert.Equal(expectedData, T2.data)
	assert.Equal(expectedShape, T2.Shape())

	/* 3D tensor */

	T = NewTensor(WithShape(5, 3, 6), WithBacking(RangeFloat64(0, 5*3*6)))

	T2 = T.sum(0)
	expectedShape = types.Shape{3, 6}
	expectedData = []float64{
		180, 185, 190, 195, 200, 205,
		210, 215, 220, 225, 230, 235,
		240, 245, 250, 255, 260, 265,
	}

	assert.Equal(expectedData, T2.data)
	assert.Equal(expectedShape, T2.Shape())

	T2 = T.sum(1)
	expectedShape = types.Shape{5, 6}
	expectedData = []float64{
		18, 21, 24, 27, 30, 33,
		72, 75, 78, 81, 84, 87,
		126, 129, 132, 135, 138, 141,
		180, 183, 186, 189, 192, 195,
		234, 237, 240, 243, 246, 249,
	}

	assert.Equal(expectedData, T2.data)
	assert.Equal(expectedShape, T2.Shape())

	T2 = T.sum(2)
	expectedShape = types.Shape{5, 3}
	expectedData = []float64{
		15, 51, 87,
		123, 159, 195,
		231, 267, 303,
		339, 375, 411,
		447, 483, 519,
	}

	assert.Equal(expectedData, T2.data)
	assert.Equal(expectedShape, T2.Shape())
}
