package tensori

import (
	"testing"

	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/stretchr/testify/assert"
)

var basicArgT = NewTensor(WithShape(2, 3, 4, 5, 2), WithBacking([]int{
	3, 1, 3, 7, 6, -2, 10, 8, 10, 2, 9, 5, -8,
	3, 7, -2, 1, -10, -4, -8, -3, -9, 8, 4, -2, 7,
	8, 10, -5, 6, -2, 2, 0, 9, -8, 9, -2, 6, -6,
	-1, -10, 4, -7, 9, 7, -8, -3, 3, -2, -8, -10, 8,
	3, -4, 9, 10, -1, -2, 1, 4, 1, 7, -1, -5, 7,
	-9, -5, 2, -5, -6, -1, 0, 8, -7, -2, 9, -2, -9,
	5, 9, 10, 0, 5, 1, -7, -6, -1, 8, 9, -1, -7,
	0, 0, 6, 1, 10, 3, 10, -3, 0, -9, -1, 6, 10,
	-5, -6, -6, -5, -6, 6, -9, 9, 9, 6, 5, -5, 5,
	-7, -10, 9, 0, -9, -8, -3, -6, 1, 5, 1, 4, -5,
	-4, -6, 2, 8, 5, 4, 3, 5, -8, -6, -6, -7, 9,
	2, 0, 7, -5, 7, -6, 0, -10, -1, -8, 9, 6, -10,
	-7, -6, -2, -8, 10, 10, 3, -2, 10, -6, -4, -7, -3,
	0, -3, 2, -9, 6, -2, -8, 2, -7, -10, 1, -7, 6,
	-3, 10, 9, 9, -6, -3, -7, -9, -10, -9, 4, -8, 1,
	-4, 2, -9, 9, 2, 2, -7, 4, -7, 5, -6, -2, -3,
	-10, -2, 10, 3, -8, -2, -2, -1, -8, -6, 0, 9, -8,
	-3, 1, 0, 9, -6, 0, -7, 0, -5, -1, 3, -1, -9,
	3, -3, -6, 3, 2, 0,
}))

var argmaxCorrect = []struct {
	correctShape types.Shape
	correctData  []int
}{
	{types.Shape{3, 4, 5, 2}, []int{
		0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1,
		0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1,
		0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
		0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1,
		0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0,
		1, 0, 1, 1, 0,
	}},
	{types.Shape{2, 4, 5, 2}, []int{
		2, 1, 2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 0,
		2, 1, 0, 0, 0, 0, 0, 1, 2, 2, 0, 2, 0, 2, 0, 1, 1, 1, 1, 2, 1, 1, 0,
		0, 0, 0, 1, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 0, 1, 0, 1, 1, 1, 2, 0, 2,
		0, 2, 2, 1, 0, 0, 2, 1, 2, 1, 1,
	}},
	{types.Shape{2, 3, 5, 2}, []int{
		1, 1, 2, 3, 1, 3, 0, 2, 0, 2, 2, 1, 3, 0, 1, 1, 1, 0, 3, 3, 0, 3, 3,
		2, 3, 1, 3, 1, 0, 3, 0, 3, 2, 3, 3, 2, 0, 2, 0, 2, 0, 0, 3, 2, 0, 2,
		1, 2, 3, 3, 1, 1, 0, 2, 2, 1, 2, 3, 3, 1,
	}},
	{types.Shape{2, 3, 4, 2}, []int{
		3, 3, 0, 0, 1, 3, 1, 1, 2, 1, 2, 2, 2, 0, 1, 2, 0, 3, 3, 2, 1, 1, 1,
		0, 3, 2, 2, 1, 1, 2, 2, 1, 0, 0, 3, 1, 2, 1, 4, 4, 2, 4, 0, 4, 2, 1,
		2, 0,
	}},
	{types.Shape{2, 3, 4, 5}, []int{
		0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
		1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1,
		1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1,
		1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1,
		0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0,
		1, 0, 0, 1, 0,
	}},
}

func TestArgmax(t *testing.T) {
	assert := assert.New(t)
	var T *Tensor
	var argmax *Tensor
	var err error

	T = basicArgT.Clone()
	for i := 0; i < T.Dims(); i++ {
		if argmax, err = T.Argmax(i); err != nil {
			t.Error(err)
			continue
		}

		assert.True(argmaxCorrect[i].correctShape.Eq(argmax.Shape()), "Argmax(%d) error. Want shape %v, got %v", i, argmaxCorrect[i].correctShape, argmax.Shape())
		assert.Equal(argmaxCorrect[i].correctData, argmax.Data(), "Argmax(%d) error. Want data %v, got %v", i, argmaxCorrect[i].correctData, argmax.Data())
	}

	// test all axes
	if argmax, err = T.Argmax(types.AllAxes); err != nil {
		t.Error(err)
		return
	}

	assert.True(argmax.IsScalar())
	assert.Equal(6, argmax.ScalarValue())

	// idiotsville
	_, err = T.Argmax(10)
	assert.NotNil(err)
}

var argminCorrect = []struct {
	correctShape types.Shape
	correctData  []int
}{
	{types.Shape{3, 4, 5, 2}, []int{
		1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0,
		1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0,
		1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1,
		1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0,
		1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1,
		0, 1, 0, 0, 1,
	}},
	{types.Shape{2, 4, 5, 2}, []int{
		1, 2, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 0, 1, 2, 0, 1, 0, 0, 0, 2, 0, 1,
		1, 2, 1, 2, 2, 2, 1, 2, 1, 0, 1, 0, 2, 0, 1, 2, 0, 0, 0, 0, 2, 0, 1,
		1, 1, 2, 0, 0, 0, 1, 2, 1, 1, 2, 1, 1, 0, 2, 0, 1, 2, 0, 2, 1, 2, 1,
		1, 0, 1, 0, 2, 1, 0, 0, 1, 0, 0,
	}},
	{types.Shape{2, 3, 5, 2}, []int{
		2, 2, 1, 1, 3, 0, 3, 1, 3, 1, 0, 3, 0, 3, 3, 2, 2, 3, 2, 0, 2, 2, 1,
		0, 0, 0, 2, 3, 3, 0, 3, 0, 0, 0, 0, 3, 3, 3, 1, 3, 3, 3, 1, 3, 1, 1,
		2, 3, 1, 2, 2, 0, 1, 3, 1, 0, 1, 2, 0, 2,
	}},
	{types.Shape{2, 3, 4, 2}, []int{
		0, 2, 1, 3, 4, 0, 2, 4, 0, 2, 0, 1, 3, 2, 2, 3, 2, 2, 0, 0, 0, 2, 4,
		3, 1, 0, 4, 0, 0, 0, 0, 2, 3, 3, 4, 2, 0, 4, 0, 0, 4, 0, 1, 3, 0, 3,
		3, 1,
	}},
	{types.Shape{2, 3, 4, 5}, []int{
		1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
		0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0,
		0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0,
		0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0,
		0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1,
		0, 1, 1, 0, 1,
	}},
}

func TestArgmin(t *testing.T) {
	assert := assert.New(t)
	var T *Tensor
	var argmin *Tensor
	var err error

	T = basicArgT.Clone()
	for i := 0; i < T.Dims(); i++ {
		if argmin, err = T.Argmin(i); err != nil {
			t.Error(err)
			continue
		}

		assert.True(argminCorrect[i].correctShape.Eq(argmin.Shape()), "Argmax(%d) error. Want shape %v, got %v", i, argminCorrect[i].correctShape, argmin.Shape())
		assert.Equal(argminCorrect[i].correctData, argmin.Data(), "Argmax(%d) error. Want data %v, got %v", i, argminCorrect[i].correctData, argmin.Data())
	}

	// test all axes
	if argmin, err = T.Argmin(types.AllAxes); err != nil {
		t.Error(err)
		return
	}

	assert.True(argmin.IsScalar())
	assert.Equal(17, argmin.ScalarValue())

	// idiotsville
	_, err = T.Argmin(10)
	assert.NotNil(err)
}
