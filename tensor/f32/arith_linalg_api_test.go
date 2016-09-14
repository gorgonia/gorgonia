package tensorf32

import (
	"testing"

	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/stretchr/testify/assert"
)

func TestDot(t *testing.T) {
	assert := assert.New(t)
	var a, b, c, r *Tensor
	var A, B, R *Tensor
	var err error
	var expectedShape types.Shape
	var expectedData []float32

	// vector-vector
	a = NewTensor(WithShape(3, 1), WithBacking(RangeFloat32(0, 3)))
	b = NewTensor(WithShape(3, 1), WithBacking(RangeFloat32(0, 3)))
	r, err = Dot(a, b)

	expectedShape = types.Shape{1}
	expectedData = []float32{5}
	assert.Nil(err)
	assert.Equal(expectedData, r.data)
	assert.True(types.ScalarShape().Eq(r.Shape()))

	// vector-mat (which is the same as mat'*vec)
	A = NewTensor(WithShape(3, 2), WithBacking(RangeFloat32(0, 6)))
	R, err = Dot(b, A)

	expectedShape = types.Shape{2, 1}
	expectedData = []float32{10, 13}
	assert.Nil(err)
	assert.Equal(expectedData, R.data)
	assert.Equal(expectedShape, R.Shape())

	/* HERE BE STUPIDS */

	// different sizes of vectors
	c = NewTensor(WithShape(1, 4))
	_, err = Dot(a, c)
	assert.NotNil(err)

	// vector mat, but with shape mismatch
	B = NewTensor(WithShape(2, 3), WithBacking(RangeFloat32(0, 6)))
	_, err = Dot(b, B)
	assert.NotNil(err)
}
