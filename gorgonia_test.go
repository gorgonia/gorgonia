package gorgonia

import (
	"testing"

	tf64 "github.com/chewxy/gorgonia/tensor/f64"
	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/stretchr/testify/assert"
)

func TestNewConstant(t *testing.T) {
	assert := assert.New(t)

	var expectedType Type

	t.Log("Testing New Constant Tensors")
	backing := tf64.RandomFloat64(9)
	T := tf64.NewTensor(tf64.WithBacking(backing), tf64.WithShape(3, 3))

	ct := NewConstant(T)
	expectedTT := newTensorType(2, Float64)
	expectedTT.shape = types.Shape{3, 3}
	expectedType = expectedTT

	assert.Equal(types.Shape{3, 3}, ct.shape)
	assert.Equal(expectedType, ct.t)

	TV := FromTensor(T)
	ct = NewConstant(TV, WithName("From TensorValue"))
	assert.Equal(types.Shape{3, 3}, ct.shape)
	assert.Equal(expectedType, ct.t)
	assert.Equal("From TensorValue", ct.name)

	t.Log("Testing Constant Scalars")
	cs := NewConstant(3.14)
	expectedType = Float64
	assert.Equal(scalarShape, cs.shape)
	assert.Equal(expectedType, cs.t)
}
