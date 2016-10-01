package gorgonia

import (
	"testing"

	tf32 "github.com/chewxy/gorgonia/tensor/f32"
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

var anyNodeTest = []struct {
	name string
	any  interface{}

	correctType  Type
	correctShape types.Shape
}{
	{"float32", float32(3.14), Float32, scalarShape},
	{"float64", float64(3.14), Float64, scalarShape},
	{"int", int(3), Int, scalarShape},
	{"bool", true, Bool, scalarShape},
	{"tf64.Tensor", tf64.NewTensor(tf64.WithShape(2, 3, 4)), &TensorType{d: 3, of: Float64}, types.Shape{2, 3, 4}},
	{"tf32.Tensor", tf32.NewTensor(tf32.WithShape(2, 3, 4)), &TensorType{d: 3, of: Float32}, types.Shape{2, 3, 4}},
	{"ScalarValue", Scalar{v: 3.14, t: Float64}, Float64, scalarShape},
	{"TensorValue", Tensor{Tensor: tf64.NewTensor(tf64.WithShape(2, 3))}, &TensorType{d: 2, of: Float64}, types.Shape{2, 3}},
}

func TestNewNodeFromAny(t *testing.T) {
	assert := assert.New(t)
	g := NewGraph()
	for _, a := range anyNodeTest {
		n := NewNodeFromAny(g, a.any, WithName(a.name))
		assert.Equal(a.name, n.name)
		assert.Equal(g, n.g)
		assert.True(typeEq(a.correctType, n.t), "%v type error: Want %v. Got %v", a.name, a.correctType, n.t)
		assert.True(a.correctShape.Eq(n.shape), "%v shape error: Want %v. Got %v", a.name, a.correctShape, n.shape)
	}
}
