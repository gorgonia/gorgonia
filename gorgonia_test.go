package gorgonia

import (
	"testing"

	tf32 "github.com/chewxy/gorgonia/tensor/f32"
	tf64 "github.com/chewxy/gorgonia/tensor/f64"
	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/chewxy/hm"
	"github.com/stretchr/testify/assert"
)

func TestNewConstant(t *testing.T) {
	assert := assert.New(t)

	var expectedType hm.Type

	t.Log("Testing New Constant Tensors")
	backing := tf64.RandomFloat64(9)
	T := tf64.NewTensor(tf64.WithBacking(backing), tf64.WithShape(3, 3))

	ct := NewConstant(T)
	expectedTT := newTensorType(2, Float64)
	expectedType = expectedTT

	assert.Equal(types.Shape{3, 3}, ct.shape)
	assert.Equal(expectedType, ct.t)

	ct = NewConstant(T, WithName("From TensorValue"))
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

	correctType  hm.Type
	correctShape types.Shape
}{
	{"float32", float32(3.14), Float32, scalarShape},
	{"float64", float64(3.14), Float64, scalarShape},
	{"int", int(3), Int, scalarShape},
	{"bool", true, Bool, scalarShape},
	{"tf64.Tensor", tf64.NewTensor(tf64.WithShape(2, 3, 4)), &TensorType{Dims: 3, Of: Float64}, types.Shape{2, 3, 4}},
	{"tf32.Tensor", tf32.NewTensor(tf32.WithShape(2, 3, 4)), &TensorType{Dims: 3, Of: Float32}, types.Shape{2, 3, 4}},
	{"ScalarValue", F64(3.14), Float64, scalarShape},
	{"TensorValue", tf64.NewTensor(tf64.WithShape(2, 3)), &TensorType{Dims: 2, Of: Float64}, types.Shape{2, 3}},
}

func TestNodeFromAny(t *testing.T) {
	assert := assert.New(t)
	g := NewGraph()
	for _, a := range anyNodeTest {
		n := NodeFromAny(g, a.any, WithName(a.name))
		assert.Equal(a.name, n.name)
		assert.Equal(g, n.g)
		assert.True(a.correctType.Eq(n.t), "%v type error: Want %v. Got %v", a.name, a.correctType, n.t)
		assert.True(a.correctShape.Eq(n.shape), "%v shape error: Want %v. Got %v", a.name, a.correctShape, n.shape)
	}
}
