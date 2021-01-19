package gorgonia

import (
	"log"
	"os"
	"testing"

	"github.com/chewxy/hm"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gorgonia.org/tensor"
	nd "gorgonia.org/tensor"
)

func TestNewConstant(t *testing.T) {
	assert := assert.New(t)

	var expectedType hm.Type

	t.Log("Testing New Constant Tensors")
	backing := nd.Random(Float64, 9)
	T := nd.New(nd.WithBacking(backing), nd.WithShape(3, 3))

	ct := NewConstant(T)
	expectedTT := makeTensorType(2, Float64)
	expectedType = expectedTT

	assert.Equal(nd.Shape{3, 3}, ct.shape)
	assert.Equal(expectedType, ct.t)

	ct = NewConstant(T, WithName("From TensorValue"))
	assert.Equal(nd.Shape{3, 3}, ct.shape)
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
	correctShape nd.Shape
}{
	{"float32", float32(3.14), Float32, scalarShape},
	{"float64", float64(3.14), Float64, scalarShape},
	{"int", int(3), Int, scalarShape},
	{"bool", true, Bool, scalarShape},
	{"nd.Tensor", nd.New(nd.Of(nd.Float64), nd.WithShape(2, 3, 4)), &TensorType{Dims: 3, Of: Float64}, nd.Shape{2, 3, 4}},
	{"nd.Tensor", nd.New(nd.Of(nd.Float32), nd.WithShape(2, 3, 4)), &TensorType{Dims: 3, Of: Float32}, nd.Shape{2, 3, 4}},
	{"ScalarValue", newF64(3.14), Float64, scalarShape},
	{"TensorValue", nd.New(nd.Of(nd.Float64), nd.WithShape(2, 3)), &TensorType{Dims: 2, Of: Float64}, nd.Shape{2, 3}},
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

func TestOneHotVector(t *testing.T) {
	assert := assert.New(t)
	assert.EqualValues(
		[]float32{0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
		OneHotVector(6, 10, nd.Float32).Value().Data())
	assert.EqualValues(
		[]float32{0, 1, 0, 0, 0},
		OneHotVector(1, 5, nd.Float32).Value().Data())
	assert.EqualValues(
		[]float32{0, 1, 0, 0, 0, 0},
		OneHotVector(1, 6, nd.Float32).Value().Data())
	assert.EqualValues(
		[]int{0, 0, 0, 1, 0},
		OneHotVector(3, 5, nd.Int).Value().Data())
	assert.EqualValues(
		[]int32{0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
		OneHotVector(6, 10, nd.Int32).Value().Data())
	assert.EqualValues(
		[]float64{0, 1, 0, 0, 0},
		OneHotVector(1, 5, nd.Float64).Value().Data())
	assert.EqualValues(
		[]int64{0, 1, 0, 0, 0, 0},
		OneHotVector(1, 6, nd.Int64).Value().Data())
}

func TestRandomNodeBackprop(t *testing.T) {
	g := NewGraph()
	a := NewVector(g, Float64, WithShape(10), WithName("a"), WithInit(Zeroes()))
	b := GaussianRandomNode(g, Float64, 0, 1, 10)
	c := Must(Add(a, b))
	d := Must(Sum(c))
	vm := NewLispMachine(g, WithLogger(log.New(os.Stderr, "", 0)))
	vm.RunAll()
	t.Logf("d.Value %v", d.Value())
}

func TestLetErrors(t *testing.T) {
	g := NewGraph()

	testCases := []struct {
		desc string
		node *Node
		val  interface{}
		err  string
	}{
		{
			desc: "DifferentShapes",
			node: NewTensor(g, tensor.Float64, 2, WithShape(1, 1), WithInit(GlorotN(1.0)), WithName("x")),
			val:  tensor.New(tensor.WithShape(1, 1, 1), tensor.WithBacking([]float64{0.5})),
			err:  "Node's expected shape is (1, 1). Got (1, 1, 1) instead",
		},
		{
			desc: "AssigningConst",
			node: NewConstant(2, WithName("x")),
			val:  tensor.New(tensor.WithShape(1, 1), tensor.WithBacking([]float64{0.5})),
			err:  "Cannot bind a value to a non input node",
		},
	}

	for _, tC := range testCases {
		t.Run(tC.desc, func(t *testing.T) {
			err := Let(tC.node, tC.val)
			if tC.err != "" {
				require.Error(t, err)
				assert.Equal(t, tC.err, err.Error())
			} else {
				require.NoError(t, err)
			}
		})
	}
}

func TestRead(t *testing.T) {
	g := NewGraph()
	xVal := tensor.New(tensor.WithShape(2, 4), tensor.WithBacking(tensor.Range(tensor.Float64, 0, 8)))
	x := NodeFromAny(g, xVal, WithName("x"))

	var v1, v2 Value
	r1 := Read(x, &v1)
	r2 := Read(x, &v2)
	r3 := Read(x, &v1)

	assert.Equal(t, r1, r3)
	assert.NotEqual(t, r1, r2)
}
