package gorgonia

import (
	"fmt"

	tf32 "github.com/chewxy/gorgonia/tensor/f32"
	tf64 "github.com/chewxy/gorgonia/tensor/f64"
	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/chewxy/hm"
	"github.com/pkg/errors"
)

// Functions in this file returns *Node and panics if an error happens

/* Helper functions to create new input nodes */

// Must indicates a node must be created. If there isn't a node created, or there was an error,
// it subsumes the error, and immediately panics
func Must(n *Node, err error, opts ...NodeConsOpt) *Node {
	if err != nil || n == nil {
		panic(err)
	}
	return n
}

// NodeFromAny creates a Node from a types.Tensor, automatically filling in shape and type info
func NodeFromAny(g *ExprGraph, any interface{}, opts ...NodeConsOpt) *Node {
	v, t, dt, err := anyToValue(any)
	if err != nil {
		panic(err)
	}

	opts = append(opts, WithValue(v))

	switch t.(type) {
	case Dtype:
		return NewScalar(g, dt, opts...)
	case TensorType:
		opts = append(opts, nil)
		copy(opts[1:], opts[0:len(opts)-1])
		opts[0] = WithShape(v.Shape()...)
		return NewTensor(g, dt, v.Shape().Dims(), opts...)
	default:
		panic(nyi("NewNodeFromAny", any))
	}
}

// NewScalar creates a Node representing a variable that holds a scalar value
func NewScalar(g *ExprGraph, t Dtype, opts ...NodeConsOpt) *Node {
	curOpts := []NodeConsOpt{withType(t), withGraph(g), WithShape()}
	curOpts = append(curOpts, opts...)

	return newUniqueNode(curOpts...)
}

// NewVector creates a Node representing a variable that holds a vector (nx1 matrix)
func NewVector(g *ExprGraph, t Dtype, opts ...NodeConsOpt) *Node {
	tt := newTensorType(1, t)
	curOpts := []NodeConsOpt{withType(tt), withGraph(g)}
	curOpts = append(curOpts, opts...)

	return newUniqueNode(curOpts...)
}

// NewMatrix creates a Node representing a variable that holds a matrix (nxm)
func NewMatrix(g *ExprGraph, t Dtype, opts ...NodeConsOpt) *Node {
	tt := newTensorType(2, t)
	curOpts := []NodeConsOpt{withType(tt), withGraph(g)}
	curOpts = append(curOpts, opts...)

	return newUniqueNode(curOpts...)
}

// NewTensor creates a Node representing a variable that holds a tensor (any n-dimensional array with dimensions greater than 2)
func NewTensor(g *ExprGraph, t Dtype, dims int, opts ...NodeConsOpt) *Node {
	tt := newTensorType(dims, t)
	curOpts := []NodeConsOpt{withType(tt), withGraph(g)}
	curOpts = append(curOpts, opts...)

	return newUniqueNode(curOpts...)
}

// NewConstant takes in any reasonable value and makes it a constant node.
func NewConstant(v interface{}, opts ...NodeConsOpt) *Node {
	var op Op
	var t hm.Type
	var name string
	var s types.Shape
	var val Value

	name = fmt.Sprintf("%v", v)
	switch a := v.(type) {
	case Scalar:
		op = constantScalar{a}
		val = a
		t = TypeOf(a)
		s = scalarShape
	case int, int64, float64, float32, byte, bool:
		val, t = anyToScalar(v)
		s = scalarShape
		op = constantScalar{val.(Scalar)}
	case types.Tensor:
		op = constantTensor{a}
		val = a
		s = a.Shape()
		t = TypeOf(a)
	}

	if op == nil || t == nil {
		panic(fmt.Sprintf("HELP. Op: %v, t: %v", op, t))
	}

	consOpts := []NodeConsOpt{withOp(op), withType(t), WithName(name), WithShape(s...), WithValue(val)}
	consOpts = append(consOpts, opts...)
	return newNode(consOpts...)
}

// UniformRandomNode creates an input node that has a random op so everytime the node is passed, random values will be plucked from
// a uniform distribution. The type of the node depends on the
// shape passed in. To get a scalar value at run time, don't pass in any shapes
func UniformRandomNode(g *ExprGraph, dt Dtype, low, high float64, shape ...int) *Node {
	op := makeRandomOp(uniform, dt, low, high, shape...)
	s := types.Shape(shape)

	var t hm.Type
	if s.Eq(scalarShape) {
		t = dt
	} else {
		t = newTensorType(s.Dims(), dt)
	}

	retVal := newUniqueNode(withType(t), withOp(op), withGraph(g), WithShape(shape...))
	return retVal
}

// GaussianRandomNode creates an input node that has a random op so everytime the node is passed, random values will be plucked from
// a gaussian distribution with the mean and stdev provided. The type of the node depends on the
// shape passed in. To get a scalar value at run time, don't pass in any shapes
func GaussianRandomNode(g *ExprGraph, dt Dtype, mean, stdev float64, shape ...int) *Node {
	op := makeRandomOp(gaussian, dt, mean, stdev, shape...)
	s := types.Shape(shape)

	var t hm.Type
	if s.Eq(scalarShape) {
		t = dt
	} else {
		t = newTensorType(s.Dims(), dt)
	}

	retVal := newUniqueNode(withType(t), withOp(op), withGraph(g), WithShape(shape...))
	return retVal
}

// BinomialRandomNode creates an input node that has a random op so that everytime the node is passed, random values will be plucked from
// a binomial distribution with the mean and stdev provided. The type of the node depends on the
// shape passed in. To get a scalar value at run time, don't pass in any shapes
//
// Whilst technically the number of trials of a binomal distribution should be a discrete value (you can't have half a trial), to keep with
// API uniformity, trials is passed in as a float64, but will be truncated to an int at runtime.
func BinomialRandomNode(g *ExprGraph, dt Dtype, trials, prob float64, shape ...int) *Node {
	op := makeRandomOp(binomial, dt, trials, prob, shape...)
	s := types.Shape(shape)

	var t hm.Type
	if s.Eq(scalarShape) {
		t = dt
	} else {
		t = newTensorType(s.Dims(), dt)
	}

	retVal := newUniqueNode(withType(t), withOp(op), withGraph(g), WithShape(shape...))
	return retVal
}

// OneHotVector creates a node representing a one hot vector
func OneHotVector(id, classes int, t Dtype, opts ...NodeConsOpt) *Node {
	switch t {
	case Float64:
		backing := make([]float64, classes)
		backing[id] = float64(1)
		T := tf64.NewTensor(tf64.WithBacking(backing))
		// dt, d := tensorInfo(T)
		return NewConstant(T, opts...)
	case Float32:
		backing := make([]float32, classes)
		backing[id] = float32(1)
		T := tf32.NewTensor(tf32.WithBacking(backing))
		return NewConstant(T, opts...)
	default:
		panic("Not yet implemented for OneHotVector")
	}
	panic("unreachable")
}

// Grad takes a scalar cost node and a list of with-regards-to, and returns the gradient
func Grad(cost *Node, WRTs ...*Node) (retVal []*Node, err error) {
	symdiffLogf("Cost:%v", cost)
	if !cost.IsScalar() {
		return nil, errors.Errorf("Expected Cost to be a scalar. Got %v instead", cost)
	}

	for i, n := range WRTs {
		if !n.isInput() {
			errors.Wrapf(err, "Can only differentiate with regards to input nodes. Node %d isn't an input", i)
			// return
		}
	}

	var dt Dtype
	var ok bool
	if dt, ok = cost.t.(Dtype); !ok {
		errors.Wrap(err, "Expected a scalar dtype for cost")
		// return
	}

	var gradOut *Node
	switch dt {
	case Float64:
		gradOut = onef64
	case Float32:
		gradOut = onef32
	default:
		return nil, errors.Wrapf(err, "%s not yet implemented for %v of %T", dt.String(), "Grad()'s gradOut")
	}

	gradOut = cost.g.AddNode(gradOut)
	return Backpropagate(Nodes{cost}, Nodes{gradOut}, Nodes(WRTs))
}

// Let binds a Value to a node that is a variable. A variable is represented as a *Node with no Op.
// It is equivalent to :
//		x = 2
func Let(n *Node, be interface{}) (err error) {
	if !n.isInput() {
		return errors.New("Cannot bind a value to a non input node")
	}

	var val Value
	// var t hm.Type
	// var dt Dtype
	if val, _, _, err = anyToValue(be); err != nil {
		return errors.Wrapf(err, anyToValueFail, be, be)
	}

	// TODO: runtime type checking
	n.bind(val)
	return
}

// Set is the equivalent of doing this:
//		a = b
// where a and b are both variables
func Set(a, b *Node) (retVal *Node) {
	op := letOp{}
	name := fmt.Sprintf("%v %s %v", a, op, b)
	return newUniqueNode(withOp(op), withChildren(Nodes{a, b}), WithName(name), withGraph(a.g))
}

// Read is one of those special snowflake tumblrina *Nodes. It allows for extraction of the value of the *Node at runtime
// into a Value. Note that a *Value (a pointer to a Value) is passed into this function, not a Value.
func Read(n *Node, into *Value) (retVal *Node) {
	op := readOp{into}
	name := fmt.Sprintf("read %v into %v", n, into)
	retVal = newUniqueNode(withOp(op), withChildren(Nodes{n}), WithName(name), withGraph(n.g))
	retVal.op = op // this ensures the correct pointer is written
	retVal.name = name
	return
}
