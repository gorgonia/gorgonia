package gorgonia

import (
	"fmt"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
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

// NodeFromAny creates a Node from a tensor.Tensor, automatically filling in shape and type info
func NodeFromAny(g *ExprGraph, any interface{}, opts ...NodeConsOpt) *Node {
	v, t, dt, err := anyToValue(any)
	if err != nil {
		panic(err)
	}

	opts = append(opts, WithValue(v))

	switch t.(type) {
	case tensor.Dtype:
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
func NewScalar(g *ExprGraph, t tensor.Dtype, opts ...NodeConsOpt) *Node {
	curOpts := []NodeConsOpt{WithType(t), In(g), WithShape()}
	curOpts = append(curOpts, opts...)

	return NewUniqueNode(curOpts...)
}

// NewVector creates a Node representing a variable that holds a vector (nx1 matrix)
func NewVector(g *ExprGraph, t tensor.Dtype, opts ...NodeConsOpt) *Node {
	tt := makeTensorType(1, t)
	curOpts := []NodeConsOpt{WithType(tt), In(g)}
	curOpts = append(curOpts, opts...)

	return NewUniqueNode(curOpts...)
}

// NewMatrix creates a Node representing a variable that holds a matrix (nxm)
func NewMatrix(g *ExprGraph, t tensor.Dtype, opts ...NodeConsOpt) *Node {
	tt := makeTensorType(2, t)
	curOpts := []NodeConsOpt{WithType(tt), In(g)}
	curOpts = append(curOpts, opts...)

	return NewUniqueNode(curOpts...)
}

// NewTensor creates a Node representing a variable that holds a tensor (any n-dimensional array with dimensions greater than 2)
func NewTensor(g *ExprGraph, t tensor.Dtype, dims int, opts ...NodeConsOpt) *Node {
	tt := makeTensorType(dims, t)
	curOpts := []NodeConsOpt{WithType(tt), In(g)}
	curOpts = append(curOpts, opts...)

	return NewUniqueNode(curOpts...)
}

// NewConstant takes in any reasonable value and makes it a constant node.
func NewConstant(v interface{}, opts ...NodeConsOpt) *Node {
	var op Op
	var t hm.Type
	var name string
	var s tensor.Shape
	var val Value

	val, t, _, err := anyToValue(v)
	if err != nil {
		panic(err)
	}
	switch vt := val.(type) {
	case Scalar:
		op = constantScalar{vt}
		s = scalarShape
	case tensor.Tensor:
		op = constantTensor{vt}
		s = vt.Shape()
	}

	if op == nil || t == nil {
		panic(fmt.Sprintf("HELP. Op: %v, t: %v", op, t))
	}

	dummy := borrowNode()
	consOpts := []NodeConsOpt{WithOp(op), WithType(t), WithShape(s...), WithValue(val)}
	consOpts = append(consOpts, opts...)
	for i := range opts {
		opts[i](dummy)
	}
	if dummy.name == "" {
		name = fmt.Sprintf("%v", v)
	} else {
		name = dummy.name
	}
	returnNode(dummy)

	consOpts = append(consOpts, WithName(name))
	return newNode(consOpts...)
}

// UniformRandomNode creates an input node that has a random op so everytime the node is passed, random values will be plucked from
// a uniform distribution. The type of the node depends on the
// shape passed in. To get a scalar value at run time, don't pass in any shapes
func UniformRandomNode(g *ExprGraph, dt tensor.Dtype, low, high float64, shape ...int) *Node {
	op := makeRandomOp(uniform, dt, low, high, shape...)
	s := tensor.Shape(shape)

	var t hm.Type
	if s.Eq(scalarShape) {
		t = dt
	} else {
		t = makeTensorType(s.Dims(), dt)
	}

	retVal := NewUniqueNode(WithType(t), WithOp(op), In(g), WithShape(shape...))
	return retVal
}

// GaussianRandomNode creates an input node that has a random op so everytime the node is passed, random values will be plucked from
// a gaussian distribution with the mean and stdev provided. The type of the node depends on the
// shape passed in. To get a scalar value at run time, don't pass in any shapes
func GaussianRandomNode(g *ExprGraph, dt tensor.Dtype, mean, stdev float64, shape ...int) *Node {
	op := makeRandomOp(gaussian, dt, mean, stdev, shape...)
	s := tensor.Shape(shape)

	var t hm.Type
	if s.Eq(scalarShape) {
		t = dt
	} else {
		t = makeTensorType(s.Dims(), dt)
	}

	retVal := NewUniqueNode(WithType(t), WithOp(op), In(g), WithShape(shape...))
	return retVal
}

// BinomialRandomNode creates an input node that has a random op so that everytime the node is passed, random values will be plucked from
// a binomial distribution with the mean and stdev provided. The type of the node depends on the
// shape passed in. To get a scalar value at run time, don't pass in any shapes
//
// Whilst technically the number of trials of a binomal distribution should be a discrete value (you can't have half a trial), to keep with
// API uniformity, trials is passed in as a float64, but will be truncated to an int at runtime.
func BinomialRandomNode(g *ExprGraph, dt tensor.Dtype, trials, prob float64, shape ...int) *Node {
	op := makeRandomOp(binomial, dt, trials, prob, shape...)
	s := tensor.Shape(shape)

	var t hm.Type
	if s.Eq(scalarShape) {
		t = dt
	} else {
		t = makeTensorType(s.Dims(), dt)
	}

	retVal := NewUniqueNode(WithType(t), WithOp(op), In(g), WithShape(shape...))
	return retVal
}

// OneHotVector creates a node representing a one hot vector
func OneHotVector(id, classes int, t tensor.Dtype, opts ...NodeConsOpt) *Node {
	T := tensor.New(tensor.Of(t), tensor.WithShape(classes))
	var err error
	// This is stupid, I want generics. - docmerlin
	switch t {
	case tensor.Float32:
		err = T.SetAt(float32(1), id)
	case tensor.Float64:
		err = T.SetAt(float64(1), id)
	case tensor.Int64:
		err = T.SetAt(int64(1), id)
	case tensor.Int:
		err = T.SetAt(int(1), id)
	case tensor.Int32:
		err = T.SetAt(int32(1), id)
	default:
		panic("tensor.Dtype not implemented")
	}
	if err != nil {
		panic(err.Error())
	}
	return NewConstant(T, opts...)
}

// Grad takes a scalar cost node and a list of with-regards-to, and returns the gradient
func Grad(cost *Node, WRTs ...*Node) (retVal Nodes, err error) {
	symdiffLogf("Cost:%v", cost)
	if !cost.IsScalar() {
		return nil, errors.Errorf("Expected Cost to be a scalar. Got %v instead", cost)
	}

	for i, n := range WRTs {
		if !n.isInput() {
			err = errors.Errorf("Can only differentiate with regards to input nodes. %dth Node %v isn't an input", i, n)
			return nil, err
		}
	}

	var dt tensor.Dtype
	var ok bool
	if dt, ok = cost.t.(tensor.Dtype); !ok {
		err = errors.Wrap(err, "Expected a scalar dtype for cost")
		return
	}

	var gradOut *Node
	switch dt {
	case Float64:
		gradOut = onef64
	case Float32:
		gradOut = onef32
	default:
		return nil, errors.Wrapf(err, "%s not yet implemented for %v of %T", dt.String(), "Grad()'s gradOut", gradOut)
	}

	gradOut = cost.g.AddNode(gradOut)
	return Backpropagate(Nodes{cost}, Nodes{gradOut}, Nodes(WRTs))
}

// Let binds a Value to a node that is a variable. A variable is represented as a *Node with no Op.
// It is equivalent to :
//		x = 2
func Let(n *Node, be interface{}) error {
	if !n.isInput() {
		return errors.New("Cannot bind a value to a non input node")
	}

	return UnsafeLet(n, be)
}

// UnsafeLet binds a Value to any node, not just a variable node. This means that you can use it to change any node's value at the runtime of the graph. UNSAFE!
//
// Additional notes: if `be` is a tensor.Slice, and the node's op is a sliceOp or sliceIncrOp, the op's slice will be replaced with the new slice.
func UnsafeLet(n *Node, be interface{}) error {
	switch v := be.(type) {
	case tensor.Slice:
		switch so := n.op.(type) {
		case *sliceOp:
			so.Slice = v
			n.op = so
		case sliceIncrOp:
			so.Slice = v
			n.op = so
		default:
			return errors.Errorf("Trying to Let() a node with a slice. Node's op is %v, not sliceOp", n.op)
		}

	case Value:
		if !n.Dtype().Eq(v.Dtype()) {
			return errors.Errorf("Unable to let %v be %v. Expected Dtype of %v. Got %v instead", n.name, be, n.Dtype(), v.Dtype())
		}
		n.bind(v)
	default:
		var val Value
		var err error
		if val, _, _, err = anyToValue(be); err != nil {
			return errors.Wrapf(err, anyToValueFail, be, be)
		}

		n.bind(val)
	}
	return nil
}

// Set is the equivalent of doing this:
//		a = b
// where a and b are both variables
func Set(a, b *Node) (retVal *Node) {
	op := letOp{}
	name := fmt.Sprintf("%v %s %v", a, op, b)
	return NewUniqueNode(WithOp(op), WithChildren(Nodes{a, b}), WithName(name), In(a.g))
}

// Read allows for extraction of the value of the *Node at runtime into a Value.
// To achieve this, a pointer to a Value (*Value) is passed into this function, not a Value.
// The 'into' value remains nil until the execution of the graph (via a call to the Run() methods of the VM)
func Read(n *Node, into *Value) (retVal *Node) {
	op := readOp{into}
	name := fmt.Sprintf("read %v into %v", n, into)
	retVal = NewUniqueNode(WithOp(op), WithChildren(Nodes{n}), WithName(name), In(n.g))
	retVal.op = op // this ensures the correct pointer is written
	retVal.name = name
	return
}
