package gorgonia

import (
	"fmt"
	"hash"
	"hash/fnv"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia/internal/execution"
	"gorgonia.org/gorgonia/internal/value"
	"gorgonia.org/tensor"
)

// DimSizer is any type (typically a tensor.Shape) that allows querying for a dimension size given an input dimension.
type DimSizer interface {
	DimSize(int) (int, error)
}

// ShapesToDimSizers is a convenience function to convert a slice of tensor.Shape to a slice of DimSizer
func ShapesToDimSizers(shapes []tensor.Shape) []DimSizer {
	retVal := make([]DimSizer, len(shapes))
	for i, s := range shapes {
		retVal[i] = s
	}
	return retVal
}

// DimSizersToShapes is a convenience function to convert a slice of DimSizer to a slice of tensor.Shape. It will return an error if any of them isn't a tensor.Shape
func DimSizersToShapes(ds []DimSizer) ([]tensor.Shape, error) {
	retVal := make([]tensor.Shape, len(ds))
	var ok bool
	for i, d := range ds {
		if retVal[i], ok = d.(tensor.Shape); !ok {
			return nil, errors.Errorf("Dimsizer %d is not a Shape.", i)
		}
	}
	return retVal, nil
}

// An Op is a symbolic representation of an operation
// Think of them as functions, taking an input (or multiple), and outputting something
//
// All Ops have type signatures that look like this:
//		OpName :: (Floats a) ⇒ Tensor a → Tensor a → Tensor a
type Op interface {
	/* Graph Building Related Methods */

	// Arity returns the number of inputs the Op expects. -1 indicates that it's n-ary and will be determined at runtime
	Arity() int

	// Informs the type of the Op (not the node). This will be used by the type system to infer the final type of the node
	Type() hm.Type

	// returns the output shape as a function of the inputs
	InferShape(...DimSizer) (tensor.Shape, error)

	/* Machine related */

	// executes the op
	Do(...value.Value) (value.Value, error)

	/* Analysis Related Methods */

	// indicates if the Op will return a pointer (allowing possible inplace edits) or by value
	// if it's false, the return value of the Op will be a copy of its input
	ReturnsPtr() bool

	// Does this op potentially call external (cgo or cuda) functions (thereby requiring extra overhead for Go's trampolining thing)
	CallsExtern() bool

	// overwriteInput() is a method which states which input the output will be overwriting.
	// This allows for some efficiency gains as the underlying arrays wouldn't have to be re-allocated.
	// The method returns an int instead of a bool because potentially different operations may be allowed
	// to overwrite certain inputs. For example, consider an operation to increment a value:
	// the IncrementOp would be a unary operator, and assuming we would like to overwrite the input,
	// the retVal of overwriteInput() will be 0 (inputs[0]).
	// -1 is returned if overwriting of input is disallowed
	OverwritesInput() int

	/* Other methods */
	WriteHash(h hash.Hash)
	Hashcode() uint32
	fmt.Stringer
}

// A UnaryOp is an Op that takes only one input
type UnaryOp interface {
	Op

	IsUnary() bool
}

// A BinaryOp is an Op that takes only two inputs
type BinaryOp interface {
	Op

	IsBinary() bool
}

// BestDoer ...
type BestDoer interface {
	Op

	BestDo(prealloc value.Value, vals ...value.Value) (value.Value, error)
}

// A NoRetOp is an Op that reads a value, but does not return any value. It's a representation of a not-pure function
type NoRetOp interface {
	Op

	ReturnsNothing() bool
}

// An ADOp is an Op that supports automatic differentiation.
type ADOp interface {
	Op

	DoDiff(ctx execution.Context, inputs Nodes, output *Node) error
}

// A SDOp is an Op that supports symbolic differentiation
type SDOp interface {
	Op

	// DiffWRT indicates if the op is differentiable with regards to the given number of inputs
	// returns []bool to indicate which input it is differentiable to
	DiffWRT(inputs int) []bool

	// SymDiff symbolically differentiates the op
	SymDiff(inputs Nodes, output, grad *Node) (retVal Nodes, err error)
}

// ReductionOp changes the shape of the node
type ReductionOp interface {
	Op

	IsReduction() bool
}

// IncrDoer increments the toIncr with the result of doing
type IncrDoer interface {
	IncrDo(toIncr value.Value, inputs ...value.Value) error
}

// UsePreallocDoer is an op that works when a preallocated value is provided
type UsePreallocDoer interface {
	UsePreallocDo(prealloc value.Value, inputs ...value.Value) (value.Value, error)
}

// UnsafeDoer is an op that will overwrite the underlying value.
type UnsafeDoer interface {
	UnsafeDo(inputs ...value.Value) (value.Value, error)
}

// CUDADoer uses CUDA to perform the Op.
type CUDADoer interface {
	CUDADo(extern execution.External, dev execution.Device, prealloc value.Value, inputs ...value.Value) (retVal value.Value, err error)
}

// CLDoer uses OpenCL to perform the Op. As of now, there are NO Ops that support OpenCL
type CLDoer interface {
	CLDo(inputs ...value.Value) (value.Value, error)
}

// CUDAADOp ...
type CUDAADOp interface {
	ADOp
	CUDADoDiff(extern execution.External, dev execution.Device, inputs Nodes, output *Node) error
}

// ApplyOp op to the node n. The children are extracted from the Graph g
func (g *ExprGraph) ApplyOp(op Op, n *Node) error {
	var children []*Node
	child := getOrderedChildren(g, n)
	if child != nil {
		//child := g.From(n.ID())
		children = make([]*Node, child.Len())
		for i := 0; child.Next(); i++ {
			children[i] = child.Node().(*Node)
		}
	}

	// TODO get children from the node
	// typecheck  before creating
	typeSysLogf("Inferring node type of %v :: %v with children: %#Y", op, op.Type(), Nodes(children))
	enterLogScope()
	defer leaveLogScope()
	var retType hm.Type
	retType, err := inferNodeType(op, children...)
	if err != nil {
		return errors.Wrapf(err, "Type inference error. Op: %v. Children: %#Y, OpType:%v", op, Nodes(children), op.Type())
	}
	typeSysLogf("Done inferring. Return type is: %#v(%T)", retType, retType)

	// infer shapes, but print errors instead of returning
	shapeLogf("op: %v(%T) inferring shape", op, op)
	err = checkArity(op, len(children))
	if err != nil {
		return err
	}

	ds := Nodes(children).dimSizers()
	s, err := op.InferShape(ds...)
	if err != nil {
		return errors.Wrapf(err, "Failed to infer shape. Op: %v", op)
	}
	shapeLogf("inferred shape %v", s)
	WithType(retType)(n)
	WithOp(op)(n)
	WithShape(s...)(n)
	returnDimSizers(ds)
	return nil
}

// ApplyOp is the generic function application - for when no specialization is required
func ApplyOp(op Op, children ...*Node) (*Node, error) {
	var g *ExprGraph

	for _, child := range children {
		if child.g != nil {
			g = child.g
			break
		}
	}

	if g == nil {
		return nil, errors.New("No Graph Supplied")
	}

	if !Nodes(children).AllSameGraph() {
		return nil, errors.New("Not all children have the same graph")
	}

	/*
		// typecheck  before creating
		typeSysLogf("Inferring node type of %v :: %v with children: %#Y", op, op.Type(), Nodes(children))
		enterLogScope()
		defer leaveLogScope()
		var retType hm.Type
		retType, err := inferNodeType(op, children...)
		if err != nil {
			return nil, errors.Wrapf(err, "Type inference error. Op: %v. Children: %#Y, OpType:%v", op, Nodes(children), op.Type())
		}
		typeSysLogf("Done inferring. Return type is: %#v(%T)", retType, retType)

		// infer shapes, but print errors instead of returning
		shapeLogf("op: %v(%T) inferring shape", op, op)
		err = checkArity(op, len(children))
		if err != nil {
			return nil, err
		}

		ds := Nodes(children).dimSizers()
		s, err := op.InferShape(ds...)
		if err != nil {
			return nil, errors.Wrapf(err, "Failed to infer shape. Op: %v", op)
		}
		shapeLogf("inferred shape %v", s)
	*/
	n := g.NewNode().(*Node)
	for i, child := range children {
		g.SetWeightedEdge(g.NewWeightedEdge(n, child, float64(i)))
	}
	g.ApplyOp(op, n)
	/*
		WithType(retType)(n)
		WithOp(op)(n)
		WithShape(s...)(n)
		returnDimSizers(ds)
		g.AddNode(n)
	*/
	return n, nil
}

// ApplyOpWithName applies the op, and then gives the node the given name
func ApplyOpWithName(op Op, name string, children ...*Node) (retVal *Node, err error) {
	if retVal, err = ApplyOp(op, children...); err == nil {
		WithName(name)(retVal)
	} else {
		return nil, errors.Wrap(err, applyOpFail)
	}
	return
}

// a constant is an unchanging value. I think everyone would know what a constant is
// a constant op is an op that creates a constant. It is also a value.Value of a constant value
type constant interface {
	Op

	isconstant() bool
	Value() value.Value
}

type constantScalar struct {
	v value.Scalar
}

func (c constantScalar) Arity() int                                   { return 0 }
func (c constantScalar) Type() hm.Type                                { return value.TypeOf(c.v) }
func (c constantScalar) InferShape(...DimSizer) (tensor.Shape, error) { return scalarShape, nil }
func (c constantScalar) ReturnsPtr() bool                             { return false }
func (c constantScalar) CallsExtern() bool                            { return false }
func (c constantScalar) OverwritesInput() int                         { return -1 }
func (c constantScalar) DiffWRT(i int) []bool                         { return nil }
func (c constantScalar) SymDiff(Nodes, *Node, *Node) (Nodes, error)   { return nil, nil }

func (c constantScalar) Do(...value.Value) (value.Value, error) { return c.v, nil }
func (c constantScalar) String() string                         { return fmt.Sprintf("const %s", c.v) }

func (c constantScalar) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "const %v: %v", value.TypeOf(c.v), c.v)
}

func (c constantScalar) Hashcode() uint32 {
	h := fnv.New32a()
	c.WriteHash(h)
	return h.Sum32()
}

func (c constantScalar) isconstant() bool   { return true }
func (c constantScalar) Value() value.Value { return c.v }

type constantTensor struct {
	v tensor.Tensor
}

func (c constantTensor) Arity() int                                   { return 1 }
func (c constantTensor) Type() hm.Type                                { return value.TypeOf(c.v) }
func (c constantTensor) InferShape(...DimSizer) (tensor.Shape, error) { return c.v.Shape(), nil }

// danger! The only reason why this is the case is because matrices may be too large. copying is costly.
// constants should return value but for the sake of memory, we're going to return pointers
func (c constantTensor) ReturnsPtr() bool                           { return true }
func (c constantTensor) OverwritesInput() int                       { return -1 }
func (c constantTensor) CallsExtern() bool                          { return false }
func (c constantTensor) DiffWRT(i int) []bool                       { return nil }
func (c constantTensor) SymDiff(Nodes, *Node, *Node) (Nodes, error) { return nil, nil }
func (c constantTensor) Do(...value.Value) (value.Value, error)     { return c.v, nil }
func (c constantTensor) String() string                             { return fmt.Sprintf("const %s", value.TypeOf(c.v)) }

func (c constantTensor) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "const %v:%v", c.Type(), c.v)
}

func (c constantTensor) Hashcode() uint32 {
	h := fnv.New32a()
	c.WriteHash(h)
	return h.Sum32()
}

func (c constantTensor) isconstant() bool   { return true }
func (c constantTensor) Value() value.Value { return c.v }
