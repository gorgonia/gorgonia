package engine

import (
	"fmt"
	"hash"
	"hash/fnv"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia/internal/execution"
	"gorgonia.org/gorgonia/internal/value"
	"gorgonia.org/gorgonia/ops"
	"gorgonia.org/tensor"
)

// ShapesToDimSizers is a convenience function to convert a slice of tensor.Shape to a slice of DimSizer
func ShapesToDimSizers(shapes []tensor.Shape) []ops.DimSizer {
	retVal := make([]ops.DimSizer, len(shapes))
	for i, s := range shapes {
		retVal[i] = s
	}
	// END_OPERATION OMIT
	return retVal
}

// DimSizersToShapes is a convenience function to convert a slice of DimSizer to a slice of tensor.Shape. It will return an error if any of them isn't a tensor.Shape
func DimSizersToShapes(ds []ops.DimSizer) ([]tensor.Shape, error) {
	retVal := make([]tensor.Shape, len(ds))
	var ok bool
	for i, d := range ds {
		if retVal[i], ok = d.(tensor.Shape); !ok {
			return nil, errors.Errorf("Dimsizer %d is not a Shape.", i)
		}
	}
	return retVal, nil
}

// An ADOp is an Op that supports automatic differentiation.
type ADOp interface {
	ops.Op

	DoDiff(ctx execution.Context, inputs Nodes, output *Node) error
}

// A SDOp is an Op that supports symbolic differentiation
type SDOp interface {
	ops.Op

	// DiffWRT indicates if the op is differentiable with regards to the given number of inputs
	// returns []bool to indicate which input it is differentiable to
	DiffWRT(inputs int) []bool

	// SymDiff symbolically differentiates the op
	SymDiff(inputs Nodes, output, grad *Node) (retVal Nodes, err error)
}

// CUDAADOp ...
type CUDAADOp interface {
	ADOp
	CUDADoDiff(extern execution.External, dev execution.Device, inputs Nodes, output *Node) error
}

// START_APPLY OMIT
func (g *ExprGraph) applyOp(op ops.Op, n *Node) error {

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
	err = ops.CheckArity(op, len(children))
	if err != nil {
		return err
	}

	ds := Nodes(children).dimSizers()
	s, err := op.InferShape(ds...)
	if err != nil {
		return errors.Wrapf(err, "Failed to infer shape. Op: %v", op)
	}
	shapeLogf("inferred shape %v", s)
	n.t = retType
	n.op = op
	n.shape = s
	//WithType(retType)(n)
	//WithOp(op)(n)
	//WithShape(s...)(n)
	returnDimSizers(ds)
	return nil
}

// END_APPLY OMIT
// START_APPLYOP OMIT

// ApplyOp op to the node n. The children are extracted from the Graph g
func (g *ExprGraph) ApplyOp(operation Operation, n *Node) error {
	opfn, err := operation(g, n)
	if err != nil {
		return err
	}
	// It's a noop, return
	if opfn == nil {
		return nil
	}
	op, ok := opfn.(ops.Op)
	if !ok {
		return errors.New("Cannot cast operator")
	}
	return g.applyOp(op, n)
}

// END_APPLYOP OMIT

// ApplyOp is the generic function application - for when no specialization is required
func ApplyOp(op ops.Op, children ...*Node) (*Node, error) {
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

	n := g.NewNode().(*Node)
	for i, child := range children {
		g.SetWeightedEdge(g.NewWeightedEdge(n, child, float64(i)))
	}
	g.applyOp(op, n)
	return n, nil
}

// ApplyOpWithName applies the op, and then gives the node the given name
func ApplyOpWithName(op ops.Op, name string, children ...*Node) (retVal *Node, err error) {
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
	ops.Op

	isconstant() bool
	Value() value.Value
}

type constantScalar struct {
	v value.Scalar
}

func (c constantScalar) Arity() int                                       { return 0 }
func (c constantScalar) Type() hm.Type                                    { return value.TypeOf(c.v) }
func (c constantScalar) InferShape(...ops.DimSizer) (tensor.Shape, error) { return scalarShape, nil }
func (c constantScalar) ReturnsPtr() bool                                 { return false }
func (c constantScalar) CallsExtern() bool                                { return false }
func (c constantScalar) OverwritesInput() int                             { return -1 }
func (c constantScalar) DiffWRT(i int) []bool                             { return nil }
func (c constantScalar) SymDiff(Nodes, *Node, *Node) (Nodes, error)       { return nil, nil }

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

func (c constantTensor) Arity() int                                       { return 1 }
func (c constantTensor) Type() hm.Type                                    { return value.TypeOf(c.v) }
func (c constantTensor) InferShape(...ops.DimSizer) (tensor.Shape, error) { return c.v.Shape(), nil }

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
