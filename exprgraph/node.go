package exprgraph

import (
	"fmt"
	"sync/atomic"

	"github.com/chewxy/hm"
	"gonum.org/v1/gonum/graph"
	"gorgonia.org/dtype"
	"gorgonia.org/gorgonia/internal/datatypes"
	"gorgonia.org/gorgonia/ops"
	"gorgonia.org/gorgonia/types"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/gorgonia/values/dual"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/dense"
)

// constraints
var (
	_ Nodelike = Value[float64, *dense.Dense[float64]]{}

	_ Node = &Value[float64, *dense.Dense[float64]]{}
	_ Node = &Symbolic[float64]{}
)

// Node is a description of a tensor that has a name and an ID.
//
// There are two possible definition of a Node:
//   - *exprgraph.Value[DT, T]
//   - *exprgraph.Symbolic[DT]
//
// It's an interface instead of a constraint because it's more useful as an interface.
type Node interface {
	datatypes.Tensor

	graph.Node      // ID() int64
	NodeID() NodeID // alternative to ID()
	Name() string

	isnode() // seals the interface to this package
}

// New creates a new Node.
func New[DT any](g *Graph, name string, opts ...tensor.ConsOpt) Node {
	c := new(tensor.Constructor)
	for _, opt := range opts {
		opt(c)
	}
	if c.Engine == nil {
		c.Engine = g
		opts = append(opts, tensor.WithEngine(g))
	}

	// symbolic or value?
	if c.Data == nil && c.InitFn == nil {
		retVal, err := NewSymbolic[DT](g, c.Shape, name)
		if err != nil {
			panic(err)
		}
		return retVal
	}

	// get tensor... which could be in c.Data
	var bas *dense.Dense[DT]
	switch d := c.Data.(type) {
	case *dense.Dense[DT]:
		bas = d
	case []DT:
		bas = dense.New[DT](opts...)
	case nil:
		// if c.Data == nil and InitFn is not nil
		bas = dense.New[DT](opts...)
	default:
		panic("NYI")
	}

	if g != nil {
		return newValueInGraph[DT](g, name, bas)
	}
	return newValue[DT](name, bas)
}

// desc represents the common things that a Value and a Symbolic node have
type desc struct {
	id      NodeID
	name    string
	waiting int32 // atomic updates only
}

func (n desc) NodeID() NodeID { return n.id }

func (n desc) ID() int64 { return int64(n.id) }

func (n *desc) AddWaiting() { atomic.AddInt32(&n.waiting, 1) }

func (n *desc) Waiting() int {
	retVal := atomic.LoadInt32(&n.waiting)
	return int(retVal)
}

func (n *desc) ZeroWaiting() { atomic.StoreInt32(&n.waiting, int32(0)) }

func (n *desc) isnode() {}

// Value represents a node that has a value of a given datatype DT and a type T.
type Value[DT any, T tensor.Basic[DT]] struct {
	tensor.Basic[DT] // note this is the interface, not the constraint
	desc

	op         ops.Op[DT, T]
	beforeLift T
}

// newValue creates a new value node. It is the most basic way to create a node.
func newValue[DT any, T tensor.Tensor[DT, T]](name string, v T) *Value[DT, T] {
	retVal := &Value[DT, T]{
		Basic: v,
		desc: desc{
			name: name,
		},
	}
	return retVal
}

// newValueInGraph is a utility function to create a new *Value[DT,T] in a graph.
func newValueInGraph[DT any, T tensor.Tensor[DT, T]](g *Graph, name string, v T) *Value[DT, T] {
	// TODO check that v has g as engine
	retVal := &Value[DT, T]{
		Basic: v,
		desc: desc{
			name: name,
			id:   g.newNodeID(),
		},
	}
	if g != nil {
		g.AddNode(retVal)
	}
	return retVal
}

// replaceValueInGraph finds a node in the graph.
// If it's a *Value[DT,T] node, then the value in that node will be replaced with `v`.
// If it's a *Symbolic[DT] then that *Symbolic[DT] will be replaced with a new *Value[DT,T] with `v` as its value.
func replaceValueInGraph[DT any, T tensor.Basic[DT]](g *Graph, name string, id NodeID, v T) *Value[DT, T] {
	retVal := &Value[DT, T]{
		Basic: v,
		desc: desc{
			name: name,
			id:   id,
		},
	}
	if g != nil {
		g.replaceNode(retVal)
	}
	return retVal
}

// Name returns the name of a node that holds a Value.
func (n *Value[DT, T]) Name() string {
	if n == nil {
		return "<nil>"
	}
	return n.name
}

func (n *Value[DT, T]) Value() T {
	switch b := n.Basic.(type) {
	case T:
		return b
	case valuer[T]:
		return b.Value()
	default:
		panic("Cannot get Value")
	}
}

func (n *Value[DT, T]) Op() ops.Op[DT, T] { return n.op }

func (n *Value[DT, T]) O() ops.Desc { return n.op }

// Format implements fmt.Formatter.
func (n *Value[DT, T]) Format(f fmt.State, c rune) {
	if n == nil {
		fmt.Fprintf(f, "<nil>")
		return
	}
	switch c {
	case 's':
		fmt.Fprintf(f, "%s", n.name)
	default:
		if n.Basic == nil {
			fmt.Fprintf(f, "node(%v,%s,%v)", n.id, n.name, n.Basic)
			return
		}
		str := consFmtStr(f, c)
		fmt.Fprintf(f, str, n.Basic)
		// switch t := n.Basic.(type) {
		// case T:

		// default:

		// }
	}
}

func (n *Value[DT, T]) V() values.V {
	switch b := n.Basic.(type) {
	case dual.V:
		return b.V()
	default:
		return n.Basic
	}
}

func (n *Value[DT, T]) prelift() values.V { return n.beforeLift }

func (n *Value[DT, T]) setLifted(lifted, original values.V) {
	n.Basic = any(lifted).(tensor.Basic[DT])
	n.beforeLift = original.(T)
}

// d will return a value if n.Basic is a dual.V. Otherwise it returns nil
func (n *Value[DT, T]) d() dual.V {
	if dv, ok := n.Basic.(dual.V); ok {
		return dv
	}
	return nil
}

func (n *Value[DT, T]) DV() values.V {
	switch b := n.Basic.(type) {
	case dual.V:
		return b.DV()
	default:
		return nil
	}
}

// Symbolic represents a symbolic node. It needs a graph as an engine.
type Symbolic[DT any] struct {
	tensor.AP
	desc
	dt     dtype.Dtype
	engine *Graph
	Op     ops.Desc // tmp
}

// NewSymbolic creates a new symbolic node.
func NewSymbolic[DT any](g *Graph, shape shapes.Shape, name string) (*Symbolic[DT], error) {
	strides := tensor.CalcStrides(shape)
	ap := tensor.MakeAP(shape, strides, 0, 0)
	dt := dtype.Datatype[DT]{}
	retVal := &Symbolic[DT]{
		AP: ap,
		desc: desc{
			name: name,
		},
		dt:     dt,
		engine: g,
	}
	if g != nil {
		newid := g.newNodeID()
		retVal.id = newid
		if err := g.AddNode(retVal); err != nil {
			return retVal, err
		}
	}
	return retVal, nil
}

func (n *Symbolic[DT]) Name() string {
	if n == nil {
		return "<nil>"
	}
	return n.name
}

func (n *Symbolic[DT]) Dtype() dtype.Dtype { return n.dt }

func (n *Symbolic[DT]) IsNil() bool { return n == nil }

func (n *Symbolic[DT]) Info() *tensor.AP { return &n.AP }

// Type returns the type of the *header. This implements hm.Typer.
func (n *Symbolic[DT]) Type() hm.Type {
	if n.Shape().IsScalar() {
		return n.dt
	}
	return types.TensorType{Dims: n.Shape().Dims(), Of: n.dt}
}

func (n *Symbolic[DT]) Engine() tensor.Engine { return n.engine } // TODO maybe instantiate

func (n *Symbolic[DT]) Format(f fmt.State, c rune) {
	if n == nil {
		fmt.Fprintf(f, "<nil>")
		return
	}
	switch c {
	case 's':
		fmt.Fprintf(f, "%s", n.name)

	default:
		fmt.Fprintf(f, "node(%v,%s)",
			n.id, n.name)

	}
}

// weird bits that are required in the gorgonia.Tensor interface

// DataSize returns the size of the data in bytes.
func (n *Symbolic[DT]) DataSize() int { return n.Shape().TotalSize() }

// liftNode lifts a node if its engine is a lifter
func liftNode(n Node) Node {
	nx, ok := n.(valuelifter)
	if !ok {
		return n
	}
	v := nx.V()
	e := v.Engine().Workhorse()
	if l, ok := e.(Lifter); ok {
		lifted := l.Lift(v).(values.V)
		nx.setLifted(lifted, v)
	}
	return n
}

// SymToVal converts a symbolic node to a value node. It is a convenience function.
func SymToVal[DT any, T tensor.Basic[DT]](n *Symbolic[DT]) *Value[DT, T] {
	var d T
	switch any(d).(type) {
	case tensor.Aliker[T]:
		d = any(d).(tensor.Aliker[T]).Alike(tensor.WithShape(n.Shape()...), tensor.WithEngine(n.engine))
	case nil:
		// if it's nill it means that T is a interface that implements tensor.Basic[DT]... there's no underlying datatype.
		// we'll use dense as a default
		x := dense.New[DT](tensor.WithShape(n.Shape()...), tensor.WithEngine(n.engine))
		d = any(x).(T)
	default:
		panic(fmt.Sprintf("d of %T not handled yet", d))
	}

	retVal := replaceValueInGraph[DT, T](n.engine, n.name, n.id, d)
	retVal.op = n.Op.(ops.Op[DT, T])
	n.engine = nil
	n.Op = nil
	return retVal
}

/*
// Node is a tuple of a Tensor, ID, and name.
type Node[T any] struct {
	Tensor[DT]
	id   int64
	name string
	Op   ops.Op
	// beforeLift tracks the old value of a tensor before it has been lifted
	beforeLift Tensor

	waiting int32 // atomic updates only please
}

// Name returns the name of the node
func (n *Node[DT]) Name() string {
	if n == nil {
		return "<nil>"
	}
	return n.name
}

// NewNode in a given graph.
func NewNode[DT any](g *Graph, name string, opts ...tensor.ConsOpt) *Node[DT] {
	t := tensor.New(append(opts, tensor.WithEngine(g.Engine))...)
	n, err := Cons(g, name, t)
	if err != nil {
		panic(err)
	}
	return n
}

// NewSymbolic creates a new symbolic Tensor (*Node[DT] itself).
func NewSymbolic[DT any](g *Graph, name string, dt dtype.Dtype, shape shapes.Shape) (*Node[DT], error) {
	hdr := newHeader(g, dt, shape)
	if g != nil {
		return cons[DT](g, name, hdr)
	}
	return &Node[DT]{Tensor: hdr, name: name, id: 0, Op: nil}, nil
}

// Cons constructs a Node. It should be used very carefully.
// If the provided graph is nil, then Cons simply constructs the node by itself. No node will be added to the graph.
func Cons[DT any](g *Graph, name string, t tensor.Tensor) (*Node[DT], error) {
	if g != nil {
		return cons(g, name, t)
	}
	return &Node[DT]{Tensor: t, name: name, id: 0, Op: nil}, nil
}

// cons is Cons but with the graph guaranteed to not be nil
func cons[DT any](g *Graph, name string, t Tensor) (*Node[DT], error) {
	nm := g.NodeOf(t)
	if nm == nil {
		nm = g.newNode()
		nm.Tensor = t
		nm.name = name
		err := g.AddNode(nm)
		if err != nil {
			return nil, err
		}
		return nm, nil
	}
	if nm.name != name {
		return nil, errors.New("A node holding the tensor exists with a different name")
	}
	return nm, nil
}

// ID allows Node  to implement gonum.org/graph.Node
func (n *Node[DT]) ID() int64 {
	if n == nil {
		return -1
	}
	return n.id
}

// NodeID returns the ID as a NodeID of the node.
func (n *Node[DT]) NodeID() NodeID { return NodeID(n.ID()) }

// Format of the node
func (n Node) Format(f fmt.State, c rune) {
	switch c {
	case 's':
		fmt.Fprintf(f, "%s", n.name)
	default:
		switch t := n.Tensor.(type) {
		case tensor.Tensor:
			str := consFmtStr(f, c)
			fmt.Fprintf(f, str, t)
		default:
			fmt.Fprintf(f, "node(%v,%s,%v)",
				n.id, n.name, t)
		}
	}
}

// Value returns the value stored in a node. If the node represents a symbolic value, it will be converted into an actual value.
func (n *Node[DT]) Value() values.Value {
	switch t := n.Tensor.(type) {
	case values.Value:
		return t
	case *header:
		// we Lift n.Tensor into a tensor.Tensor
		var v tensor.Tensor
		v = tensor.New(tensor.Of(t.Dtype()), tensor.WithShape(t.Shape()...), tensor.WithEngine(t.g.Engine))
		if l, ok := v.Engine().(Lifter); ok {
			lifted := l.Lift(v)
			n.Tensor = lifted
			return lifted.(values.Value)
		}
		n.Tensor = v
		return v

	default:
		panic(fmt.Sprintf("Tensor of %T unhandled by node.Value()", n.Tensor))

	}
}

func (n *Node[DT]) AddWaiting() { atomic.AddInt32(&n.waiting, 1) }

func (n *Node[DT]) Waiting() int {
	retVal := atomic.LoadInt32(&n.waiting)
	return int(retVal)
}

func (n *Node[DT]) ZeroWaiting() { atomic.StoreInt32(&n.waiting, int32(0)) }
*/
