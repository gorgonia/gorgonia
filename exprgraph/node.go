package exprgraph

import (
	"sync/atomic"

	"github.com/chewxy/hm"
	"gonum.org/v1/gonum/graph"
	"gorgonia.org/dtype"
	"gorgonia.org/gorgonia/ops"
	"gorgonia.org/gorgonia/types"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/dense"
)

// constraints
var (
	_ Nodelike = Value[float64, *dense.Dense[float64]]{}
)

// Node is a description of a tensor that has a name and an ID.
//
// There are two possible definition of a Node:
//   - Value[DT, T]
//   - Symbolic[DT]
//
// It's an interface instead of a constraint because it's more useful as an interface.
type Node interface {
	graph.Node // ID() int64
	tensor.Desc
	Name() string
}

// desc represents the common things that a Value and a Symbolic node have
type desc struct {
	id      NodeID
	name    string
	waiting int32 // atomic updates only
}

func (n desc) ID() int64 { return int64(n.id) }

func (n *desc) AddWaiting() { atomic.AddInt32(&n.waiting, 1) }

func (n *desc) Waiting() int {
	retVal := atomic.LoadInt32(&n.waiting)
	return int(retVal)
}

func (n *desc) ZeroWaiting() { atomic.StoreInt32(&n.waiting, int32(0)) }

// Value represents a node that has a value.
type Value[DT any, T tensor.Tensor[DT, T]] struct {
	tensor.Basic[DT] // note this is the interface, not the constraint
	desc

	Op         ops.Op[DT, T]
	beforeLift tensor.Basic[DT]
}

// Name returns the name of a node that holds a Value.
func (n *Value[DT, T]) Name() string {
	if n == nil {
		return "<nil>"
	}
	return n.name
}

func (n *Value[DT, T]) Value() values.V { return n.Basic }

func (n *Value[DT, T]) prelift() values.V { return n.beforeLift }

func (n *Value[DT, T]) setLifted(lifted, original values.V) {
	n.Basic = lifted.(tensor.Basic[DT])
	n.beforeLift = original.(tensor.Basic[DT])
}

// Symbolic represents a symbolic node. It needs a graph as an engine.
type Symbolic[DT any] struct {
	tensor.AP
	desc
	dt     dtype.Dtype
	engine *Graph
	Op     any // tmp
}

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
	if err := g.AddNode(retVal); err != nil {
		return nil, err
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

func (n *Symbolic[DT]) Info() *tensor.AP { return &n.AP }

// Type returns the type of the *header. This implements hm.Typer.
func (n *Symbolic[DT]) Type() hm.Type {
	if n.Shape().IsScalar() {
		return n.dt
	}
	return types.TensorType{Dims: n.Shape().Dims(), Of: n.dt}
}

func (n *Symbolic[DT]) Engine() tensor.Engine { return n.engine } // TODO maybe instantiate

// liftNode lifts a node if its engine is a lifter
func liftNode(n Node) Node {
	nx, ok := n.(valuelifter)
	if !ok {
		return n
	}
	v := nx.Value()
	e := v.Engine()
	if l, ok := e.(Lifter); ok {
		lifted := l.Lift(v).(values.V)
		nx.setLifted(lifted, v)
	}
	return n
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
