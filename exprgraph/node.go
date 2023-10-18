package exprgraph

import (
	"errors"
	"fmt"
	"sync/atomic"

	"gorgonia.org/dtype"
	"gorgonia.org/gorgonia/ops"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
)

// constraints
var (
	_ Tensor   = &Node{}
	_ Nodelike = &Node{}
)

// Node is a tuple of a Tensor, ID, and name.
type Node[DT any] struct {
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
func NewNode(g *Graph, name string, opts ...tensor.ConsOpt) *Node[DT] {
	t := tensor.New(append(opts, tensor.WithEngine(g.Engine))...)
	n, err := Cons(g, name, t)
	if err != nil {
		panic(err)
	}
	return n
}

// NewSymbolic creates a new symbolic Tensor (*Node[DT] itself).
func NewSymbolic(g *Graph, name string, dt dtype.Dtype, shape shapes.Shape) (*Node[DT], error) {
	hdr := newHeader(g, dt, shape)
	if g != nil {
		return cons(g, name, hdr)
	}
	return &Node{Tensor: hdr, name: name, id: 0, Op: nil}, nil
}

// Cons constructs a Node. It should be used very carefully.
// If the provided graph is nil, then Cons simply constructs the node by itself. No node will be added to the graph.
func Cons(g *Graph, name string, t tensor.Tensor) (*Node[DT], error) {
	if g != nil {
		return cons(g, name, t)
	}
	return &Node{Tensor: t, name: name, id: 0, Op: nil}, nil
}

// cons is Cons but with the graph guaranteed to not be nil
func cons(g *Graph, name string, t Tensor) (*Node[DT], error) {
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
