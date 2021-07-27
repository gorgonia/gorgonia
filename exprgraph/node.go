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
	_ Tensor = &Node{}
)

// NodeID represents a Node's ID
type NodeID int64

// ID returns the ID as an int64. This is used to fulfil gonum.org/gonum/graph.Node interface.
func (n NodeID) ID() int64 { return int64(n) }

// Node is a tuple of a Tensor, ID, and name.
type Node struct {
	Tensor
	id   int64
	name string
	Op   ops.Op
	// beforeLift tracks the old value of a tensor before it has been lifted
	beforeLift Tensor

	waiting int32 // atomic updates only please
}

// GetName returns the name of the node
func (n *Node) GetName() string {
	return n.name
}

// NewNode in a given graph.
func NewNode(g *Graph, name string, opts ...tensor.ConsOpt) *Node {
	t := tensor.New(append(opts, tensor.WithEngine(g.Engine))...)
	n, err := Cons(g, name, t)
	if err != nil {
		panic(err)
	}
	return n
}

// NewSymbolic creates a new symbolic Tensor (*Node itself).
func NewSymbolic(g *Graph, name string, dt dtype.Dtype, shape shapes.Shape) (*Node, error) {
	hdr := newHeader(g, dt, shape)
	if g != nil {
		return cons(g, name, hdr)
	}
	return &Node{Tensor: hdr, name: name, id: 0, Op: nil}, nil
}

// Cons constructs a Node. It should be used very carefully.
// If the provided graph is nil, then Cons simply constructs the node by itself. No node will be added to the graph.
func Cons(g *Graph, name string, t tensor.Tensor) (*Node, error) {
	if g != nil {
		return cons(g, name, t)
	}
	return &Node{Tensor: t, name: name, id: 0, Op: nil}, nil
}

// cons is Cons but with the graph guaranteed to not be nil
func cons(g *Graph, name string, t Tensor) (*Node, error) {
	nm := g.NodeOf(t)
	if nm == nil {
		nm = g.NewNode()
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
func (n *Node) ID() int64 { return n.id }

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
func (n *Node) Value() values.Value {
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

func (n *Node) AddWaiting() { atomic.AddInt32(&n.waiting, 1) }

func (n *Node) Waiting() int {
	retVal := atomic.LoadInt32(&n.waiting)
	return int(retVal)
}

func (n *Node) ZeroWaiting() { atomic.StoreInt32(&n.waiting, int32(0)) }
