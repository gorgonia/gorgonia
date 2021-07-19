package exprgraph

import (
	"errors"
	"fmt"

	"gorgonia.org/dtype"
	"gorgonia.org/gorgonia/ops"
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
		nm := g.NodeOf(hdr)
		if nm == nil {
			nm = g.NewNode()
			nm.Tensor = hdr
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
	return &Node{Tensor: hdr, name: name, id: 0, Op: nil}, nil
}

// Cons constructs a Node. It should be used very carefully.
// If the provided graph is nil, then Cons simply constructs the node by itself. No node will be added to the graph.
func Cons(g *Graph, name string, t tensor.Tensor) (*Node, error) {
	if g != nil {
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
	return &Node{Tensor: t, name: name, id: 0, Op: nil}, nil
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
