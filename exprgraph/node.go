package exprgraph

import (
	"errors"
	"fmt"
	"log"

	"gorgonia.org/gorgonia"
	"gorgonia.org/gorgonia/ops"
	"gorgonia.org/tensor"
)

// constraints
var (
	_ gorgonia.Tensor = &Node{}
)

// NodeID represents a Node's ID
type NodeID int64

// ID returns the ID as an int64. This is used to fulfil gonum.org/gonum/graph.Node interface.
func (n NodeID) ID() int64 { return int64(n) }

// Node is a tuple of a Tensor, ID, and name.
type Node struct {
	gorgonia.Tensor
	id   int64
	name string
	Op   ops.Op
}

// GetName returns the name of the node
func (n *Node) GetName() string {
	return n.name
}

// NewNode in a given graph.
func NewNode(g *Graph, name string, opts ...tensor.ConsOpt) *Node {
	t := tensor.New(opts...)
	n, err := Cons(g, name, t)
	if err != nil {
		panic(err)
	}
	return n
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
			log.Printf("tensor type %T unsupported for node.Format", n.Tensor)
		}
	}
}

/*
// GraphNode is a tuple of a graph object and a node. This allows for querying the payload of the Node.
//
// This is the object that should be used for any kind of query (topsort, etc)
type GraphNode struct {
	*Graph
	Node
}

//go:notinheap
type gn struct {
	*Graph
	Node
}

// node is a node for internal use. Its graph is defined by the links (i.e. pointers).
// if the ID is negative, it means that the node is in-progress
type node struct {
	Node
	flag
	execution.Device
}

*/
