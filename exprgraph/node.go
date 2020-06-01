package exprgraph

import "gorgonia.org/tensor"

// NodeID represents a Node's ID
type NodeID int64

// ID returns the ID as an int64. This is used to fulfil gonum.org/gonum/graph.Node interface.
func (n NodeID) ID() int64 { return int64(n) }

// Node is a tuple of a Tensor, ID, and name.
type Node struct {
	tensor.Tensor
	NodeID
	name string
}

func Make(g *Graph, name string, opts ...tensor.ConsOpt) Node {
	consOpts := append([]tensor.ConsOpt{tensor.WithEngine(g.StdEng), inGraph(), WithName(name)}, opts...)
	t := tensor.New(consOpts...)
	return g.nodeOf(t)
}

// OK returns true if the Node is good for processing.
func (n *Node) OK() bool { return n.Tensor != nil }

// Node implements gorgonia.Result

func (n *Node) Node() Node   { return *n }
func (n *Node) Nodes() Nodes { return Nodes{n} }
func (n *Node) Err() error   { return nil }

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
	children []*node
	flag
	device
	Op
}

/* TODO */

type Op interface{}

type device int16 // only reason why it's int16 is so that we fill up the struct
