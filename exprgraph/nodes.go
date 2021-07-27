package exprgraph

import (
	"gonum.org/v1/gonum/graph"
)

// Nodes is an iterator over Nodes
type Nodes struct {
	ns []*Node
	i  int
}

// Len returns the remaining number of nodes to be iterated over.
func (n *Nodes) Len() int { return len(n.ns) }

// Next returns whether the next call of Node will return a valid node.
func (n *Nodes) Next() bool { n.i++; return n.i < len(n.ns) }

// Node returns the current node of the iterator. Next must have been
// called prior to a call to Node.
func (n *Nodes) Node() graph.Node {
	if n.i < 0 || n.i >= len(n.ns) {
		return nil
	}
	return n.ns[n.i]
}

// Reset returns the iterator to its initial state.
func (n *Nodes) Reset() { n.i = -1 }

// NodeSlice returns all the remaining nodes in the iterator and advances
// the iterator. The order of nodes within the returned slice is not
// specified.
func (n *Nodes) NodeSlice() []*Node {
	if n.i < 0 {
		n.i = 0
	}
	retVal := n.ns[n.i:len(n.ns)]
	n.i = len(n.ns)
	return retVal
}

// NodesFromOrdered returns a Nodes initialized with the
// provided nodes, a map of node IDs to graph.Nodes, and the set
// of edges, a map of to-node IDs to graph.WeightedEdge, that can be
// traversed to reach the nodes that the NodesByEdge will iterate
// over.
func NodesFromOrdered(ns []*Node) *Nodes { return &Nodes{ns: ns, i: -1} }

type nodeIDs []NodeID

func (ns nodeIDs) Contains(a NodeID) bool {
	for _, n := range ns {
		if n == a {
			return true
		}
	}
	return false
}
