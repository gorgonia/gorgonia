package exprgraph

import (
	"gonum.org/v1/gonum/graph"
)

// Nodes is an iterator over Nodes
type Nodes struct {
	nodes map[int64]*Node
	edges int
	iter  *mapIter
	pos   int
	curr  *Node
}

// Len returns the remaining number of nodes to be iterated over.
func (n *Nodes) Len() int {
	return n.edges - n.pos
}

// Next returns whether the next call of Node will return a valid node.
func (n *Nodes) Next() bool {
	if n.pos >= n.edges {
		return false
	}
	ok := n.iter.next()
	if ok {
		n.pos++
		n.curr = n.nodes[n.iter.id()]
	}
	return ok
}

// Node returns the current node of the iterator. Next must have been
// called prior to a call to Node.
func (n *Nodes) Node() graph.Node {
	return n.curr
}

// Reset returns the iterator to its initial state.
func (n *Nodes) Reset() {
	n.curr = nil
	n.pos = 0
	n.iter.it = nil
}

// NodeSlice returns all the remaining nodes in the iterator and advances
// the iterator. The order of nodes within the returned slice is not
// specified.
func (n *Nodes) NodeSlice() []*Node {
	if n.Len() == 0 {
		return nil
	}
	nodes := make([]*Node, 0, n.Len())
	for n.iter.next() {
		nodes = append(nodes, n.nodes[n.iter.id()])
	}
	n.pos = n.edges
	return nodes
}

// NewNodes returns a Nodes initialized with the
// provided nodes, a map of node IDs to graph.Nodes, and the set
// of edges, a map of to-node IDs to graph.WeightedEdge, that can be
// traversed to reach the nodes that the NodesByEdge will iterate
// over. No check is made that the keys match the graph.Node IDs,
// and the map keys are not used.
// If edges is nil, an iterator over the random nodes is provided
//
// Behavior of the NodesByEdge is unspecified if nodes or edges
// is mutated after the call the NewNodes.
func NewNodes(nodes map[int64]*Node, edges map[int64]graph.WeightedEdge) *Nodes {
	if edges == nil {
		return &Nodes{nodes: nodes, iter: newMapIterNodes(nodes)}
	}
	return &Nodes{
		nodes: nodes,
		edges: len(edges),
		iter:  newMapIterWeightedEdges(edges),
	}
}

type nodeIDs []NodeID

func (ns nodeIDs) Contains(a NodeID) bool {
	for _, n := range ns {
		if n == a {
			return true
		}
	}
	return false
}
