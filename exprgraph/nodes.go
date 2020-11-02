package exprgraph

import (
	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/iterator"
)

// Nodes is an iterator over Nodes
type Nodes struct {
	*iterator.Nodes
	*iterator.NodesByEdge
}

// Len returns the remaining number of nodes to be iterated over.
func (n *Nodes) Len() int {
	if n.NodesByEdge != nil {
		return n.NodesByEdge.Len()
	}
	return n.Nodes.Len()
}

// Next returns whether the next call of Node will return a valid node.
func (n *Nodes) Next() bool {
	if n.NodesByEdge != nil {
		return n.NodesByEdge.Next()
	}
	return n.Nodes.Next()
}

// Node returns the current node of the iterator. Next must have been
// called prior to a call to Node.
func (n *Nodes) Node() graph.Node {
	if n.NodesByEdge != nil {
		return n.NodesByEdge.Node()
	}
	return n.Nodes.Node()
}

// Reset returns the iterator to its initial state.
func (n *Nodes) Reset() {
	if n.NodesByEdge != nil {
		n.NodesByEdge.Reset()
		return
	}
	n.Nodes.Reset()
}

// NodeSlice returns all the remaining nodes in the iterator and advances
// the iterator. The order of nodes within the returned slice is not
// specified.
func (n *Nodes) NodeSlice() []*Node {
	if n.Len() == 0 {
		return nil
	}
	nodes := make([]*Node, 0, n.Len())
	var ns []graph.Node
	if n.NodesByEdge != nil {
		ns = n.NodesByEdge.NodeSlice()
	} else {
		ns = n.Nodes.NodeSlice()
	}
	for i, n := range ns {
		nodes[i] = n.(*Node)
	}
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
	ns := make(map[int64]graph.Node, len(nodes))
	for i, n := range nodes {
		ns[i] = graph.Node(n)
	}
	if edges == nil {
		return &Nodes{
			Nodes: iterator.NewNodes(ns),
		}
	}
	return &Nodes{
		NodesByEdge: iterator.NewNodesByWeightedEdge(ns, edges),
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
