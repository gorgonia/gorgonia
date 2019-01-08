package graph

import (
	"strconv"

	"gonum.org/v1/gonum/graph"
	"gorgonia.org/gorgonia/internal/execution"
)

// SetWeightedEdge adds a weighted edge from one node to another.
// If the nodes do not exist, they are added and are set to the nodes of the edge otherwise.
// It will panic if the IDs of the e.From and e.To are equal.
func (g *ExprGraph) SetWeightedEdge(e graph.WeightedEdge) {
	g.w.SetWeightedEdge(e)

}

// NewWeightedEdge returns a new weighted edge from the source to the destination node.
func (g *ExprGraph) NewWeightedEdge(from, to graph.Node, w float64) graph.WeightedEdge {
	return g.w.NewWeightedEdge(from, to, w)

}

// AddNode adds n to the graph. It panics if the added node ID matches an existing node ID.
func (g *ExprGraph) AddNode(n graph.Node) {
	g.w.AddNode(n)

}

// NewNode returns a new unique Node to be added to g. The Node's ID does not become valid in g until the Node is added to g.
func (g *ExprGraph) NewNode() graph.Node {
	n := borrowNode()
	n.dataOn = execution.CPU
	n.id = g.w.NewNode().ID()
	n.name = strconv.FormatInt(n.id, 10)
	n.g = g
	n.fix()
	return n
}

// Node returns the node with the given ID if it exists in the graph, and nil otherwise.
func (g *ExprGraph) Node(id int64) graph.Node {
	return g.w.Node(id)
}

// Nodes returns all the nodes in the graph.
func (g *ExprGraph) Nodes() graph.Nodes {
	return g.w.Nodes()
}

// From returns all nodes in g that can be reached directly from n.
func (g *ExprGraph) From(id int64) graph.Nodes {
	return g.w.From(id)
}

// HasEdgeBetween returns whether an edge exists between nodes x and y without considering direction.
func (g *ExprGraph) HasEdgeBetween(xid, yid int64) bool {
	return g.w.HasEdgeBetween(xid, yid)
}

// Edge returns the edge from u to v if such an edge exists and nil otherwise.
// The node v must be directly reachable from u as defined by the From method.
func (g *ExprGraph) Edge(uid, vid int64) graph.Edge {
	return g.w.Edge(uid, vid)
}

// HasEdgeFromTo returns whether an edge exists in the graph from u to v.
func (g *ExprGraph) HasEdgeFromTo(uid, vid int64) bool {
	return g.w.HasEdgeFromTo(uid, vid)
}

// To returns all nodes in g that can reach directly to n.
func (g *ExprGraph) To(id int64) graph.Nodes {
	return g.w.To(id)
}

// Weight returns the weight for the edge between x and y if Edge(x, y) returns a non-nil Edge.
// If x and y are the same node or there is no joining edge between the two nodes
// the weight value returned is either the graph's absent or self value.
// Weight returns true if an edge exists between x and y or if x and y have the same ID, false otherwise.
func (g *ExprGraph) Weight(xid, yid int64) (w float64, ok bool) {
	return g.w.Weight(xid, yid)
}

// WeightedEdge returns the weighted edge from u to v if such an edge exists and nil otherwise.
// The node v must be directly reachable from u as defined by the From method.
func (g *ExprGraph) WeightedEdge(uid, vid int64) graph.WeightedEdge {
	return g.w.WeightedEdge(uid, vid)
}
