package gorgonia

import "gonum.org/v1/gonum/graph"

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
	n.dataOn = CPU
	n.id = g.w.NewNode().ID()
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
