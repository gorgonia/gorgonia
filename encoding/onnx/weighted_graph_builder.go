package gorgonnx

import (
	"math"

	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/simple"
)

// NewGraph ...
func newGraph() *graph {
	return &graph{
		g: simple.NewWeightedDirectedGraph(math.MaxFloat64, -1),
	}
}

// SetWeightedEdge adds an edge from one node to
// another. If the graph supports node addition
// the nodes will be added if they do not exist,
// otherwise SetWeightedEdge will panic.
// The behavior of a WeightedEdgeAdder when the IDs
// returned by e.From() and e.To() are equal is
// implementation-dependent.
// Whether e, e.From() and e.To() are stored.
// Fulfills the graph.WeightedEdgeAdder interface
func (g *graph) SetWeightedEdge(e graph.WeightedEdge) {
	g.g.SetWeightedEdge(e)
}

// NewWeightedEdge returns a new WeightedEdge from the source to the destination node.
// Fulfills the graph.WeightedEdgeAdder interface
func (g *graph) NewWeightedEdge(from, to graph.Node, w float64) graph.WeightedEdge {
	return g.g.NewWeightedEdge(from, to, w)

}

// AddNode ...
func (g *graph) AddNode(n graph.Node) {
	g.g.AddNode(n)

}

// NewNode ...
func (g *graph) NewNode() graph.Node {
	n := g.g.NewNode()
	return &Node{
		id: n.ID(),
	}
}

// Node ...
func (g *graph) Node(id int64) graph.Node {
	return g.g.Node(id)
}

// Nodes ...
func (g *graph) Nodes() graph.Nodes {
	return g.g.Nodes()
}

// From ...
func (g *graph) From(id int64) graph.Nodes {
	return g.g.From(id)
}

// HasEdgeBetween ...
func (g *graph) HasEdgeBetween(xid, yid int64) bool {
	return g.g.HasEdgeBetween(xid, yid)
}

// Edge ...
func (g *graph) Edge(uid, vid int64) graph.Edge {
	return g.g.Edge(uid, vid)
}

// HasEdgeFromTo ...
func (g *graph) HasEdgeFromTo(uid, vid int64) bool {
	return g.g.HasEdgeFromTo(uid, vid)
}

// To ...
func (g *graph) To(id int64) graph.Nodes {
	return g.g.To(id)
}
