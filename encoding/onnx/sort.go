package gorgonnx

import (
	"sort"

	"gonum.org/v1/gonum/graph"
)

// getOrderedChildren returns the children nodes of the current node
func getOrderedChildren(g graph.WeightedDirected, n *Node) []*Node {
	// Get all the edges that reach the node n
	children := g.From(n.ID())
	// Now get the edges
	edges := make([]graph.WeightedEdge, children.Len())
	for i := 0; children.Next(); i++ {
		edges[i] = g.WeightedEdge(n.ID(), children.Node().ID())
	}
	sort.Sort(byWeight(edges))
	nodes := make([]*Node, len(edges))
	for i := 0; i < len(edges); i++ {
		nodes[i] = edges[i].To().(*Node)
	}

	return nodes
}

type byWeight []graph.WeightedEdge

func (w byWeight) Len() int           { return len(w) }
func (w byWeight) Swap(i, j int)      { w[i], w[j] = w[j], w[i] }
func (w byWeight) Less(i, j int) bool { return w[i].Weight() < w[j].Weight() }
