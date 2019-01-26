package onnx

import (
	"sort"

	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/iterator"
)

type byWeight []graph.WeightedEdge

func (c byWeight) Len() int { return len(c) }
func (c byWeight) Less(i, j int) bool {
	return c[i].Weight() < c[j].Weight()
}
func (c byWeight) Swap(i, j int) { c[i], c[j] = c[j], c[i] }

// getOrderedChildren returns an iterator of children nodes ordered by the weighted edges
func getOrderedChildren(g graph.WeightedDirected, n graph.Node) *iterator.OrderedNodes {
	// Get all the edges that reach the node n
	children := g.From(n.ID())
	// Now get the edges
	if children.Len() == 0 {
		return nil
	}
	edges := make([]graph.WeightedEdge, children.Len())
	for i := 0; children.Next(); i++ {
		edges[i] = g.WeightedEdge(n.ID(), children.Node().ID())
	}
	sort.Sort(byWeight(edges))

	children.Reset()
	orderWeightedEdges := iterator.NewOrderedWeightedEdges(edges)
	nodes := make([]graph.Node, children.Len())
	for i := 0; orderWeightedEdges.Next(); i++ {
		nodes[i] = orderWeightedEdges.WeightedEdge().To()
	}
	return iterator.NewOrderedNodes(nodes)
}
