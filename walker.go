package gorgonia

import (
	"sort"

	"github.com/pkg/errors"
	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/iterator"
	"gonum.org/v1/gonum/graph/topo"
)

// WalkGraph walks a graph. It returns a channel of *Nodes, so be sure to consume the channel or there may be a deadlock
func WalkGraph(start *Node) <-chan *Node {
	ch := make(chan *Node)
	walked := NewNodeSet()

	go func() {
		walkGraph(start, ch, walked)
		close(ch)
	}()
	return ch
}

func walkGraph(start *Node, ch chan *Node, walked NodeSet) {
	defer func() {
		walked.Add(start)
	}()
	if _, ok := walked[start]; ok {
		return // walked before
	}

	ch <- start

	for _, child := range start.children {
		walkGraph(child, ch, walked)
	}

}

// Sort topologically sorts a ExprGraph: root of graph will be first
func Sort(g *ExprGraph) (*iterator.OrderedNodes, error) {
	// if sortedNodes, err = topo.Sort(g); err != nil {
	sortedNodes, err := topo.SortStabilized(g, reverseLexical)
	if err != nil {
		return nil, errors.Wrap(err, sortFail)
	}

	return iterator.NewOrderedNodes(sortedNodes), nil
}

// UnstableSort performs a topological sort of the directed graph g returning the 'from' to 'to' sort order. If a topological ordering is not possible, an Unorderable error is returned listing cyclic components in g with each cyclic component's members sorted by ID. When an Unorderable error is returned, each cyclic component's topological position within the sorted nodes is marked with a nil graph.Node.
func UnstableSort(g *ExprGraph) (*iterator.OrderedNodes, error) {
	sortedNodes, err := topo.Sort(g)
	if err != nil {
		return nil, errors.Wrap(err, sortFail)
	}

	return iterator.NewOrderedNodes(sortedNodes), nil
}

func reverseNodes(sorted Nodes) {
	for i := len(sorted)/2 - 1; i >= 0; i-- {
		j := len(sorted) - i - 1
		sorted[i], sorted[j] = sorted[j], sorted[i]
	}
}

type byID []graph.Node

func (ns byID) Len() int           { return len(ns) }
func (ns byID) Less(i, j int) bool { return ns[i].ID() > ns[j].ID() }
func (ns byID) Swap(i, j int)      { ns[i], ns[j] = ns[j], ns[i] }

func reverseLexical(a []graph.Node) {
	sort.Sort(byID(a))
}

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

type byWeight []graph.WeightedEdge

func (c byWeight) Len() int { return len(c) }
func (c byWeight) Less(i, j int) bool {
	return c[i].Weight() < c[j].Weight()
}
func (c byWeight) Swap(i, j int) { c[i], c[j] = c[j], c[i] }
