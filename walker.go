package gorgonia

import (
	"sort"

	"github.com/pkg/errors"
	"gonum.org/v1/gonum/graph"
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

// los stores the state of level order traversal
type lot struct {
	data []Nodes
	seen NodeSet
}

func walkLOT(root *Node) []Nodes {
	s := new(lot)
	s.data = make([]Nodes, len(root.children))
	s.seen = make(NodeSet)
	s.dfs(root, 0)

	// reverse
	for i, j := 0, len(s.data)-1; i < j; i, j = i+1, j-1 {
		s.data[i], s.data[j] = s.data[j], s.data[i]
	}
	// set
	for i, grp := range s.data {
		s.data[i] = grp.Set()
	}

	return s.data
}

func (s *lot) dfs(n *Node, level int) {
	if n == nil {
		return
	}

	s.data[level] = append(s.data[level], n)
	if len(s.data) <= level+1 {
		s.data = append(s.data, make(Nodes, 0, len(n.children)))
	}

	for _, child := range n.children {
		s.dfs(child, level+1)
	}
}

// Sort topologically sorts a ExprGraph: root of graph will be first
func Sort(g *ExprGraph) (sorted Nodes, err error) {
	var sortedNodes []graph.Node
	// if sortedNodes, err = topo.Sort(g); err != nil {
	if sortedNodes, err = topo.SortStabilized(g, reverseLexical); err != nil {
		return nil, errors.Wrap(err, sortFail)
	}

	sorted = graphNodeToNode(sortedNodes)
	return
}

func UnstableSort(g *ExprGraph) (sorted Nodes, err error) {
	var sortedNodes []graph.Node
	if sortedNodes, err = topo.Sort(g); err != nil {
		return nil, errors.Wrap(err, sortFail)
	}

	sorted = graphNodeToNode(sortedNodes)
	return
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
