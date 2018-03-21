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
	prev NodeSet
	l    Nodes
}

func walkLOT(p *program) []Nodes {
	s := new(lot)
	s.data = make([]Nodes, 1, 8)
	s.seen = make(NodeSet)
	s.prev = make(NodeSet)

	s.l = p.sorted

	l0 := make(Nodes, len(p.g.leaves)+len(p.g.constants))
	copy(l0, p.g.leaves)
	copy(l0[len(p.g.leaves):], p.g.constants)
	s.data[0] = l0

	for _, n := range l0 {
		s.seen.Add(n)
		s.prev.Add(n)
	}
	i := 1
	for {
		if s.do(i) {
			break
		}
		i++
	}
	return s.data
}

func (s *lot) do(level int) bool {
	// log.Printf("level %d", level)
	if len(s.data) <= level {
		s.data = append(s.data, make(Nodes, 0, 4))
	}
	for _, n := range s.l {
		if s.seen.Contains(n) {
			continue
		}
		var seenCount int
		for _, child := range n.children {
			if s.prev.Contains(child) {
				seenCount++
			}
		}
		if seenCount == len(n.children) {
			s.data[level] = append(s.data[level], n)
			s.seen.Add(n)
		}
	}
	for _, n := range s.data[level] {
		s.prev.Add(n)
	}
	return len(s.seen) == len(s.l)
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
