package gorgonia

import (
	"github.com/gonum/graph"
	"github.com/gonum/graph/topo"
	"github.com/pkg/errors"
)

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

func Sort(g *ExprGraph) (sorted Nodes, err error) {
	var sortedNodes []graph.Node
	if sortedNodes, err = topo.Sort(g); err != nil {
		return nil, errors.Wrap(err, sortFail)
	}

	sorted = graphNodeToNode(sortedNodes)
	return
}
