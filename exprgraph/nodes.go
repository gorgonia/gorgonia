package exprgraph

import "gonum.org/v1/gonum/graph"

// iterator implements graph.Nodes
//
// e.g:
//	for c := it.Next(); c{
//		n := it.Node()
//		... // use n
//	}
type iterator struct {
	cur   int
	nodes Nodes
}

// iterator implements graph.Nodes

func (it *iterator) Next() bool       { it.cur++; return it.cur < len(it.nodes) }
func (it *iterator) Len() int         { return len(it.nodes) }
func (it *iterator) Reset()           { it.cur = 0 }
func (it *iterator) Node() graph.Node { return it.nodes[it.cur] }

// Nodes returns the raw slice of nodes
func (it *iterator) Nodes() Nodes { return it.nodes }

type Nodes []Node
