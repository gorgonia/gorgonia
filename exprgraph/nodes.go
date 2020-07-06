package exprgraph

import (
	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/iterator"
)

type Nodes []Node

type nodeIDs []NodeID

func (ns nodeIDs) Contains(a NodeID) bool {
	for _, n := range ns {
		if n == a {
			return true
		}
	}
	return false
}

/* various other interop */

func nodeIDsToGraphNodes(ids []NodeID) graph.Nodes {
	retVal := make([]graph.Node, 0, len(ids))
	for i := range ids {
		retVal = append(retVal, ids[i])
	}
	return iterator.NewOrderedNodes(retVal)
}
