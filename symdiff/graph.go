package symdiff

import "gorgonia.org/gorgonia/exprgraph"

// Graph is a data structure that holds the expression graph and the maps of partial derivatives for each node.
type Graph struct {
	*exprgraph.Graph

	deriv   map[exprgraph.NodeID]exprgraph.NodeID  // key: node. value: derivative of node.
	derivOf map[exprgraph.NodeID]exprgraph.NodeIDs // key: node. value: node is a derivative of these nodes.
}

func (g *Graph) Deriv(a exprgraph.Nodelike) *exprgraph.Node {
	d := g.deriv[exprgraph.NodeID(a.ID())]
	return g.Graph.Get(d)
}

func (g *Graph) DerivOf(a exprgraph.Nodelike) []*exprgraph.Node {
	ds := g.derivOf[exprgraph.NodeID(a.ID())]
	return exprgraph.NodesFromNodeIDs(g.Graph, ds)
}
