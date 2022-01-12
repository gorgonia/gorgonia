package symdiff

import "gorgonia.org/gorgonia/exprgraph"

type Graph struct {
	*exprgraph.Graph

	deriv   map[exprgraph.NodeID]exprgraph.NodeID  // key: node. value: derivative of node.
	derivOf map[exprgraph.NodeID]exprgraph.NodeIDs // key: node. value: node is a derivative of these nodes.
}
