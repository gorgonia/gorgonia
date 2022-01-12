package symdiff

import "gorgonia.org/gorgonia/exprgraph"

// Error provides the context at which a symbolic differentiation error has occured.
type Error struct {
	g         *exprgraph.Graph
	single    exprgraph.NodeID
	nodes     exprgraph.NodeIDs
	grad      exprgraph.NodeID
	nodeGrads map[exprgraph.NodeID]exprgraph.NodeIDs
	err       error
}

// Error implements error.
func (err Error) Error() string { return err.err.Error() }

// NodeIDs return the NodeIDs that caused the error.
func (err Error) NodeIDs() exprgraph.NodeIDs { return err.nodes }

// Nodes returns the nodes involved in the error
func (err Error) Nodes() []*exprgraph.Node { return exprgraph.NodesFromNodeIDs(err.g, err.nodes) }

// Node returns a specific node involved in the error.
func (err Error) Node() *exprgraph.Node { return err.g.Get(err.single) }

// Grads returns the grads involved in the error.
func (err Error) GradMap() map[exprgraph.NodeID]exprgraph.NodeIDs { return err.nodeGrads }

// Grad returns a specific grad involved in the error.
func (err Error) Grad() *exprgraph.Node { return err.g.Get(err.grad) }
