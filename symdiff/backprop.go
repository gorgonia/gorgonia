package symdiff

import (
	"gorgoniaa.org/gorgonia/exprgraph"
)

// Backporopagate computes the symbolic differentiation of the outputs with regards to the inputs.
func Backporopagate(g *exprgraph.Graph, outputs, gradOutputs, wrt exprgraph.Nodes) (retVal exprgraph.Nodes, err error) {
	panic("NYI")
}
