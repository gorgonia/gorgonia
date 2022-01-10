package symdiff

import (
	"gorgonia.org/gorgonia/exprgraph"
	"gorgonia.org/gorgonia/ops"
)

// Op represents an Op that can be symbolically differentiated
type Op interface {
	ops.Op

	DiffWRT(inputs int) []bool

	SymDiff(inputs []*exprgraph.Node, output, grad *exprgraph.Node) (retVal []*exprgraph.Node, err error)
}
