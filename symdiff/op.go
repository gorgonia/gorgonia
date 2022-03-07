package symdiff

import (
	"gorgonia.org/gorgonia/exprgraph"
	"gorgonia.org/gorgonia/ops"
)

// Op represents an Op that can be symbolically differentiated
type Op interface {
	ops.Op

	DiffWRT(inputs int) []bool

	SymDiff(g *exprgraph.Graph, inputs []*exprgraph.Node, output, grad *exprgraph.Node) (retVal []*exprgraph.Node, err error)
}

func isStmt(op Op) bool { _, ok := op.(ops.Statement); return ok }

func isInput(op ops.Op) bool { return op == nil }
