package gorgonia

import (
	"gorgonia.org/gorgonia/node"
)

// Apply ...
func (g *ExprGraph) Apply(operation Operation, n node.Node) error {
	op, err := operation(g, n)
	if err != nil {
		return err
	}
	g.ApplyOp(op.(Op), n.(*Node))

	return nil
}
