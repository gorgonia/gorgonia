package stdops

import (
	"fmt"

	"gorgonia.org/gorgonia/exprgraph"
)

// SymDiff performs the symbolic differentiation of add.
func (op addOp) SymDiff(g *exprgraph.Graph, inputs []*exprgraph.Node, output *exprgraph.Node, grad *exprgraph.Node) (retVal []*exprgraph.Node, err error) {
	return []*exprgraph.Node{grad, grad}, nil
}

// SymDiff performs the symbolic differentiation of sub.
func (op subOp) SymDiff(g *exprgraph.Graph, inputs []*exprgraph.Node, output *exprgraph.Node, grad *exprgraph.Node) (retVal []*exprgraph.Node, err error) {
	neg := negOp{}
	y := inputs[1]
	dzdy, err := g.Apply(neg, fmt.Sprintf("∂%v", y.Name()), grad)
	if err != nil {
		return nil, err
	}
	return []*exprgraph.Node{grad, dzdy}, nil
}
