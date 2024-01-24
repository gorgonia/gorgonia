package stdops

import (
	"fmt"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia/exprgraph"
	"gorgonia.org/gorgonia/internal/encoding"
	"gorgonia.org/gorgonia/ops"
)

const symdiffErr = `SymDiff Error: Cannot symdiff %v. Failed to compute %v`

func apply[DT any](g *exprgraph.Graph, op ops.Desc, optional string, inputs ...exprgraph.Node) (exprgraph.Node, error) {
	var name string
	switch len(inputs) {
	case 1:
		name = fmt.Sprintf("%v%v %v", optional, op, inputs[0])
	case 2:
		name = fmt.Sprintf("%v%v %v %v", optional, inputs[0], op, inputs[1])
	}
	return exprgraph.Apply[DT](g, op, name, inputs...)
}

// grN is a function that creates a name for the partial derivative.
func grN(a exprgraph.Node) string { return fmt.Sprintf("∂%v: ", a.Name()) }

// setGroup sets the Group of the given node.
func setGroup(g *exprgraph.Graph, group encoding.Group, inputs ...exprgraph.Node) {
	for _, in := range inputs {
		g.SetGroup(in, group)
	}
}

// SymDiff performs the symbolic differentiation of add.
func (op addOp[DT, T]) SymDiff(g *exprgraph.Graph, inputs []exprgraph.Node, output exprgraph.Node, grad exprgraph.Node) (retVal []exprgraph.Node, err error) {
	g.SetGroup(grad, encoding.GradientCluster)
	return []exprgraph.Node{grad, grad}, nil
}

// SymDiff performs the symbolic differentiation of sub.
func (op subOp[DT, T]) SymDiff(g *exprgraph.Graph, inputs []exprgraph.Node, output exprgraph.Node, grad exprgraph.Node) (retVal []exprgraph.Node, err error) {
	neg := negOp[DT, T]{}
	y := inputs[1]
	dzdy, err := apply[DT](g, neg, grN(y), grad)
	if err != nil {
		return nil, errors.Wrapf(err, symdiffErr, op, grN(y))
	}
	setGroup(g, encoding.GradientCluster, grad, dzdy)
	return []exprgraph.Node{grad, dzdy}, nil
}

// SymDiff performs the symbolic differentiation of mul.
func (op mulOp[DT, T]) SymDiff(g *exprgraph.Graph, inputs []exprgraph.Node, output exprgraph.Node, grad exprgraph.Node) (retVal []exprgraph.Node, err error) {
	x := inputs[0]
	y := inputs[1]

	dzdx, err := apply[DT](g, Mul[DT, T](grad, y), grN(x), grad, y)
	if err != nil {
		return nil, errors.Wrapf(err, symdiffErr, op, grN(x))
	}
	dzdy, err := apply[DT](g, Mul[DT, T](grad, x), grN(y), grad, x)
	if err != nil {
		return nil, errors.Wrapf(err, symdiffErr, op, grN(y))
	}
	setGroup(g, encoding.GradientCluster, dzdx, dzdy, grad)
	return []exprgraph.Node{dzdx, dzdy}, nil
}

// SymDiff performs the symbolic differentiation of div.
func (op divOp[DT, T]) SymDiff(g *exprgraph.Graph, inputs []exprgraph.Node, output exprgraph.Node, grad exprgraph.Node) (retVal []exprgraph.Node, err error) {
	x := inputs[0]
	y := inputs[1]

	var dzdx, dzdy exprgraph.Node
	if dzdx, err = apply[DT](g, Div[DT, T](grad, y), grN(x), grad, y); err != nil {
		return nil, errors.Wrapf(err, symdiffErr, op, grN(x))
	}

	if dzdy, err = apply[DT](g, Div[DT, T](output, y), "", output, y); err != nil {
		return nil, errors.Wrapf(err, symdiffErr, op, "output/y")
	}
	g.SetGroup(dzdy, encoding.GradientCluster)
	if dzdy, err = apply[DT](g, negOp[DT, T]{}, "", dzdy); err != nil {
		return nil, errors.Wrapf(err, symdiffErr, op, "-(output/y)")
	}
	g.SetGroup(dzdy, encoding.GradientCluster)
	if dzdy, err = apply[DT](g, Mul[DT, T](dzdy, grad), grN(y), grad, y); err != nil {
		return nil, errors.Wrapf(err, symdiffErr, op, grN(y))
	}
	g.SetGroup(dzdy, encoding.GradientCluster)
	return []exprgraph.Node{dzdx, dzdy}, nil
}

/* TENSOR FUNCTION SYMDIFFS */

// SymDiff performs the symbolic differentiation of Reshape
func (op *Reshape[DT, T]) SymDiff(g *exprgraph.Graph, inputs []exprgraph.Node, output, grad exprgraph.Node) (retVal []exprgraph.Node, err error) {
	x := inputs[0]
	op2 := &Reshape[DT, T]{To: x.Shape().Clone()}
	dydx, err := apply[DT](g, op2, grN(x), grad)
	if err != nil {
		return nil, errors.Wrapf(err, symdiffErr, op, grN(x))
	}
	setGroup(g, encoding.GradientCluster, dydx)
	return []exprgraph.Node{dydx}, nil
}

// SymDiff performs the symbolic differentiation of Reshape
func (op Slice[DT, T]) SymDiff(g *exprgraph.Graph, inputs []exprgraph.Node, output, grad exprgraph.Node) (retVal []exprgraph.Node, err error) {
	x := inputs[0]
	op2 := sliceDiff[DT, T]{op}
	dydx, err := apply[DT](g, op2, grN(x), grad)
	if err != nil {
		return nil, errors.Wrapf(err, symdiffErr, op, grN(x))
	}
	setGroup(g, encoding.GradientCluster, dydx)
	return []exprgraph.Node{dydx}, nil
}
