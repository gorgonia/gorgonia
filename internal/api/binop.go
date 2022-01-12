package gapi

import (
	"gorgonia.org/gorgonia/exprgraph"
	"gorgonia.org/gorgonia/internal/datatypes"
	"gorgonia.org/gorgonia/ops"
)

func binopSymbolic(op ops.Op, g *exprgraph.Graph, a, b datatypes.Tensor) (retVal datatypes.Tensor, err error) {
	aname, err := g.NameOf(a)
	if err != nil {
		return nil, err
	}
	bname, err := g.NameOf(b)
	if err != nil {
		return nil, err
	}
	cname := aname + op.String() + bname

	// construct node
	anode := g.NodeOf(a)
	if anode == nil {
		return nil, err
	}
	bnode := g.NodeOf(b)
	if bnode == nil {
		return nil, err
	}

	cnode, err := g.Apply(op, cname, anode, bnode)
	if err != nil {
		return nil, err
	}
	retVal = cnode
	return retVal, nil
}
