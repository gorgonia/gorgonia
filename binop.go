package gorgonia

import (
	"gorgonia.org/gorgonia/exprgraph"
	"gorgonia.org/gorgonia/ops"
)

// this file provides utility functions for the binary operation APIs

func binopSymbolic(op ops.Op, g *exprgraph.Graph, a, b Tensor) (retVal Tensor, err error) {
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
