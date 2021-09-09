package gorgonia

import (
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia/execution/engines"
	"gorgonia.org/gorgonia/exprgraph"
	gerrors "gorgonia.org/gorgonia/internal/errors"
	"gorgonia.org/gorgonia/ops"
	stdops "gorgonia.org/gorgonia/ops/std"
	"gorgonia.org/tensor"
)

func MatMul(a, b Tensor) (retVal Tensor, err error) {
	hybrid, ok := a.Engine().(engines.Hybrid)
	if !ok {
		hybrid, ok = b.Engine().(engines.Hybrid)
	}

	var op ops.Op = stdops.MatMul{}
	if ok {
		// do symbolic stuff
		g := hybrid.Graph()
		if retVal, err = binopSymbolic(op, g, a, b); err != nil {
			return nil, errors.Wrapf(err, gerrors.SymbolicOpFail, "MatMul")
		}
	}

	// do actual thing
	mm, ok := a.Engine().(tensor.MatMuler)
	if !ok {
		mm, ok = b.Engine().(tensor.MatMuler)
	}
	if ok {
		mm.MatMul(nil, exprgraph.T2T(a), exprgraph.T2T(b), nil)
	}
	panic("Unreachable")
}

/*

func Add(a, b Tensor) (Tensor, error) {
	eng := a.Engine().(GraphEngine)
	if eng == nil {
		eng = b.Engine().(GraphEngine)
	}

	g := eng.Graph()
	aname, err := g.NameOf(a)
	if err != nil {
		return nil, err
	}
	bname, err := g.NameOf(b)
	if err != nil {
		return nil, err
	}
	cname := aname + "+" + bname

	switch e := eng.(type) {
	case *exprgraph.Graph:
		aNode := g.NodeOf(a)
		if aNode == nil {
			return nil, exprgraph.ErrNotFoundInGraph
		}
		bNode := g.NodeOf(b)
		if bNode == nil {
			return nil, exprgraph.ErrNotFoundInGraph
		}
		// TODO: check shapes obvs
		shp := a.Shape().Clone()
		dt := a.Dtype()
		retVal := exprgraph.NewSymbolic(g, e, dt, shp)
		c, err := exprgraph.Cons(e, aname+"+"+bname, exprgraph.T2T(retVal))
		if err != nil {
			return nil, err
		}
		err = e.AddChildren(c, aNode, bNode)
		if err != nil {
			return nil, err
		}
		return retVal, nil
	case tensor.Adder:
		at := exprgraph.T2T(a)
		bt := exprgraph.T2T(b)
		// note this brief example is specific to the examples.
		// More switch cases are needed to figure out leftScalar vs rightScalar
		ct, err := e.AddScalar(at, bt, true)
		if err != nil {
			return nil, err
		}
		return exprgraph.Cons(g, cname, ct)
	}
	return nil, errors.New("NotImplemented")
}
*/
