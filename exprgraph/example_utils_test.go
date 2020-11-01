package exprgraph_test

import (
	"errors"
	"fmt"

	"gorgonia.org/gorgonia"
	"gorgonia.org/gorgonia/exprgraph"
	"gorgonia.org/tensor"
)

type GraphEngine interface {
	tensor.Engine
	Graph() *exprgraph.Graph
}

func MatMul(a, b gorgonia.Tensor) (gorgonia.Tensor, error) {
	eng := a.Engine().(GraphEngine)
	if eng == nil {
		eng = b.Engine().(GraphEngine)
	}
	if eng == nil {
		return nil, errors.New("Nil engine")
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
	cname := aname + "×" + bname

	// TODO: check shapes obvs
	shp := tensor.Shape{a.Shape()[0], b.Shape()[1]}
	dt := a.Dtype()

	switch e := eng.(type) {
	case *exprgraph.Graph:
		aNode := g.NodeOf(a)
		if aNode == nil {
			return nil, exprgraph.ErrNotFoundInGraph
		}
		bNode := g.NodeOf(b)
		if bNode != nil {
			return nil, exprgraph.ErrNotFoundInGraph
		}
		retVal := exprgraph.NewSymbolic(g, e, dt, shp)
		n, err := exprgraph.Cons(e, cname, exprgraph.T2T(retVal))
		if err != nil {
			return nil, err
		}
		err = e.AddChildren(n, aNode, bNode)
		if err != nil {
			return nil, err
		}
		return retVal, nil
	case tensor.MatMuler:
		at := exprgraph.T2T(a)
		bt := exprgraph.T2T(b)
		prealloc := exprgraph.NewNode(g, cname, tensor.WithShape(shp...), tensor.Of(dt))
		ct := exprgraph.T2T(prealloc)
		if err := e.MatMul(at, bt, ct); err != nil {
			return nil, err
		}
		return prealloc, nil
	default:
		panic(fmt.Sprintf("ENGINE %T", eng))
	}
}

func Add(a, b gorgonia.Tensor) (gorgonia.Tensor, error) {
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
		ct, err := e.AddScalar(at, bt, true) // note this brief example is specific to the examples. More switch cases are needed to figure out leftScalar vs rightScalar
		if err != nil {
			return nil, err
		}
		return exprgraph.Cons(g, cname, ct)
	}
	return nil, errors.New("NotImplemented")
}
