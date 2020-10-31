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
	/*
		aid := g.IDOf(a)
		bid := g.IDOf(b)
				retVal := exprgraph.NewSymbolic(g, e, dt, shp)
				id := e.Insert(retVal)
				e.Name(retVal, cname) // TODO: add op
				err := e.AddChildren(id, aid, bid)
				if err != nil {
					return nil, err
				}
				return retVal, nil
	*/
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
	return nil, errors.New("NotImplemented")
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
		/*
			aid := g.IDOf(a)
			bid := g.IDOf(b)
			// TODO: check shapes obvs
			shp := a.Shape().Clone()
			dt := a.Dtype()
					retVal := exprgraph.NewSymbolic(g, e, dt, shp)
					id := e.Insert(retVal)
					e.Name(retVal, aname+"+"+bname) // TODO: add op
					err := e.AddChildren(id, aid, bid)
					if err != nil {
						return nil, err
					}
					return retVal, nil
		*/
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
