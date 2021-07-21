package exprgraph_test

import (
	"context"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/gorgonia/exprgraph"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
)

type GraphEngine interface {
	tensor.Engine
	Graph() *exprgraph.Graph
}

// matmul is an Op
type matmul struct{}

// Arity returns the number of inputs the Op expects. -1 indicates that it's n-ary and will be determined at runtime.
func (op matmul) Arity() int { return 2 }

// Type informs the type of the Op (not the node). This will be used by the type system to infer the final type of the node.
func (op matmul) Type() hm.Type {
	return hm.NewFnType(hm.TypeVariable('a'), hm.TypeVariable('a'), hm.TypeVariable('a'))
}

// ShapeExpr informs the shape operations that the Op will do. A quick primer is given in the README of the shapes package.
func (op matmul) ShapeExpr() shapes.Expr {
	a := shapes.Var('a')
	b := shapes.Var('b')
	c := shapes.Var('c')
	return shapes.MakeArrow(
		shapes.Abstract{a, b},
		shapes.Abstract{b, c},
		shapes.Abstract{a, c},
	)
}

// Do executes the op.
func (op matmul) Do(ctx context.Context, vs ...values.Value) (values.Value, error) {
	a := vs[0].(tensor.Tensor)
	b := vs[1].(tensor.Tensor)
	return tensor.MatMul(a, b)
}

func (op matmul) String() string { return "×" }

func (op matmul) PreallocDo(ctx context.Context, prealloc values.Value, vs ...values.Value) (values.Value, error) {
	a := vs[0].(tensor.Tensor)
	b := vs[1].(tensor.Tensor)
	return tensor.MatMul(a, b, tensor.WithReuse(prealloc))
}

func MatMul(a, b gorgonia.Tensor) (retVal gorgonia.Tensor, err error) {
	eng, ok := a.Engine().(GraphEngine)
	if !ok {
		eng, ok = b.Engine().(GraphEngine)
	}

	op := matmul{}
	if ok {
		// do symbolic stuff
		g := eng.Graph()
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

		// shape checks are done here
		cnode, err := g.Apply(op, cname, anode, bnode)
		if err != nil {
			return nil, err
		}
		retVal = cnode
	}

	// do the values stuff
	at := exprgraph.T2T(a)
	bt := exprgraph.T2T(b)
	var ct tensor.Tensor
	if retVal != nil {
		ct = exprgraph.T2T(retVal)
	} else {
		// we'd have to create one ourselves
		shp := tensor.Shape{a.Shape()[0], b.Shape()[1]}
		dt := a.Dtype()
		ct = tensor.New(tensor.WithShape(shp...), tensor.Of(dt))
	}

	if ct, err = op.PreallocDo(nil, ct, at, bt); err != nil {
		return nil, err
	}
	if retVal == nil {
		retVal = ct // return not the Node, but the value.
	}

	return
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
		c, err := exprgraph.NewSymbolic(g, aname+"+"+bname, dt, shp)
		if err != nil {
			return nil, err
		}
		err = e.AddChildren(c, aNode, bNode)
		if err != nil {
			return nil, err
		}
		return c, nil
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

// ExampleOperations is a placeholder to display documentation of
// the MatMul and Add functions used in the other examples.
func Example_operations() {
	// See other examples for usage
}
