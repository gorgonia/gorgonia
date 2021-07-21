package exprgraph_test

import (
	"context"
	"fmt"
	"log"

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

		var aname, bname string
		var anode, bnode *exprgraph.Node

		if aname, err = g.NameOf(a); err != nil {
			// create a node
			aname = randomName(a)
			if anode, err = exprgraph.Cons(g, aname, a.(tensor.Tensor)); err != nil {
				return nil, err
			}
		}
		if bname, err = g.NameOf(b); err != nil {
			// create b node
			bname = randomName(b)
			if bnode, err = exprgraph.Cons(g, bname, b.(tensor.Tensor)); err != nil {
				return nil, err
			}
		}
		cname := aname + op.String() + bname

		// construct node
		if anode == nil {
			if anode = g.NodeOf(a); anode == nil {
				return nil, errors.Errorf("MatMul: Cannot find Node a of %v", a)
			}
		}
		if bnode == nil {
			if bnode = g.NodeOf(b); bnode == nil {
				return nil, errors.Errorf("MatMul: Cannot find Node b of %v", b)
			}
		}

		// shape checks are done here
		cnode, err := g.Apply(op, cname, anode, bnode)
		if err != nil {
			return nil, err
		}
		retVal = cnode
	}

	// check if engine supports MatMul. If not, return
	if _, ok := a.Engine().(tensor.MatMuler); !ok {
		if _, ok := b.Engine().(tensor.MatMuler); !ok {
			return
		}
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

// add is addition with a scalar on the right
type add struct{}

// Arity returns the number of inputs the Op expects. -1 indicates that it's n-ary and will be determined at runtime.
func (op add) Arity() int { return 2 }

// Type informs the type of the Op (not the node). This will be used by the type system to infer the final type of the node.
func (op add) Type() hm.Type {
	return hm.NewFnType(hm.TypeVariable('a'), hm.TypeVariable('a'), hm.TypeVariable('a'))
}

// ShapeExpr informs the shape operations that the Op will do. A quick primer is given in the README of the shapes package.
func (op add) ShapeExpr() shapes.Expr {
	a := shapes.Var('a')
	return shapes.MakeArrow(a, shapes.ScalarShape(), a)
}

// Do executes the op.
func (op add) Do(ctx context.Context, vs ...values.Value) (values.Value, error) {
	a := vs[0].(tensor.Tensor)
	b := vs[1].(tensor.Tensor)
	return tensor.Add(a, b)
}

func (op add) String() string { return "+" }

func (op add) PreallocDo(ctx context.Context, prealloc values.Value, vs ...values.Value) (values.Value, error) {
	a := vs[0].(tensor.Tensor)
	b := vs[1].(tensor.Tensor)
	c, err := tensor.Add(a, b, tensor.WithReuse(prealloc))
	if err != nil {
		log.Printf("ERR %v", err)
	}
	return c, err
}

func Add(a, b gorgonia.Tensor) (retVal gorgonia.Tensor, err error) {
	eng, ok := a.Engine().(GraphEngine)
	if !ok {
		eng, ok = b.Engine().(GraphEngine)
	}

	op := add{}
	if ok {
		// do symbolic stuff
		g := eng.Graph()

		var aname, bname string
		var anode, bnode *exprgraph.Node

		if aname, err = g.NameOf(a); err != nil {
			// create a node
			aname = randomName(a)
			if anode, err = exprgraph.Cons(g, aname, a.(tensor.Tensor)); err != nil {
				return nil, err
			}
		}
		if bname, err = g.NameOf(b); err != nil {
			// create b node
			bname = randomName(b)
			if bnode, err = exprgraph.Cons(g, bname, b.(tensor.Tensor)); err != nil {
				return nil, err
			}
		}
		cname := aname + op.String() + bname

		// construct node
		if anode == nil {
			if anode = g.NodeOf(a); anode == nil {
				return nil, errors.Errorf("MatMul: Cannot find Node a of %v", a)
			}
		}
		if bnode == nil {
			if bnode = g.NodeOf(b); bnode == nil {
				return nil, errors.Errorf("MatMul: Cannot find Node b of %v", b)
			}
		}

		// shape checks are done here
		cnode, err := g.Apply(op, cname, anode, bnode)
		if err != nil {
			return nil, err
		}
		retVal = cnode
	}

	// check if engine supports MatMul. If not, return
	if _, ok := a.Engine().(tensor.Adder); !ok {
		if _, ok := b.Engine().(tensor.Adder); !ok {
			return
		}
	}

	// do the values stuff
	at := exprgraph.T2T(a)
	bt := exprgraph.T2T(b)
	var ct tensor.Tensor
	if retVal != nil {
		ct = exprgraph.T2T(retVal)
		if ct, err = op.PreallocDo(nil, ct, at, bt); err != nil {
			return nil, err
		}
		return
	} else {
		return op.Do(nil, at, bt)
	}

	return nil, errors.New("NotImplemented")
}

// ExampleOperations is a placeholder to display documentation of
// the MatMul and Add functions used in the other examples.
func Example_operations() {
	// See other examples for usage
}

var rndCounter int

func randomName(a gorgonia.Tensor) string {
	rndCounter++
	return fmt.Sprintf("Random_%d", rndCounter)
}
