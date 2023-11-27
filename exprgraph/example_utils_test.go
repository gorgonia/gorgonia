package exprgraph_test

import (
	"context"
	"fmt"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/gorgonia/exprgraph"
	"gorgonia.org/gorgonia/ops"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/gorgonia/values/dual"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
)

type NoOp struct{}

func (NoOp) NoOp()         {}
func (NoOp) Error() string { return "NoOp" }

type GraphEngine interface {
	tensor.Engine
	Graph() *exprgraph.Graph
}

type ADOp interface {
	ops.Op
	DoDiff(ctx context.Context, inputs []gorgonia.Tensor, output gorgonia.Tensor) error
}

type Queueer interface {
	Q(op ops.Op, inputs []gorgonia.Tensor, output gorgonia.Tensor) error
}

// matmul is an Op
type matmul[DT tensor.Num, T tensor.Tensor[DT, T]] struct{}

// Arity returns the number of inputs the Op expects. -1 indicates that it's n-ary and will be determined at runtime.
func (op matmul[DT, T]) Arity() int { return 2 }

// Type informs the type of the Op (not the node). This will be used by the type system to infer the final type of the node.
func (op matmul[DT, T]) Type() hm.Type {
	return hm.NewFnType(hm.TypeVariable('a'), hm.TypeVariable('a'), hm.TypeVariable('a'))
}

// ShapeExpr informs the shape operations that the Op will do. A quick primer is given in the README of the shapes package.
func (op matmul[DT, T]) ShapeExpr() shapes.Expr {
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
func (op matmul[DT, T]) Do(ctx context.Context, vs ...T) (T, error) {
	a := vs[0].(tensor.Tensor)
	b := vs[1].(tensor.Tensor)
	return tensor.MatMul(a, b)
}

func (op matmul[DT, T]) String() string { return "×" }

func (op matmul[DT, T]) PreallocDo(ctx context.Context, prealloc T, vs ...T) (retVal T, err error) {
	if ctx != nil {
		select {
		case <-ctx.Done():
			return retVal, NoOp{}
		default:
		}

	}
	a := vs[0].(tensor.Tensor)
	b := vs[1].(tensor.Tensor)
	return tensor.MatMul(a, b, tensor.WithReuse(prealloc), tensor.WithContext(ctx))
}

func (op matmul[DT, T]) DoDiff(ctx context.Context, inputs []gorgonia.Tensor, output gorgonia.Tensor) error {
	adv := exprgraph.T2T[DT](inputs[0]).(*dual.Dual[DT, T])
	bdv := exprgraph.T2T[DT](inputs[1]).(*dual.Dual[DT, T])
	cdv := exprgraph.T2T[DT](output).(*dual.Dual[DT, T])

	advd := adv.Deriv()
	bdvd := bdv.Deriv()

	// temporary transpose
	if err := bdv.Value.T(); err != nil {
		return err
	}
	if err := adv.Value.T(); err != nil {
		return err
	}
	defer bdv.Value.UT()
	defer adv.Value.UT()

	// dA = C×B'
	if _, err := op.PreallocDo(ctx, advd, cdv.Value(), bdv.Value()); err != nil {
		return err
	}

	// dB = A'×C
	if _, err := op.PreallocDo(ctx, bdvd, adv.Value(), cdv.Value()); err != nil {
		return err
	}
	return nil
}

func MatMul[DT tensor.Num, T tensor.Tensor[DT, T]](a, b gorgonia.Tensor) (retVal gorgonia.Tensor, err error) {
	eng, ok := a.Engine().(GraphEngine)
	if !ok {
		eng, ok = b.Engine().(GraphEngine)
	}

	op := matmul[DT, T]{}
	if ok {
		// do symbolic stuff
		g := eng.Graph()

		var aname, bname string
		var anode, bnode exprgraph.Node

		if aname, err = g.NameOf(a); err != nil {
			// create a node
			aname = randomName(a)
			anode = exprgraph.New[DT](g, aname, tensor.WithBacking(a))
		}
		if bname, err = g.NameOf(b); err != nil {
			// create b node
			bname = randomName(b)
			bnode = exprgraph.New[DT](g, bname, tensor.WithBacking(b))
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
	at := exprgraph.T2T[DT](a)
	bt := exprgraph.T2T[DT](b)
	var ct T

	switch {
	case at != nil && bt != nil && retVal != nil:
		// both a and b  are values, so we can "materialize" c
		ct = retVal.(*exprgraph.Value[DT, T]).Value() // Value will "lift" *header into a proper tensor.Dense
	case at != nil && bt != nil && retVal == nil:
		// we'd have to create one ourselves
		shp := tensor.Shape{a.Shape()[0], b.Shape()[1]}
		ct = ct.Alike(tensor.WithEngine(a.Engine()), tensor.WithShape(shp...))
	default:
		// one of a or b is not a value tensor
		return retVal, nil
	}

	if ct, err = op.PreallocDo(nil, ct, at, bt); err != nil {
		return nil, err
	}
	if retVal == nil {
		retVal = ct // return not the Node, but the value.
	}

	// check if engine is backwards (i.e. requires a queue)
	// if not, return.
	var q Queueer
	q, ok = a.Engine().(Queueer)
	if !ok {
		q, ok = b.Engine().(Queueer)
	}
	if q != nil {
		// do queue stuff here
		err = q.Q(op, []gorgonia.Tensor{a, b}, retVal)
	}

	return
}

// add is addition with a scalar on the right
type add struct{}

// Arity returns the number of inputs the Op expects. -1 indicates that it's n-ary and will be determined at runtime.
func (op add) Arity() int { return 2 }

// Type informs the type of the Op (not the node). This will be used by the type system to infer the final type of the node.
func (op add) Type() hm.Type {
	return hm.NewFnType(hm.TypeVariable('a'), hm.TypeVariable('b'), hm.TypeVariable('a'))
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
	if ctx != nil {
		select {
		case <-ctx.Done():
			return nil, NoOp{}
		default:
		}

	}

	a := vs[0].(tensor.Tensor)
	b := vs[1].(tensor.Tensor)
	return tensor.Add(a, b, tensor.WithReuse(prealloc))
}

func (op add) DoDiff(ctx context.Context, inputs []gorgonia.Tensor, output gorgonia.Tensor) error {
	adv := exprgraph.T2T(inputs[0]).(*dual.Dual)
	bdv := exprgraph.T2T(inputs[1]).(*dual.Dual)

	advd := adv.Deriv()
	bdvd := bdv.Deriv()
	if _, err := tensor.Add(advd, 1.0, tensor.UseUnsafe()); err != nil {
		return err
	}
	if _, err := tensor.Add(bdvd, 1.0, tensor.UseUnsafe()); err != nil {
		return err
	}
	return nil
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

	// do the values stuff'
	at := exprgraph.T2T(a)
	bt := exprgraph.T2T(b)
	var ct tensor.Tensor
	switch {
	case at != nil && bt != nil && retVal != nil:
		// both a and b  are values, so we can "materialize" c
		ct = retVal.(*exprgraph.Node).Value() // Value will "lift" *header into a proper tensor.Dense
	case at != nil && bt != nil && retVal == nil:
		// we'd have to create one ourselves
		shp := a.Shape().Clone()
		dt := a.Dtype()
		ct = tensor.New(tensor.WithEngine(a.Engine()), tensor.WithShape(shp...), tensor.Of(dt))
	default:
		// one of a or b is not a value tensor
		return retVal, nil
	}
	if ct, err = op.PreallocDo(nil, ct, at, bt); err != nil {
		return nil, err
	}
	if retVal == nil {
		retVal = ct // return not the Node, but the value.
	}

	// check if engine is backwards (i.e. requires a queue)
	// if not, return.
	var q Queueer
	q, ok = a.Engine().(Queueer)
	if !ok {
		q, ok = b.Engine().(Queueer)
	}
	if q != nil {
		// do queue stuff here
		err = q.Q(op, []gorgonia.Tensor{a, b}, retVal)
	}

	return
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

// getDeriv is a utility function
func getDeriv(t gorgonia.Tensor) values.Value {
	n := t.(*exprgraph.Node)
	return n.Tensor.(*dual.Dual).Deriv()
}
