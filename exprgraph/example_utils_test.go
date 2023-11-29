package exprgraph_test

import (
	"context"
	"fmt"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/gorgonia/exprgraph"
	"gorgonia.org/gorgonia/ops"
	"gorgonia.org/gorgonia/values/dual"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/dense"
)

var (
	_ ops.Op[float64, *dense.Dense[float64]] = matmul[float64, *dense.Dense[float64]]{}
)

type NoOp struct{}

func (NoOp) NoOp()         {}
func (NoOp) Error() string { return "NoOp" }

type GraphEngine interface {
	tensor.Engine
	Graph() *exprgraph.Graph
}

type ADOp[DT any, T tensor.Tensor[DT, T]] interface {
	ops.Op[DT, T]
	DoDiff(ctx context.Context, inputs []gorgonia.Tensor, output gorgonia.Tensor) error
}

type Queueer[DT any, T tensor.Tensor[DT, T]] interface {
	Q(op ops.Op[DT, T], inputs []gorgonia.Tensor, output gorgonia.Tensor) error
}

type matmuler[T any] interface {
	MatMul(u T, opts ...tensor.FuncOpt) (T, error)
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
func (op matmul[DT, T]) Do(ctx context.Context, vs ...T) (retVal T, err error) {
	if ctx != nil {
		select {
		case <-ctx.Done():
			return retVal, NoOp{}
		default:
		}

	}
	a := vs[0]
	b := vs[1]
	mm, ok := any(a).(matmuler[T])
	if !ok {
		return retVal, errors.Errorf("expected %T to have a MatMul method", a)
	}
	return mm.MatMul(b)
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
	a := vs[0]
	b := vs[1]
	mm, ok := any(a).(matmuler[T])
	if !ok {
		return retVal, errors.Errorf("expected %T to have a MatMul method", a)
	}
	return mm.MatMul(b, tensor.WithReuse(prealloc))
}

func (op matmul[DT, T]) DoDiff(ctx context.Context, inputs []gorgonia.Tensor, output gorgonia.Tensor) (err error) {
	adv := exprgraph.T2T[DT](inputs[0]).(*dual.Dual[DT, T])
	bdv := exprgraph.T2T[DT](inputs[1]).(*dual.Dual[DT, T])
	cdv := exprgraph.T2T[DT](output).(*dual.Dual[DT, T])

	advd := adv.Deriv()
	bdvd := bdv.Deriv()

	// temporary transpose
	var bdvT, advT T
	if bdvT, err = bdv.Value().T(); err != nil {
		return err
	}
	if advT, err = adv.Value().T(); err != nil {
		return err
	}

	// dA = C×B'
	if _, err := op.PreallocDo(ctx, advd, cdv.Value(), bdvT); err != nil {
		return err
	}

	// dB = A'×C
	if _, err := op.PreallocDo(ctx, bdvd, advT, cdv.Value()); err != nil {
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
		cnode, err := exprgraph.Apply[DT, T](g, op, cname, anode, bnode)
		if err != nil {
			return nil, err
		}
		retVal = cnode
	}

	// check if engine supports MatMul. If not, return
	if _, ok := a.Engine().(tensor.BLA[DT, T]); !ok {
		if _, ok := b.Engine().(tensor.BLA[DT, T]); !ok {
			return
		}
	}
	// do the values stuff
	at, aok := a.(T)
	bt, bok := b.(T)
	var ct T

	switch {
	case aok && bok && retVal != nil:
		// both a and b  are values, so we can "materialize" c
		rv := exprgraph.SymToVal[DT, T](retVal.(*exprgraph.Symbolic[DT])) // turn a Symbolic into a Value
		retVal = rv
		ct = rv.Value()
	case aok && bok && retVal == nil:
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
	var q Queueer[DT, T]
	q, ok = a.Engine().(Queueer[DT, T])
	if !ok {
		q, ok = b.Engine().(Queueer[DT, T])
	}
	if q != nil {
		// do queue stuff here
		err = q.Q(op, []gorgonia.Tensor{a, b}, retVal)
	}

	return
}

type adder[T any] interface {
	Add(T, ...tensor.FuncOpt) (T, error)
}

// add is addition with a scalar on the right
type add[DT tensor.Num, T tensor.Tensor[DT, T]] struct{}

// Arity returns the number of inputs the Op expects. -1 indicates that it's n-ary and will be determined at runtime.
func (op add[DT, T]) Arity() int { return 2 }

// Type informs the type of the Op (not the node). This will be used by the type system to infer the final type of the node.
func (op add[DT, T]) Type() hm.Type {
	return hm.NewFnType(hm.TypeVariable('a'), hm.TypeVariable('b'), hm.TypeVariable('a'))
}

// ShapeExpr informs the shape operations that the Op will do. A quick primer is given in the README of the shapes package.
func (op add[DT, T]) ShapeExpr() shapes.Expr {
	a := shapes.Var('a')
	return shapes.MakeArrow(a, shapes.ScalarShape(), a)
}

// Do executes the op.
func (op add[DT, T]) Do(ctx context.Context, vs ...T) (retVal T, err error) {
	a := vs[0]
	b := vs[1]
	mm, ok := any(a).(adder[T])
	if !ok {
		return retVal, errors.Errorf("expected %T to have a Add method", a)
	}
	return mm.Add(b)
}

func (op add[DT, T]) String() string { return "+" }

func (op add[DT, T]) PreallocDo(ctx context.Context, prealloc T, vs ...T) (retVal T, err error) {
	if ctx != nil {
		select {
		case <-ctx.Done():
			return retVal, NoOp{}
		default:
		}

	}

	a := vs[0]
	b := vs[1]
	mm, ok := any(a).(adder[T])
	if !ok {
		return retVal, errors.Errorf("expected %T to have a Add method", a)
	}
	return mm.Add(b, tensor.WithReuse(prealloc))
}

func (op add[DT, T]) DoDiff(ctx context.Context, inputs []gorgonia.Tensor, output gorgonia.Tensor) error {
	adv := exprgraph.T2T[DT](inputs[0]).(*dual.Dual[DT, T])
	bdv := exprgraph.T2T[DT](inputs[1]).(*dual.Dual[DT, T])

	advd := adv.Deriv()
	bdvd := bdv.Deriv()
	// this should be replaced with a kernel call somewhere
	data := advd.Data()
	for i := range data {
		data[i] += 1
	}

	data = bdvd.Data()
	for i := range data {
		data[i] += 1
	}

	return nil
}

func Add[DT tensor.Num, T tensor.Tensor[DT, T]](a, b gorgonia.Tensor) (retVal gorgonia.Tensor, err error) {
	eng, ok := a.Engine().(GraphEngine)
	if !ok {
		eng, ok = b.Engine().(GraphEngine)
	}

	op := add[DT, T]{}
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
		cnode, err := exprgraph.Apply[DT, T](g, op, cname, anode, bnode)
		if err != nil {
			return nil, err
		}
		retVal = cnode
	}

	// check if engine supports MatMul. If not, return
	if _, ok := a.Engine().(tensor.Adder[DT, T]); !ok {
		if _, ok := b.Engine().(tensor.Adder[DT, T]); !ok {
			return
		}
	}

	// do the values stuff'
	at, aok := a.(T)
	bt, bok := b.(T)
	var ct T

	switch {
	case aok && bok && retVal != nil:
		// both a and b  are values, so we can "materialize" c
		rv := exprgraph.SymToVal[DT, T](retVal.(*exprgraph.Symbolic[DT])) // turn a Symbolic into a Value
		retVal = rv
		ct = rv.Value()
	case aok && bok && retVal == nil:
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
	var q Queueer[DT, T]
	q, ok = a.Engine().(Queueer[DT, T])
	if !ok {
		q, ok = b.Engine().(Queueer[DT, T])
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
func getDeriv[DT tensor.Num, T tensor.Tensor[DT, T]](t gorgonia.Tensor) T {
	n := t.(*exprgraph.Value[DT, T])
	return n.Basic.(*dual.Dual[DT, T]).Deriv()
}
