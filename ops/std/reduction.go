package stdops

import (
	"context"
	"fmt"
	"runtime/trace"

	"github.com/chewxy/hm"
	"gorgonia.org/gorgonia/exprgraph"
	gctx "gorgonia.org/gorgonia/internal/context"
	"gorgonia.org/gorgonia/internal/errors"
	"gorgonia.org/tensor/dense"

	"gorgonia.org/gorgonia/ops"
	"gorgonia.org/gorgonia/types"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/shapes"
)

// In general all reductions will reduce the dims
// { a → b | (D b = D a - 1) }
func reductionShapeExpr(along shapes.Axes) shapes.Expr {
	var reducts shapes.ReductOf
	if len(along) == 0 {
		reducts.A = shapes.Var('a')
		reducts.Along = shapes.AllAxes
	} else {
		reducts = shapes.Reduce(shapes.Var('a'), along)

	}
	return shapes.Arrow{
		shapes.Var('a'),
		reducts,
	}
}

func reductionTypeExpr(along shapes.Axes) hm.Type {
	a := hm.TypeVariable('a')
	d := types.MakeReduct(a, along)
	return hm.NewFnType(a, d)
}

func denseReduction[DT any](task *trace.Task, ctx context.Context, f func(t *dense.Dense[DT], along ...int) (*dense.Dense[DT], error), along []int, input *dense.Dense[DT]) (retVal *dense.Dense[DT], err error) {
	defer task.End()
	// TODO: put ctx into input.Engine somehow
	var ret *dense.Dense[DT]
	if ret, err = f(input, along...); err != nil {
		return nil, errors.Wrapf(err, "Failed to perform reduction of %v", errors.ThisFn())
	}
	return ret, err
}

type Reduction[DT any, T values.Value[DT]] struct {
	op    ops.Op[DT, T]
	along shapes.Axes
	def   T
}

// Arity returns the number of inputs the Op expects. -1 indicates that it's n-ary and will be determined at runtime.
func (op *Reduction[DT, T]) Arity() int { return 1 }

// Type informs the type of the Op (not the node). This will be used by the type system to infer the final type of the node.
func (op *Reduction[DT, T]) Type() hm.Type { return reductionTypeExpr(op.along) }

// ShapeExpr informs the shape operations that the Op will do. A quick primer is given in the README of the shapes package.
func (op *Reduction[DT, T]) ShapeExpr() shapes.Expr { return reductionShapeExpr(op.along) }

// Do executes the op.
func (op *Reduction[DT, T]) Do(ctx context.Context, vs ...T) (retVal T, err error) {
	panic("not implemented") // TODO: Implement
}

func (op *Reduction[DT, T]) String() string { return fmt.Sprintf("%v/", op.op) }

// Sum is an op that performs a reduction with + along the given axes
type Sum[DT any, T values.Value[DT]] struct {
	along shapes.Axes
}

// Arity returns the number of inputs the Op expects. -1 indicates that it's n-ary and will be determined at runtime.
func (op *Sum[DT, T]) Arity() int { return 1 }

// Type informs the type of the Op (not the node). This will be used by the type system to infer the final type of the node.
func (op *Sum[DT, T]) Type() hm.Type { return reductionTypeExpr(op.along) }

// ShapeExpr informs the shape operations that the Op will do. A quick primer is given in the README of the shapes package.
func (op *Sum[DT, T]) ShapeExpr() shapes.Expr { return reductionShapeExpr(op.along) }

// Do executes the op.
func (op *Sum[DT, T]) Do(ctx context.Context, vs ...T) (retVal T, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return retVal, err
	}

	switch t := any(vs[0]).(type) {
	case *dense.Dense[DT]:
		ctx2, task := trace.NewTask(ctx, op.String())
		var ret any
		ret, err = denseReduction(task, ctx2, (*dense.Dense[DT]).Sum, axesToInts(op.along), t)
		if err != nil {
			return retVal, err
		}
		retVal = ret.(T)
		return
	default:
		return retVal, errors.NYI(t)
	}
}

func (op *Sum[DT, T]) String() string { return "∑" }

func (op *Sum[DT, T]) DiffWRT(inputs int) []bool { return onetrue }

func (op *Sum[DT, T]) SymDiff(g *exprgraph.Graph, inputs []*exprgraph.Node, output *exprgraph.Node, grad *exprgraph.Node) (retVal []*exprgraph.Node, err error) {
	panic("not implemented") // TODO: Implement
}
