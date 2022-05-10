package stdops

import (
	"context"
	"fmt"
	"runtime/trace"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	gctx "gorgonia.org/gorgonia/internal/context"
	gerrors "gorgonia.org/gorgonia/internal/errors"
	"gorgonia.org/gorgonia/ops"
	"gorgonia.org/gorgonia/types"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
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

func denseReduction(task *trace.Task, ctx context.Context, f func(t *tensor.Dense, along ...int) (*tensor.Dense, error), along []int, input *tensor.Dense) (retVal values.Value, err error) {
	defer task.End()
	// TODO: put ctx into input.Engine somehow
	var ret *tensor.Dense
	if ret, err = f(input, along...); err != nil {
		return nil, errors.Wrapf(err, "Failed to perform reduction of %v", funcName(f))
	}
	return ret, err
}

type Reduction struct {
	op    ops.Op
	along shapes.Axes
	def   values.Value
}

// Arity returns the number of inputs the Op expects. -1 indicates that it's n-ary and will be determined at runtime.
func (op *Reduction) Arity() int { return 1 }

// Type informs the type of the Op (not the node). This will be used by the type system to infer the final type of the node.
func (op *Reduction) Type() hm.Type { return reductionTypeExpr(op.along) }

// ShapeExpr informs the shape operations that the Op will do. A quick primer is given in the README of the shapes package.
func (op *Reduction) ShapeExpr() shapes.Expr { return reductionShapeExpr(op.along) }

// Do executes the op.
func (op *Reduction) Do(ctx context.Context, vs ...values.Value) (retVal values.Value, err error) {
	panic("not implemented") // TODO: Implement
}

func (op *Reduction) String() string { return fmt.Sprintf("%v/", op.op) }

// Sum is an op that performs a reduction with + along the given axes
type Sum struct {
	along shapes.Axes
}

// Arity returns the number of inputs the Op expects. -1 indicates that it's n-ary and will be determined at runtime.
func (op *Sum) Arity() int { return 1 }

// Type informs the type of the Op (not the node). This will be used by the type system to infer the final type of the node.
func (op *Sum) Type() hm.Type { return reductionTypeExpr(op.along) }

// ShapeExpr informs the shape operations that the Op will do. A quick primer is given in the README of the shapes package.
func (op *Sum) ShapeExpr() shapes.Expr { return reductionShapeExpr(op.along) }

// Do executes the op.
func (op *Sum) Do(ctx context.Context, vs ...values.Value) (retVal values.Value, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return nil, err
	}

	switch t := vs[0].(type) {
	case *tensor.Dense:
		ctx2, task := trace.NewTask(ctx, op.String())
		return denseReduction(task, ctx2, (*tensor.Dense).Sum, axesToInts(op.along), t)
	default:
		return nil, gerrors.NYI(t)
	}
}

func (op *Sum) String() string { return "∑" }

/*
 UTILITY FUNCTIONS
*/
