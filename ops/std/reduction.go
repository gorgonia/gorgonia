package stdops

import (
	"context"
	"fmt"

	"github.com/chewxy/hm"
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
		reducts = shapes.ReductOf{shapes.Var('a'), along[0]}
		for _, a := range along[1:] {
			reducts = shapes.ReductOf{reducts, a}
		}
	}
	return shapes.Arrow{
		shapes.Var('a'),
		reducts,
	}
}

func reductionTypeExpr(along shapes.Axes) hm.Type {
	a := hm.TypeVariable('a')
	d := types.MakeReduction(a, a)
	return hm.NewFnType(a, d)
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

/* Machine related */ // Do executes the op.
func (op *Sum) Do(ctx context.Context, vs ...values.Value) (retVal values.Value, err error) {
	panic("not implemented") // TODO: Implement
}

func (op *Sum) String() string { return "∑" }
