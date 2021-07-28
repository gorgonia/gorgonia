package stdops

import (
	"context"
	"runtime/trace"

	"github.com/chewxy/hm"
	"gorgonia.org/gorgonia/types"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
)

type MatMul struct{ binop }

// Type informs the type of the MatMul: Matrix a → Matrix a → Matrix a
func (op MatMul) Type() hm.Type {
	a := hm.TypeVariable('a')
	t := types.MakeTensorType(2, a) // Matrix a
	return hm.NewFnType(t, t, t)
}

// ShapeExpr informs the shape operations of MatMul: (a, b) → (b, c) → (a, c)
func (op MatMul) ShapeExpr() shapes.Expr {
	a := shapes.Var('a')
	b := shapes.Var('b')
	c := shapes.Var('c')
	return shapes.MakeArrow(
		shapes.Abstract{a, b},
		shapes.Abstract{b, c},
		shapes.Abstract{a, c},
	)
}

// Do performs the matrix multiplication
func (op MatMul) Do(ctx context.Context, vs ...values.Value) (values.Value, error) {
	if err := handleCtx(ctx); err != nil {
		return nil, err
	}

	a := vs[0]
	b := vs[1]
	ctx2, task := trace.NewTask(ctx, op.String())
	retVal, err := tensor.MatMul(a, b, tensor.WithContext(ctx2))
	task.End()
	return retVal, err
}

func (op MatMul) PreallocDo(ctx context.Context, prealloc values.Value, vs ...values.Value) (values.Value, error) {
	if err := handleCtx(ctx); err != nil {
		return nil, err
	}
	a := vs[0].(tensor.Tensor)
	b := vs[1].(tensor.Tensor)
	ctx2, task := trace.NewTask(ctx, op.String())
	retVal, err := tensor.MatMul(a, b, tensor.WithReuse(prealloc), tensor.WithContext(ctx2))
	task.End()
	return retVal, err
}

func (op MatMul) String() string { return "×" }
