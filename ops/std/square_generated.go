package stdops

// Code generated by genops, which is a ops generation tool for Gorgonia. DO NOT EDIT.

import (
	"context"
	"runtime/trace"

	gctx "gorgonia.org/gorgonia/internal/context"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/tensor"
)

// square is a elementwise square.
type squareOp[DT any, T values.Value[DT]] struct{ unop }

// String implements fmt.Stringer.
func (op squareOp[DT, T]) String() string { return "²" }

// Do performs elementwise square.
func (op squareOp[DT, T]) Do(ctx context.Context, vs ...T) (retVal T, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return retVal, err
	}

	a := vs[0]
	ctx2, task := trace.NewTask(ctx, op.String())
	retVal, err = tensor.Square(a, tensor.WithContext(ctx2))
	task.End()
	return retVal, err
}

// PreallocDo performs elementwise square but with a preallocated return value.
// PreallocDo allows add to implement ops.PreallocOp.
func (op squareOp[DT, T]) PreallocDo(ctx context.Context, prealloc T, vs ...T) (retVal T, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return retVal, err
	}

	a := vs[0]
	ctx2, task := trace.NewTask(ctx, op.String())
	retVal, err = tensor.Square(a, tensor.WithReuse(prealloc), tensor.WithContext(ctx2))
	task.End()
	return retVal, err
}

// DiffWRT returns {true} for square
func (op squareOp[DT, T]) DiffWRT(inputs int) []bool { return onetrue }
