package stdops

// Code generated by genops, which is a ops generation tool for Gorgonia. DO NOT EDIT.

import (
	"context"
	"runtime/trace"

	gctx "gorgonia.org/gorgonia/internal/context"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/tensor"
)

// abs is a elementwise absolute value.
type absOp[DT any, T values.Value[DT]] struct{ unop }

// String implements fmt.Stringer.
func (op absOp[DT, T]) String() string { return "|·|" }

// Do performs elementwise absolute value.
func (op absOp[DT, T]) Do(ctx context.Context, vs ...T) (retVal T, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return retVal, err
	}

	a := vs[0]
	ctx2, task := trace.NewTask(ctx, op.String())
	retVal, err = tensor.Abs(a, tensor.WithContext(ctx2))
	task.End()
	return retVal, err
}

// PreallocDo performs elementwise absolute value but with a preallocated return value.
// PreallocDo allows add to implement ops.PreallocOp.
func (op absOp[DT, T]) PreallocDo(ctx context.Context, prealloc T, vs ...T) (retVal T, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return retVal, err
	}

	a := vs[0]
	ctx2, task := trace.NewTask(ctx, op.String())
	retVal, err = tensor.Abs(a, tensor.WithReuse(prealloc), tensor.WithContext(ctx2))
	task.End()
	return retVal, err
}
