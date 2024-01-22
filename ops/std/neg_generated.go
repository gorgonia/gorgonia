package stdops

// Code generated by genops, which is a ops generation tool for Gorgonia. DO NOT EDIT.

import (
	"context"
	"runtime/trace"

	gctx "gorgonia.org/gorgonia/internal/context"
	"gorgonia.org/gorgonia/values"
)

// neg is a elementwise negation.
type negOp[DT any, T values.Value[DT]] struct{ unop }

// String implements fmt.Stringer.
func (op negOp[DT, T]) String() string { return "Neg" }

// Do performs elementwise negation.
func (op negOp[DT, T]) Do(ctx context.Context, vs ...T) (retVal T, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return retVal, err
	}

	a := vs[0]
	ctx2, task := trace.NewTask(ctx, op.String())
	retVal, err = tensor.Neg(a, tensor.WithContext(ctx2))
	task.End()
	return retVal, err
}

// PreallocDo performs elementwise negation but with a preallocated return value.
// PreallocDo allows add to implement ops.PreallocOp.
func (op negOp[DT, T]) PreallocDo(ctx context.Context, prealloc T, vs ...T) (retVal T, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return retVal, err
	}

	a := vs[0]
	ctx2, task := trace.NewTask(ctx, op.String())
	retVal, err = tensor.Neg(a, tensor.WithReuse(prealloc), tensor.WithContext(ctx2))
	task.End()
	return retVal, err
}

// DiffWRT returns {true} for neg
func (op negOp[DT, T]) DiffWRT(inputs int) []bool { return onetrue }
