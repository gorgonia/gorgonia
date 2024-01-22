package stdops

// Code generated by genops, which is a ops generation tool for Gorgonia. DO NOT EDIT.

import (
	"context"
	"runtime/trace"

	gctx "gorgonia.org/gorgonia/internal/context"
	"gorgonia.org/gorgonia/values"
)

// sign is a elementwise sign.
type signOp[DT any, T values.Value[DT]] struct{ unop }

// String implements fmt.Stringer.
func (op signOp[DT, T]) String() string { return "Sign" }

// Do performs elementwise sign.
func (op signOp[DT, T]) Do(ctx context.Context, vs ...values.Value) (retVal values.Value, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return nil, err
	}

	a := vs[0].(tensor.Tensor)
	ctx2, task := trace.NewTask(ctx, op.String())
	retVal, err = tensor.Sign(a, tensor.WithContext(ctx2))
	task.End()
	return retVal, err
}

// PreallocDo performs elementwise sign but with a preallocated return value.
// PreallocDo allows add to implement ops.PreallocOp.
func (op signOp[DT, T]) PreallocDo(ctx context.Context, prealloc values.Value, vs ...values.Value) (retVal values.Value, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return nil, err
	}

	a := vs[0].(tensor.Tensor)
	ctx2, task := trace.NewTask(ctx, op.String())
	retVal, err = tensor.Sign(a, tensor.WithReuse(prealloc), tensor.WithContext(ctx2))
	task.End()
	return retVal, err
}
