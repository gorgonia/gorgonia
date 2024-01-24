package stdops

// Code generated by genops, which is a ops generation tool for Gorgonia. DO NOT EDIT.

import (
	"context"
	"runtime/trace"

	gctx "gorgonia.org/gorgonia/internal/context"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/tensor"
)

// divOp is the base op for elementwise division.
type divOp[DT any, T values.Value[DT]] struct{ binop }

// String implements fmt.Stringer.
func (op divOp[DT, T]) String() string { return "÷" }

// Do performs elementwise division.
func (op divOp[DT, T]) Do(ctx context.Context, vs ...T) (retVal T, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return retVal, err
	}

	a := vs[0]
	b := vs[1]

	ctx2, task := trace.NewTask(ctx, op.String())
	retVal, err = tensor.Div(a, b, tensor.WithContext(ctx2))
	task.End()
	return retVal, err
}

// PreallocDo performs elementwise division but with a preallocated return value.
// PreallocDo allows div to implement ops.PreallocOp.
func (op divOp[DT, T]) PreallocDo(ctx context.Context, prealloc T, vs ...T) (retVal T, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return retVal, err
	}

	a := vs[0]
	b := vs[1]

	ctx2, task := trace.NewTask(ctx, op.String())
	retVal, err = tensor.Div(a, b, tensor.WithReuse(prealloc), tensor.WithContext(ctx2))
	task.End()
	return retVal, err
}

// divVV is a tensor-tensor elementwise division.
type divVV[DT any, T values.Value[DT]] struct {
	divOp[DT, T]
	binopVV
}

// divVS is a tensor-scalar elementwise division.
type divVS[DT any, T values.Value[DT]] struct {
	divOp[DT, T]
	binopVS
}

// String implements fmt.Stringer.
func (op divVS[DT, T]) String() string { return "÷·" }

// divSV is a scalar-tensor elementwise division.
type divSV[DT any, T values.Value[DT]] struct {
	divOp[DT, T]
	binopSV
}

// String implements fmt.Stringer.
func (op divSV[DT, T]) String() string { return "·÷" }
