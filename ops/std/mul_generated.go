package stdops

// Code generated by genops, which is a ops generation tool for Gorgonia. DO NOT EDIT.

import (
	"context"
	"runtime/trace"

	gctx "gorgonia.org/gorgonia/internal/context"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/tensor"
)

// mulOp is the base op for elementwise multiplciatio=.
type mulOp[DT any, T values.Value[DT]] struct{ binop }

// String implements fmt.Stringer.
func (op mulOp[DT, T]) String() string { return "*" }

// Do performs elementwise multiplciatio=.
func (op mulOp[DT, T]) Do(ctx context.Context, vs ...T) (retVal T, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return retVal, err
	}

	a := vs[0]
	b := vs[1]

	ctx2, task := trace.NewTask(ctx, op.String())
	retVal, err = tensor.Mul(a, b, tensor.WithContext(ctx2))
	task.End()
	return retVal, err
}

// PreallocDo performs elementwise multiplciatio= but with a preallocated return value.
// PreallocDo allows mul to implement ops.PreallocOp.
func (op mulOp[DT, T]) PreallocDo(ctx context.Context, prealloc T, vs ...T) (retVal T, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return retVal, err
	}

	a := vs[0]
	b := vs[1]

	ctx2, task := trace.NewTask(ctx, op.String())
	retVal, err = tensor.Mul(a, b, tensor.WithReuse(prealloc), tensor.WithContext(ctx2))
	task.End()
	return retVal, err
}

// mulVV is a tensor-tensor elementwise multiplciatio=.
type mulVV[DT any, T values.Value[DT]] struct {
	mulOp[DT, T]
	binopVV
}

// mulVS is a tensor-scalar elementwise multiplciatio=.
type mulVS[DT any, T values.Value[DT]] struct {
	mulOp[DT, T]
	binopVS
}

// String implements fmt.Stringer.
func (op mulVS[DT, T]) String() string { return "*·" }

// mulSV is a scalar-tensor elementwise multiplciatio=.
type mulSV[DT any, T values.Value[DT]] struct {
	mulOp[DT, T]
	binopSV
}

// String implements fmt.Stringer.
func (op mulSV[DT, T]) String() string { return "·*" }
