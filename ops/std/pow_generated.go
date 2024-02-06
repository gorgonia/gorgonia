package stdops

// Code generated by genops, which is a ops generation tool for Gorgonia. DO NOT EDIT.

import (
	"context"
	"runtime/trace"

	gctx "gorgonia.org/gorgonia/internal/context"
	"gorgonia.org/gorgonia/internal/errors"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/tensor"
)

// powOp is the base op for elementwise exponentiation.
type powOp[DT any, T values.Value[DT]] struct{ binop }

// String implements fmt.Stringer.
func (op powOp[DT, T]) String() string { return "^" }

// Do performs elementwise exponentiation.
func (op powOp[DT, T]) Do(ctx context.Context, vs ...T) (retVal T, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return retVal, err
	}

	a := vs[0]
	b := vs[1]

	ctx2, task := trace.NewTask(ctx, op.String())
	defer task.End()

	e, newAPA, newAPB, ret, fo, err := tensor.PrepBasicBinOpCis[DT](a, b)
	if err != nil {
		return retVal, err
	}
	toIncr := fo.Incr
	toBroadcast := fo.Broadcast

	arither, ok := e.(tensor.Arither[DT, tensor.Basic[DT]])
	if !ok {
		return retVal, errors.Errorf(errors.EngineSupport, e, arither, errors.ThisFn())
	}

	switch {
	case toBroadcast:
		err = arither.PowBroadcastable(ctx, a, b, ret, newAPA, newAPB, toIncr)
	default:
		if err := checkCompatibleShape(a.Shape(), b.Shape()); err != nil {
			return retVal, err
		}
		err = arither.Pow(ctx2, a, b, ret, toIncr)
	}
	retVal = ret.(T)
	return retVal, err
}

// PreallocDo performs elementwise exponentiation but with a preallocated return value.
// PreallocDo allows pow to implement ops.PreallocOp.
func (op powOp[DT, T]) PreallocDo(ctx context.Context, prealloc T, vs ...T) (retVal T, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return retVal, err
	}

	a := vs[0]
	b := vs[1]

	ctx2, task := trace.NewTask(ctx, op.String())
	defer task.End()

	e, newAPA, newAPB, ret, fo, err := tensor.PrepBasicBinOpCis[DT](a, b, tensor.WithReuse(prealloc))
	if err != nil {
		return retVal, err
	}
	toIncr := fo.Incr
	toBroadcast := fo.Broadcast

	arither, ok := e.(tensor.Arither[DT, tensor.Basic[DT]])
	if !ok {
		return retVal, errors.Errorf(errors.EngineSupport, e, arither, errors.ThisFn())
	}

	switch {
	case toBroadcast:
		err = arither.PowBroadcastable(ctx, a, b, ret, newAPA, newAPB, toIncr)
	default:
		if err := checkCompatibleShape(a.Shape(), b.Shape()); err != nil {
			return retVal, err
		}
		err = arither.Pow(ctx2, a, b, ret, toIncr)
	}
	retVal = ret.(T)
	return retVal, err
}

// powVV is a tensor-tensor elementwise exponentiation.
type powVV[DT any, T values.Value[DT]] struct {
	powOp[DT, T]
	binopVV
}

// powVS is a tensor-scalar elementwise exponentiation.
type powVS[DT any, T values.Value[DT]] struct {
	powOp[DT, T]
	binopVS
}

// String implements fmt.Stringer.
func (op powVS[DT, T]) String() string { return "^·" }

// powSV is a scalar-tensor elementwise exponentiation.
type powSV[DT any, T values.Value[DT]] struct {
	powOp[DT, T]
	binopSV
}

// String implements fmt.Stringer.
func (op powSV[DT, T]) String() string { return "·^" }
