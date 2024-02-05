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

// modOp is the base op for elementwise mod.
type modOp[DT any, T values.Value[DT]] struct{ binop }

// String implements fmt.Stringer.
func (op modOp[DT, T]) String() string { return "%" }

// Do performs elementwise mod.
func (op modOp[DT, T]) Do(ctx context.Context, vs ...T) (retVal T, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return retVal, err
	}

	a := vs[0]
	b := vs[1]

	ctx2, task := trace.NewTask(ctx, op.String())
	defer task.End()

	e, newAPA, newAPB, retVal, fo, err := tensor.PrepBinOpCis[DT](a, b)
	if err != nil {
		return retVal, err
	}
	toIncr := fo.Incr
	toBroadcast := fo.Broadcast

	arither, ok := e.(tensor.Arither[DT, Basic[DT]])
	if !ok {
		return retVal, errors.Errorf(errors.EngineSupport, e, arither, errors.ThisFn())
	}

	switch {
	case toBroadcast:
		err = arither.modBroadcastable(ctx, a, b, retVal, newAPA, newAPB, toIncr)
	default:
		if err := checkCompatibleShape(a.Shape(), b.Shape()); err != nil {
			return retVal, err
		}
		err = arither.mod(ctx2, a, b, retVal, toIncr)
	}
	return retVal, err
}

// PreallocDo performs elementwise mod but with a preallocated return value.
// PreallocDo allows mod to implement ops.PreallocOp.
func (op modOp[DT, T]) PreallocDo(ctx context.Context, prealloc T, vs ...T) (retVal T, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return retVal, err
	}

	a := vs[0]
	b := vs[1]

	ctx2, task := trace.NewTask(ctx, op.String())
	defer task.End()

	e, newAPA, newAPB, retVal, fo, err := tensor.PrepBinOpCis[DT](a, b, tensor.WithReuse(prealloc))
	if err != nil {
		return retVal, err
	}
	toIncr := fo.Incr
	toBroadcast := fo.Broadcast

	arither, ok := e.(tensor.Arither[DT, Basic[DT]])
	if !ok {
		return retVal, errors.Errorf(errors.EngineSupport, e, arither, errors.ThisFn())
	}

	switch {
	case toBroadcast:
		err = arither.modBroadcastable(ctx, a, b, retVal, newAPA, newAPB, toIncr)
	default:
		if err := checkCompatibleShape(a.Shape(), b.Shape()); err != nil {
			return retVal, err
		}
		err = arither.mod(ctx2, a, b, retVal, toIncr)
	}
	return retVal, err
}                                                 // DiffWRT returns {false, false} for mod
func (op modOp[DT, T]) DiffWRT(inputs int) []bool { return twofalses }

// modVV is a tensor-tensor elementwise mod.
type modVV[DT any, T values.Value[DT]] struct {
	modOp[DT, T]
	binopVV
}

// modVS is a tensor-scalar elementwise mod.
type modVS[DT any, T values.Value[DT]] struct {
	modOp[DT, T]
	binopVS
}

// String implements fmt.Stringer.
func (op modVS[DT, T]) String() string { return "%·" }

// modSV is a scalar-tensor elementwise mod.
type modSV[DT any, T values.Value[DT]] struct {
	modOp[DT, T]
	binopSV
}

// String implements fmt.Stringer.
func (op modSV[DT, T]) String() string { return "·%" }
