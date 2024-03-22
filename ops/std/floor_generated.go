package stdops

// Code generated by genops, which is a ops generation tool for Gorgonia. DO NOT EDIT.

import (
	"context"
	"runtime/trace"

	"gorgonia.org/gorgonia/internal"
	"gorgonia.org/gorgonia/internal/errors"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/tensor"
)

// floor is a elementwise floor.
type floorOp[DT any, T values.Value[DT]] struct{ unop }

// String implements fmt.Stringer.
func (op floorOp[DT, T]) String() string { return "⌊·⌋" }

// Do performs elementwise floor.
func (op floorOp[DT, T]) Do(ctx context.Context, vs ...T) (retVal T, err error) {
	if err := internal.HandleCtx(ctx); err != nil {
		return retVal, err
	}

	a := vs[0]
	ctx2, task := trace.NewTask(ctx, op.String())
	e := tensor.GetEngine(a)
	var intrepr IntRepr[DT, T]
	var ok bool
	if intrepr, ok = e.(IntRepr[DT, T]); !ok {
		return retVal, errors.Errorf(errors.EngineSupport, e, intrepr, errors.ThisFn())
	}
	if retVal, _, err = handleFuncOpts[DT, T](e, a, a.Shape()); err != nil {
		return retVal, errors.Wrapf(err, errors.FailedFuncOpt, errors.ThisFn())
	}
	if err = intrepr.Floor(ctx2, a, retVal); err != nil {
		return retVal, err
	}
	// retVal, err = tensor.Floor(a, tensor.WithContext(ctx2))
	task.End()
	return retVal, err
}

// PreallocDo performs elementwise floor but with a preallocated return value.
// PreallocDo allows add to implement ops.PreallocOp.
func (op floorOp[DT, T]) PreallocDo(ctx context.Context, prealloc T, vs ...T) (retVal T, err error) {
	if err := internal.HandleCtx(ctx); err != nil {
		return retVal, err
	}

	a := vs[0]
	ctx2, task := trace.NewTask(ctx, op.String())
	e := tensor.GetEngine(a)
	var intrepr IntRepr[DT, T]
	var ok bool
	if intrepr, ok = e.(IntRepr[DT, T]); !ok {
		return retVal, errors.Errorf(errors.EngineSupport, e, intrepr, errors.ThisFn())
	}
	// TODO check that prealloc has the same shape as expected reetVal shape
	if err = intrepr.Floor(ctx2, a, prealloc); err != nil {
		return retVal, err
	}
	task.End()
	return prealloc, err
}