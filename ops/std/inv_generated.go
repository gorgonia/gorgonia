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

// inv is a elementwise 1/x.
type invOp[DT any, T values.Value[DT]] struct{ unop }

// String implements fmt.Stringer.
func (op invOp[DT, T]) String() string { return "1/·" }

// Do performs elementwise 1/x.
func (op invOp[DT, T]) Do(ctx context.Context, vs ...T) (retVal T, err error) {
	if err := internal.HandleCtx(ctx); err != nil {
		return retVal, err
	}

	a := vs[0]
	ctx2, task := trace.NewTask(ctx, op.String())
	e := tensor.GetEngine(a)
	var inver Inver[DT, T]
	var ok bool
	if inver, ok = e.(Inver[DT, T]); !ok {
		return retVal, errors.Errorf(errors.EngineSupport, e, inver, errors.ThisFn())
	}
	if retVal, _, err = handleFuncOpts[DT, T](e, a, a.Shape()); err != nil {
		return retVal, errors.Wrapf(err, errors.FailedFuncOpt, errors.ThisFn())
	}
	if err = inver.Inv(ctx2, a, retVal); err != nil {
		return retVal, err
	}
	// retVal, err = tensor.Inv(a, tensor.WithContext(ctx2))
	task.End()
	return retVal, err
}

// PreallocDo performs elementwise 1/x but with a preallocated return value.
// PreallocDo allows add to implement ops.PreallocOp.
func (op invOp[DT, T]) PreallocDo(ctx context.Context, prealloc T, vs ...T) (retVal T, err error) {
	if err := internal.HandleCtx(ctx); err != nil {
		return retVal, err
	}

	a := vs[0]
	ctx2, task := trace.NewTask(ctx, op.String())
	e := tensor.GetEngine(a)
	var inver Inver[DT, T]
	var ok bool
	if inver, ok = e.(Inver[DT, T]); !ok {
		return retVal, errors.Errorf(errors.EngineSupport, e, inver, errors.ThisFn())
	}
	// TODO check that prealloc has the same shape as expected reetVal shape
	if err = inver.Inv(ctx2, a, prealloc); err != nil {
		return retVal, err
	}
	task.End()
	return prealloc, err
}

// DiffWRT returns {true} for inv
func (op invOp[DT, T]) DiffWRT(inputs int) []bool { return onetrue }
