package stdops

// Code generated by genops, which is a ops generation tool for Gorgonia. DO NOT EDIT.

import (
	"context"
	"runtime/trace"

	gctx "gorgonia.org/gorgonia/internal/context"
	"gorgonia.org/gorgonia/internal/errors"
	"gorgonia.org/gorgonia/values"
)

// log1p is a elementwise log1p.
type log1pOp[DT any, T values.Value[DT]] struct{ unop }

// String implements fmt.Stringer.
func (op log1pOp[DT, T]) String() string { return "Log1p" }

// Do performs elementwise log1p.
func (op log1pOp[DT, T]) Do(ctx context.Context, vs ...T) (retVal T, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return retVal, err
	}

	a := vs[0]
	ctx2, task := trace.NewTask(ctx, op.String())
	e := getEngine(a)
	var exploger ExpLoger[DT, T]
	var ok bool
	if exploger = e.(ExpLoger[DT, T]); !ok {
		return retVal, errors.Errorf(errors.EngineSupport, e, exploger, errors.ThisFn())
	}
	if retVal, _, err = handleFuncOpts[DT, T](e, a, a.Shape()); err != nil {
		return retVal, errors.Wrapf(err, errors.FailedFuncOpt, errors.ThisFn())
	}
	if err = exploger.Log1p(ctx2, a, retVal); err != nil {
		return retVal, err
	}
	// retVal, err = tensor.Log1p(a, tensor.WithContext(ctx2))
	task.End()
	return retVal, err
}

// PreallocDo performs elementwise log1p but with a preallocated return value.
// PreallocDo allows add to implement ops.PreallocOp.
func (op log1pOp[DT, T]) PreallocDo(ctx context.Context, prealloc T, vs ...T) (retVal T, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return retVal, err
	}

	a := vs[0]
	ctx2, task := trace.NewTask(ctx, op.String())
	e := getEngine(a)
	var exploger ExpLoger[DT, T]
	var ok bool
	if exploger = e.(ExpLoger[DT, T]); !ok {
		return retVal, errors.Errorf(errors.EngineSupport, e, exploger, errors.ThisFn())
	}
	// TODO check that prealloc has the same shape as expected reetVal shape
	if err = exploger.Log1p(ctx2, a, prealloc); err != nil {
		return retVal, err
	}
	task.End()
	return retVal, err
}

// DiffWRT returns {true} for log1p
func (op log1pOp[DT, T]) DiffWRT(inputs int) []bool { return onetrue }
