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

// tanh is a elementwise tanh.
type tanhOp[DT any, T values.Value[DT]] struct{ unop }

// String implements fmt.Stringer.
func (op tanhOp[DT, T]) String() string { return "tanh" }

// Do performs elementwise tanh.
func (op tanhOp[DT, T]) Do(ctx context.Context, vs ...T) (retVal T, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return retVal, err
	}

	a := vs[0]
	ctx2, task := trace.NewTask(ctx, op.String())
	e := tensor.GetEngine(a)
	var tanher Tanher[DT, T]
	var ok bool
	if tanher, ok = e.(Tanher[DT, T]); !ok {
		return retVal, errors.Errorf(errors.EngineSupport, e, tanher, errors.ThisFn())
	}
	if retVal, _, err = handleFuncOpts[DT, T](e, a, a.Shape()); err != nil {
		return retVal, errors.Wrapf(err, errors.FailedFuncOpt, errors.ThisFn())
	}
	if err = tanher.Tanh(ctx2, a, retVal); err != nil {
		return retVal, err
	}
	// retVal, err = tensor.Tanh(a, tensor.WithContext(ctx2))
	task.End()
	return retVal, err
}

// PreallocDo performs elementwise tanh but with a preallocated return value.
// PreallocDo allows add to implement ops.PreallocOp.
func (op tanhOp[DT, T]) PreallocDo(ctx context.Context, prealloc T, vs ...T) (retVal T, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return retVal, err
	}

	a := vs[0]
	ctx2, task := trace.NewTask(ctx, op.String())
	e := tensor.GetEngine(a)
	var tanher Tanher[DT, T]
	var ok bool
	if tanher, ok = e.(Tanher[DT, T]); !ok {
		return retVal, errors.Errorf(errors.EngineSupport, e, tanher, errors.ThisFn())
	}
	// TODO check that prealloc has the same shape as expected reetVal shape
	if err = tanher.Tanh(ctx2, a, prealloc); err != nil {
		return retVal, err
	}
	task.End()
	return prealloc, err
}

// DiffWRT returns {true} for tanh
func (op tanhOp[DT, T]) DiffWRT(inputs int) []bool { return onetrue }
