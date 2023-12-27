package gtu

import (
	"gorgonia.org/gorgonia/internal/errors"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
)

/*
// HandleFuncOpts handles funcOpts from package tensor.
func HandleFuncOpts(expShape shapes.Shape, expType dtype.Dtype, o tensor.DataOrder, strict bool, opts ...tensor.FuncOpt) (ctx context.Context, reuse tensor.DenseTensor, safe, toReuse, incr, same bool, err error) {
	fo := tensor.ParseFuncOpts(opts...)
	ctx = fo.Context()

	reuseT, incr := fo.IncrReuse()
	safe = fo.Safe()
	same = fo.Same()
	toReuse = reuseT != nil

	if toReuse {
		if reuse, err = GetDenseTensor(reuseT); err != nil {
			err = errors.Wrapf(err, "Cannot reuse a Tensor that isn't a DenseTensor. Got %T instead", reuseT)
			return
		}

		if (strict || same) && reuse.Dtype() != expType {
			err = errors.Errorf(errors.TypeMismatch, expType, reuse.Dtype())
			err = errors.Wrapf(err, "Cannot use reuse")
			return
		}

		if reuse.DataSize() != expShape.TotalSize() && !expShape.IsScalar() {
			err = errors.Errorf(errors.ShapeMismatch, reuse.Shape(), expShape)
			err = errors.Wrapf(err, "Cannot use reuse: shape mismatch - reuse.len() %v, expShape.TotalSize() %v", reuse.DataSize(), expShape.TotalSize())
			return
		}
		if !reuse.Shape().Eq(expShape) {
			cloned := expShape.Clone()
			if err = reuse.Reshape(cloned...); err != nil {
				return

			}
			tensor.ReturnInts([]int(cloned))
		}

		if !incr && reuse != nil {
			//reuse.setDataOrder(o)
			// err = reuse.reshape(expShape...)
		}

	}
	return
}

*/

func HandleFuncOpts[DT any, T tensor.Tensor[DT, T]](e tensor.Engine, t T, expShape shapes.Shape, opts ...tensor.FuncOpt) (retVal T, fo tensor.Option, err error) {
	switch e := e.(type) {
	case tensor.SpecializedFuncOptHandler[DT, T]:
		return e.HandleFuncOptsSpecialized(t, expShape, opts...)
	case tensor.FuncOptHandler[DT]:
		var ret tensor.Basic[DT]
		ret, fo, err = e.HandleFuncOpts(t, expShape, opts...)
		if err != nil {
			return retVal, fo, errors.Wrapf(err, errors.FailedFuncOpt, errors.ThisFn())
		}
		var ok bool
		if retVal, ok = ret.(T); !ok {
			return retVal, fo, errors.Errorf("Expected retVal type to be %T", retVal)
		}
		return
	case tensor.DescFuncOptHandler[DT]:
		var ret tensor.DescWithStorage
		ret, fo, err = e.HandleFuncOptsDesc(t, expShape, opts...)
		if err != nil {
			return retVal, fo, err
		}
		var ok bool
		if retVal, ok = ret.(T); !ok {
			return retVal, fo, errors.Errorf("Expected retVal type to be %T", retVal)
		}
		return
	}
	return retVal, fo, errors.Errorf(errors.EngineSupport, e, e, errors.ThisFn())
}
