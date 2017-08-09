package tensor

import (
	"reflect"

	"github.com/chewxy/gorgonia/tensor/internal/execution"
	"github.com/chewxy/gorgonia/tensor/internal/storage"
	"github.com/pkg/errors"
)

func (e StdEng) Map(fn interface{}, a Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	var reuse *Dense
	var safe, _, incr bool
	if reuse, safe, _, incr, _, err = prepUnaryTensor(a, nil, opts...); err != nil {
		return
	}
	switch {
	case safe && reuse == nil:
		// create reuse
		if v, ok := a.(View); ok {
			if v.IsMaterializable() {
				reuse = v.Materialize().(*Dense)
			} else {
				reuse = v.Clone().(*Dense)
			}
		} else {
			reuse = New(Of(a.Dtype()), WithShape(a.Shape().Clone()...))
		}
	case reuse != nil:
		if !reuse.IsNativelyAccessible() {
			return nil, errors.Errorf(inaccessibleData, reuse)
		}
		if a.Size() != reuse.Size() {
			return nil, errors.Errorf(shapeMismatch, a.Shape(), reuse.Shape())
		}
	}

	// PREP DATA
	typ := a.Dtype().Type
	var dataA, dataReuse, used *storage.Header
	var ait, rit, uit Iterator
	var useIter bool
	if dataA, dataReuse, ait, rit, useIter, err = prepDataUnary(a, reuse); err != nil {
		return nil, errors.Wrapf(err, "StdEng.Map")
	}

	// HANDLE USE CASES
	switch {
	case !safe:
		used = dataA
		uit = ait
	default:
		used = dataReuse
		uit = rit
	}

	// DO
	if useIter {
		err = e.E.MapIter(typ, fn, used, incr, uit)
	} else {
		err = e.E.Map(typ, fn, used, incr)
	}
	if err != nil {
		err = errors.Wrapf(err, "Unable to apply function %v to tensor of %v", fn, typ)
		return
	}

	// SET RETVAL
	switch {
	case reuse != nil:
		if err = reuseCheckShape(reuse, a.Shape()); err != nil {
			err = errors.Wrapf(err, "Reuse shape check failed")
			return
		}
		retVal = reuse
	case !safe:
		retVal = a
	default:
		retVal = reuse
	}
	return
}

func (e StdEng) Reduce(fn interface{}, a Tensor, axis int, defaultValue interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	if !a.IsNativelyAccessible() {
		return nil, errors.Errorf(inaccessibleData, a)
	}
	var at, reuse *Dense
	var dataA, dataReuse *storage.Header
	if at, reuse, dataA, dataReuse, err = e.prepReduce(a, axis, opts...); err != nil {
		err = errors.Wrap(err, "Prep Reduce failed")
		return
	}

	lastAxis := a.Dims() - 1
	typ := a.Dtype().Type

	// actual call out to the internal engine
	switch {
	case (axis == 0 && at.DataOrder().isRowMajor()) || ((axis == lastAxis || axis == len(a.Shape())-1) && at.DataOrder().isColMajor()):
		var size, split int
		if at.DataOrder().isColMajor() {
			return nil, errors.Errorf("NYI: colmajor")
		}
		size = a.Shape()[0]
		split = a.DataSize() / size
		storage.CopySliced(typ, dataReuse, 0, split, dataA, 0, split)
		err = e.E.ReduceFirst(typ, dataA, dataReuse, split, size, fn)
	case (axis == lastAxis && at.DataOrder().isRowMajor()) || (axis == 0 && at.DataOrder().isColMajor()):
		var dimSize int
		if at.DataOrder().isColMajor() {
			return nil, errors.Errorf("NYI: colmajor")
		}
		dimSize = a.Shape()[axis]
		err = e.E.ReduceLast(typ, dataA, dataReuse, dimSize, defaultValue, fn)
	default:
		dim0 := a.Shape()[0]
		dimSize := a.Shape()[axis]
		outerStride := a.Strides()[0]
		stride := a.Strides()[axis]
		expected := reuse.Strides()[0]
		err = e.E.ReduceDefault(typ, dataA, dataReuse, dim0, dimSize, outerStride, stride, expected, fn)
	}
	retVal = reuse
	return
}

func (e StdEng) OptimizedReduce(a Tensor, axis int, firstFn, lastFn, defaultFn, defaultValue interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	if !a.IsNativelyAccessible() {
		return nil, errors.Errorf(inaccessibleData, a)
	}

	var at, reuse *Dense
	var dataA, dataReuse *storage.Header
	if at, reuse, dataA, dataReuse, err = e.prepReduce(a, axis, opts...); err != nil {
		err = errors.Wrap(err, "Prep Reduce failed")
		return
	}

	lastAxis := a.Dims() - 1
	typ := a.Dtype().Type

	// actual call out to the internal engine
	switch {
	case (axis == 0 && at.DataOrder().isRowMajor()) || ((axis == lastAxis || axis == len(a.Shape())-1) && at.DataOrder().isColMajor()):
		var size, split int
		if at.DataOrder().isColMajor() {
			return nil, errors.Errorf("NYI: colmajor")
		}
		size = a.Shape()[0]
		split = a.DataSize() / size
		storage.CopySliced(typ, dataReuse, 0, split, dataA, 0, split)
		err = e.E.ReduceFirst(typ, dataA, dataReuse, split, size, firstFn)
	case (axis == lastAxis && at.DataOrder().isRowMajor()) || (axis == 0 && at.DataOrder().isColMajor()):
		var dimSize int
		if at.DataOrder().isColMajor() {
			return nil, errors.Errorf("NYI: colmajor")
		}
		dimSize = a.Shape()[axis]
		err = e.E.ReduceLast(typ, dataA, dataReuse, dimSize, defaultValue, lastFn)
	default:
		dim0 := a.Shape()[0]
		dimSize := a.Shape()[axis]
		outerStride := a.Strides()[0]
		stride := a.Strides()[axis]
		expected := reuse.Strides()[0]
		err = e.E.ReduceDefault(typ, dataA, dataReuse, dim0, dimSize, outerStride, stride, expected, defaultFn)
	}
	retVal = reuse
	return
}

func (e StdEng) Sum(a Tensor, along ...int) (retVal Tensor, err error) {
	switch at := a.(type) {
	case *Dense:
		hdr := at.hdr()
		typ := at.t.Type
		monotonic, incr1 := IsMonotonicInts(along) // if both are true, then it means all axes are accounted for, then it'll return a scalar value
		if (monotonic && incr1 && len(along) == a.Dims()) || len(along) == 0 {
			var ret interface{}
			if ret, err = execution.MonotonicSum(typ, hdr); err != nil {
				return
			}
			return New(FromScalar(ret)), nil
		}
		var firstFn, lastFn, defaultFn interface{}
		if firstFn, lastFn, defaultFn, err = execution.SumMethods(typ); err != nil {
			return
		}
		defaultVal := reflect.Zero(typ).Interface()

		retVal = a
		prev := -1
		dims := len(retVal.Shape())

		for _, axis := range along {
			if prev == -1 {
				prev = axis
			}
			if axis > prev {
				axis--
			}

			if axis >= dims {
				err = errors.Errorf(dimMismatch, retVal.Dims(), axis)
				return
			}
			if retVal, err = e.OptimizedReduce(retVal, axis, firstFn, lastFn, defaultFn, defaultVal); err != nil {
				return
			}
		}
		return

	default:
		return nil, errors.Errorf("Cannot perform Sum on %T", a)
	}
}

func (e StdEng) Min(a Tensor, along ...int) (retVal Tensor, err error) {
	switch at := a.(type) {
	case *Dense:
		hdr := at.hdr()
		typ := at.t.Type
		monotonic, incr1 := IsMonotonicInts(along) // if both are true, then it means all axes are accounted for, then it'll return a scalar value
		if (monotonic && incr1 && len(along) == a.Dims()) || len(along) == 0 {
			var ret interface{}
			if ret, err = execution.MonotonicMin(typ, hdr); err != nil {
				return
			}
			return New(FromScalar(ret)), nil
		}
		var firstFn, lastFn, defaultFn interface{}
		if firstFn, lastFn, defaultFn, err = execution.MinMethods(typ); err != nil {
			return
		}
		defaultVal := reflect.Zero(typ).Interface()

		retVal = a
		prev := -1
		dims := len(retVal.Shape())

		for _, axis := range along {
			if prev == -1 {
				prev = axis
			}
			if axis > prev {
				axis--
			}

			if axis >= dims {
				err = errors.Errorf(dimMismatch, retVal.Dims(), axis)
				return
			}

			if retVal, err = e.OptimizedReduce(retVal, axis, firstFn, lastFn, defaultFn, defaultVal); err != nil {
				return
			}
		}
		return

	default:
		return nil, errors.Errorf("Cannot perform Min on %T", a)
	}
}

func (e StdEng) Max(a Tensor, along ...int) (retVal Tensor, err error) {
	switch at := a.(type) {
	case *Dense:
		hdr := at.hdr()
		typ := at.t.Type
		monotonic, incr1 := IsMonotonicInts(along) // if both are true, then it means all axes are accounted for, then it'll return a scalar value
		if (monotonic && incr1 && len(along) == a.Dims()) || len(along) == 0 {
			var ret interface{}
			if ret, err = execution.MonotonicMax(typ, hdr); err != nil {
				return
			}
			return New(FromScalar(ret)), nil
		}
		var firstFn, lastFn, defaultFn interface{}
		if firstFn, lastFn, defaultFn, err = execution.MaxMethods(typ); err != nil {
			return
		}
		defaultVal := reflect.Zero(typ).Interface()

		retVal = a
		prev := -1
		dims := len(retVal.Shape())

		for _, axis := range along {
			if prev == -1 {
				prev = axis
			}
			if axis > prev {
				axis--
			}

			if axis >= dims {
				err = errors.Errorf(dimMismatch, retVal.Dims(), axis)
				return
			}

			if retVal, err = e.OptimizedReduce(retVal, axis, firstFn, lastFn, defaultFn, defaultVal); err != nil {
				return
			}
		}
		return

	default:
		return nil, errors.Errorf("Cannot perform Max on %T", a)
	}
}

func (StdEng) prepReduce(a Tensor, axis int, opts ...FuncOpt) (at, reuse *Dense, dataA, dataReuse *storage.Header, err error) {
	if axis >= a.Dims() {
		err = errors.Errorf(dimMismatch, axis, a.Dims())
		return
	}

	var newShape Shape
	for i, s := range a.Shape() {
		if i == axis {
			continue
		}
		newShape = append(newShape, s)
	}

	// FUNC PREP
	var safe bool
	if reuse, safe, _, _, _, err = prepUnaryTensor(a, nil, opts...); err != nil {
		err = errors.Wrap(err, "Unable to prep unary tensor")
		return
	}

	switch {
	case !safe:
		err = errors.New("Reduce only supports safe operations.")
		return
	case reuse != nil && !reuse.IsNativelyAccessible():
		err = errors.Errorf(inaccessibleData, reuse)
		return
	case reuse != nil:
		if reuse.Shape().TotalSize() != newShape.TotalSize() {
			err = errors.Errorf(shapeMismatch, reuse.Shape(), newShape)
			return
		}
		reuse.Reshape(newShape...)
	case safe && reuse == nil:
		reuse = New(Of(a.Dtype()), WithShape(newShape...))
	}

	// DATA PREP
	var useIter bool
	if dataA, dataReuse, _, _, useIter, err = prepDataUnary(a, reuse); err != nil {
		err = errors.Wrapf(err, "StdEng.Reduce data prep")
		return
	}

	var ok bool
	if at, ok = a.(*Dense); !ok || useIter {
		err = errors.Errorf("Reduce does not (yet) support iterable tensors")
		return
	}
	return
}
