package tensor

import (
	"github.com/chewxy/gorgonia/tensor/internal/storage"
	"github.com/pkg/errors"
)

func (e StdEng) Map(a Tensor, fn interface{}, opts ...FuncOpt) error {
	var reuse *Dense
	var _, toReuse, incr bool // safe has been commented out
	if reuse, _, toReuse, _, _, err = prepUnaryTensor(a, nil, opts...); err != nil {
		return
	}
	if reuse != nil && !reuse.IsNativelyAccessible() {
		err = errors.Errorf(inaccessibleData, reuse)
		return
	}
	typ := a.Dtype().Type
	var dataA, dataReuse *storage.Header
	var ait, iit Iterator
	var useIter bool
	if dataA, dataReuse, ait, iit, useIter, err = prepDataUnary(a, reuse); err != nil {
		return errors.Wrapf(err, "StdEng.Map")
	}
	if toReuse {
		storage.Copy(dataReuse, dataA)
		ait = rit
	}
	if useIter {
		return e.E.MapIter(typ, fn, a, false, ait)
	}
	return e.E.Map(typ, fn, a, false)
}

func (e StdEng) Reduce(a Tensor, axis int, fn, defaultValue interface{}, opts ...FuncOpt) error {
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
	lastAxis := a.Dims() - 1

	// DATA PREP
	var reuse *Dense
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, _, _, err = prepUnaryTensor(a, nil, opts...); err != nil {
		return errors.Wrap(err, "Unable to prep unary tensor")
	}
	if !safe {
		return errors.New("Reduce only supports safe operations.")
	}
	if reuse != nil && !reuse.IsNativelyAccessible() {
		return errors.Errorf(inaccessibleData, reuse)
	}
	if reuse == nil {
		reuse = New(Of(a.Dtype()), WithShape(newShape...))
	}

	typ := a.Dtype().Type
	var dataA, dataReuse *storage.Header
	var useIter bool
	if dataA, dataReuse, _, _, useIter, err = prepDataUnary(a, reuse); err != nil {
		return errors.Wrapf(err, "StdEng.Reduce")
	}
	at, ok := a.(DenseTensor)
	if useIter || !ok {
		return errors.Errorf("Reduce does not (yet) support iterable tensors")
	}

	// actual call out to the internal engine
	switch {
	case (axis == 0 && at.DataOrder().isRowMajor()) || ((axis == lastAxis || axis == len(a.Shape())-1) && at.DataOrder().isColMajor()):
		var size, split int
		if at.DataOrder().isColMajor() {
			return errors.Errorf("NYI: colmajor")

		}
		size = a.Shape()[0]
		split = a.DataSize() / size
		storage.CopySliced(dataReuse, 0, split, dataA, 0, split)
		return e.E.ReduceFirst(typ, dataA, dataReuse, split, size, fn)
	case (axis == lastAxis && at.DataOrder().isRowMajor()) || (axis == 0 && at.DataOrder().isColMajor()):
		var dimSize int
		if at.DataOrder().isColMajor() {
			return errors.Errorf("NYI: colmajor")
		}
		dimSize = a.Shape()[axis]
		return e.E.ReduceLast(typ, dataA, dataReuse, dimSize, defaultValue, fn)
	default:
		dim0 := a.Shape()[0]
		dimSize := a.Shape()[axis]
		outerStride := a.Strides()[0]
		stride := a.Strides()[axis]
		expected := reuse.Strides()[0]
		return e.E.ReduceDefault(typ, dataA, dataReuse, dim0, dimSize, outerStride, stride, expected, fn)
	}
	return nil
}
