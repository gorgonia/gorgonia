package tensor

import (
	"unsafe"

	"github.com/chewxy/gorgonia/tensor/internal/execution"
	"github.com/chewxy/gorgonia/tensor/internal/storage"
	"github.com/pkg/errors"

	_ "github.com/chewxy/vecf64"
)

//go:linkname vecMulF64 github.com/chewxy/vecf64.Mul
func vecMulF64(a []float64, b []float64)

//go:linkname mulIncrF64 github.com/chewxy/vecf64.IncrMul
func mulIncrF64(a []float64, b []float64, incr []float64)

//go:linkname vecScaleF64 github.com/chewxy/vecf64.Scale
func vecScaleF64(a []float64, b float64)

func handleFuncOptsF64(expShape Shape, opts ...FuncOpt) (reuse DenseTensor, safe, toReuse, incr bool, err error) {
	fo := ParseFuncOpts(opts...)

	reuseT, incr := fo.IncrReuse()
	safe = fo.Safe()
	toReuse = reuseT != nil

	if toReuse {
		var ok bool
		if reuse, ok = reuseT.(DenseTensor); !ok {
			returnOpOpt(fo)
			err = errors.Errorf("Cannot reuse a different type of Tensor in a *Dense-Scalar operation. Reuse is of %T", reuseT)
			return
		}
		if reuse.len() != expShape.TotalSize() {
			returnOpOpt(fo)
			err = errors.Errorf(shapeMismatch, reuse.Shape(), expShape)
			err = errors.Wrapf(err, "Cannot use reuse: shape mismatch")
			return
		}
	}
	returnOpOpt(fo)
	return
}

func prepDataVSF64(a Tensor, b interface{}, reuse DenseTensor) (dataA *storage.Header, dataB float64, dataReuse *storage.Header, ait, iit Iterator, useIter bool, err error) {
	dataB = b.(float64)
	switch at := a.(type) {
	case DenseTensor:
		dataA = at.hdr()
		if reuse != nil {
			dataReuse = reuse.hdr()
		}
		if at.requiresIterator() || (reuse != nil && reuse.requiresIterator()) {
			ait = IteratorFromDense(at)
			if reuse != nil {
				iit = IteratorFromDense(reuse)
			}
			useIter = true
		}
	case *CS:
		err = errors.Errorf("NYI")
	default:
		err = errors.Errorf("NYI")
	}
	return
}

func prepDataSVF64(a interface{}, b Tensor, reuse DenseTensor) (dataA float64, dataB, dataReuse *storage.Header, bit, iit Iterator, useIter bool, err error) {
	dataA = a.(float64)
	switch bt := b.(type) {
	case DenseTensor:
		dataB = bt.hdr()
		if reuse != nil {
			dataReuse = reuse.hdr()
		}
		if bt.requiresIterator() || (reuse != nil && reuse.requiresIterator()) {
			bit = IteratorFromDense(bt)
			if reuse != nil {
				iit = IteratorFromDense(reuse)
			}
			useIter = true
		}
	case *CS:
		err = errors.Errorf("NYI")
	default:
		err = errors.Errorf("NYI")
	}
	return
}

// Float64Engine is an execution engine that is optimized to only work with float64s. It assumes all data will are float64s.
//
// Use this engine only as form of optimization. You should probably be using the basic default engine for most cases.
type Float64Engine struct {
	StdEng
}

// makeArray allocates a slice for the array
func (e Float64Engine) makeArray(arr *array, t Dtype, size int) {
	if t != Float64 {
		panic("Float64Engine only creates float64s")
	}
	s := make([]float64, size)
	arr.t = t
	arr.L = size
	arr.C = size
	arr.Ptr = unsafe.Pointer(&s[0])
	arr.fix()
}

func (e Float64Engine) Mul(a Tensor, b Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	var reuse DenseTensor
	var safe, toReuse, incr bool
	// if reuse, safe, toReuse, incr, _, err = handleFuncOpts(a.Shape(), a.Dtype(), true, opts...); err != nil {
	if reuse, safe, toReuse, incr, err = handleFuncOptsF64(a.Shape(), opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}

	typ := a.Dtype().Type
	var dataA, dataB, dataReuse *storage.Header
	var ait, bit, iit Iterator
	var useIter, swap bool
	if dataA, dataB, dataReuse, ait, bit, iit, useIter, swap, err = prepDataVV(a, b, reuse); err != nil {
		return nil, errors.Wrapf(err, "StdEng.Mul")
	}
	if useIter {
		switch {
		case incr:
			err = execution.MulIterIncrF64(dataA.Float64s(), dataB.Float64s(), dataReuse.Float64s(), ait, bit, iit)
			retVal = reuse
		case toReuse:
			storage.CopyIter(typ, dataReuse, dataA, iit, ait)
			ait.Reset()
			iit.Reset()
			err = execution.MulIterF64(dataReuse.Float64s(), dataB.Float64s(), iit, bit)
			retVal = reuse
		case !safe:
			err = execution.MulIterF64(dataA.Float64s(), dataB.Float64s(), ait, bit)
			retVal = a
		default:
			var ret headerer
			if swap {
				ret = b.Clone().(headerer)
			} else {
				ret = a.Clone().(headerer)
			}
			err = execution.MulIterF64(ret.hdr().Float64s(), dataB.Float64s(), ait, bit)
			retVal = ret.(Tensor)
		}
		return
	}
	switch {
	case incr:
		mulIncrF64(dataA.Float64s(), dataB.Float64s(), dataReuse.Float64s())
		retVal = reuse
	case toReuse:
		storage.Copy(typ, dataReuse, dataA)
		vecMulF64(dataReuse.Float64s(), dataB.Float64s())
		retVal = reuse
	case !safe:
		vecMulF64(dataA.Float64s(), dataB.Float64s())
		retVal = a
	default:
		var ret headerer
		if swap {
			ret = b.Clone().(headerer)
		} else {
			ret = a.Clone().(headerer)
		}
		vecMulF64(ret.hdr().Float64s(), dataB.Float64s())
		retVal = ret.(Tensor)
	}
	return
}

func (e Float64Engine) MulScalar(t Tensor, s interface{}, leftTensor bool, opts ...FuncOpt) (retVal Tensor, err error) {
	var reuse DenseTensor
	var safe, toReuse, incr bool
	// if reuse, safe, toReuse, incr, _, err = handleFuncOpts(t.Shape(), t.Dtype(), true, opts...); err != nil {
	if reuse, safe, toReuse, incr, err = handleFuncOptsF64(t.Shape(), opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}
	a := t
	typ := t.Dtype().Type
	var ait, bit, iit Iterator

	var dataTensor, dataReuse *storage.Header
	var scalar float64
	var useIter bool

	if leftTensor {
		if dataTensor, scalar, dataReuse, ait, iit, useIter, err = prepDataVSF64(t, s, reuse); err != nil {
			return nil, errors.Wrapf(err, opFail, "StdEng.Mul")
		}
	} else {
		if scalar, dataTensor, dataReuse, bit, iit, useIter, err = prepDataSVF64(s, t, reuse); err != nil {
			return nil, errors.Wrapf(err, opFail, "StdEng.Mul")
		}
	}

	if useIter {
		switch {
		case incr && leftTensor:
			err = execution.MulIterIncrVSF64(dataTensor.Float64s(), scalar, dataReuse.Float64s(), ait, iit)
			retVal = reuse
		case incr && !leftTensor:
			err = execution.MulIterIncrSVF64(scalar, dataTensor.Float64s(), dataReuse.Float64s(), bit, iit)
			retVal = reuse
		case toReuse && leftTensor:
			storage.CopyIter(typ, dataReuse, dataTensor, iit, ait)
			ait.Reset()
			iit.Reset()

			err = execution.MulIterVSF64(dataReuse.Float64s(), scalar, iit)
			retVal = reuse
		case toReuse && !leftTensor:
			storage.CopyIter(typ, dataReuse, dataTensor, iit, bit)
			iit.Reset()
			bit.Reset()
			err = execution.MulIterSVF64(scalar, dataReuse.Float64s(), iit)
			retVal = reuse
		case !safe && leftTensor:
			err = execution.MulIterVSF64(dataTensor.Float64s(), scalar, ait)
			retVal = a
		case !safe && !leftTensor:
			err = execution.MulIterSVF64(scalar, dataTensor.Float64s(), bit)
			retVal = a
		default:
			ret := a.Clone().(headerer)
			if leftTensor {
				err = execution.MulIterVSF64(ret.hdr().Float64s(), scalar, ait)
			} else {
				err = execution.MulIterSVF64(scalar, ret.hdr().Float64s(), bit)
			}
			retVal = ret.(Tensor)
		}
		return
	}
	switch {
	case incr && leftTensor:
		execution.MulIncrVSF64(dataTensor.Float64s(), scalar, dataReuse.Float64s())
		retVal = reuse
	case incr && !leftTensor:
		execution.MulIncrSVF64(scalar, dataTensor.Float64s(), dataReuse.Float64s())
		retVal = reuse
	case toReuse && leftTensor:
		storage.Copy(typ, dataReuse, dataTensor)
		vecScaleF64(dataReuse.Float64s(), scalar)
		retVal = reuse
	case toReuse && !leftTensor:
		storage.Copy(typ, dataReuse, dataTensor)
		vecScaleF64(dataReuse.Float64s(), scalar)
		retVal = reuse
	case !safe:
		vecScaleF64(dataTensor.Float64s(), scalar)
		retVal = a
	default:
		ret := a.Clone().(headerer)
		vecScaleF64(ret.hdr().Float64s(), scalar)
		retVal = ret.(Tensor)
	}
	return
}

func (e Float64Engine) Inner(a, b Tensor) (retVal interface{}, err error) {
	var A, B []float64
	var ok bool
	if A, ok = a.Data().([]float64); !ok {
		return nil, errors.Errorf("A's data is not []float64")
	}
	if B, ok = b.Data().([]float64); !ok {
		return nil, errors.Errorf("B's data is not []float64")
	}
	retVal = whichblas.Ddot(len(A), A, 1, B, 1)
	return
}
