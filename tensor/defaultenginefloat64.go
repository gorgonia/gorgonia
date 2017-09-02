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

// func handleFuncOptsF64(expShape Shape, opts ...FuncOpt) (reuse DenseTensor, safe, toReuse, incr bool, err error) {
// 	fo := ParseFuncOpts(opts...)

// 	reuseT, incr := fo.IncrReuse()
// 	safe = fo.Safe()
// 	toReuse = reuseT != nil

// 	if toReuse {
// 		var ok bool
// 		if reuse, ok = reuseT.(DenseTensor); !ok {
// 			returnOpOpt(fo)
// 			err = errors.Errorf("Cannot reuse a different type of Tensor in a *Dense-Scalar operation. Reuse is of %T", reuseT)
// 			return
// 		}
// 		if reuse.len() != expShape.TotalSize() {
// 			returnOpOpt(fo)
// 			err = errors.Errorf(shapeMismatch, reuse.Shape(), expShape)
// 			err = errors.Wrapf(err, "Cannot use reuse: shape mismatch")
// 			return
// 		}
// 	}
// 	returnOpOpt(fo)
// 	return
// }

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

// func prepDataSVF64(a interface{}, b Tensor, reuse DenseTensor) (dataA float64, dataB, dataReuse *storage.Header, bit, iit Iterator, useIter bool, err error) {
// 	dataA = a.(float64)
// 	switch bt := b.(type) {
// 	case DenseTensor:
// 		dataB = bt.hdr()
// 		if reuse != nil {
// 			dataReuse = reuse.hdr()
// 		}
// 		if bt.requiresIterator() || (reuse != nil && reuse.requiresIterator()) {
// 			bit = IteratorFromDense(bt)
// 			if reuse != nil {
// 				iit = IteratorFromDense(reuse)
// 			}
// 			useIter = true
// 		}
// 	case *CS:
// 		err = errors.Errorf("NYI")
// 	default:
// 		err = errors.Errorf("NYI")
// 	}
// 	return
// }

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

func (e Float64Engine) FMA(a, x, y Tensor) (retVal Tensor, err error) {
	var reuse DenseTensor
	var ok bool

	if !a.IsNativelyAccessible() {
		return nil, errors.Errorf(inaccessibleData, a)
	}
	if !x.IsNativelyAccessible() {
		return nil, errors.Errorf(inaccessibleData, x)
	}
	if !y.IsNativelyAccessible() {
		return nil, errors.Errorf(inaccessibleData, y)
	}

	if reuse, ok = y.(DenseTensor); !ok {
		return nil, errors.New("y has to be a DenseTensor")
	}

	if a.Dtype() != Float64 {
		return nil, errors.Errorf("Expected a to be of Float64. Got %v instead", a.Dtype())
	}
	if a.Dtype() != x.Dtype() || x.Dtype() != y.Dtype() {
		return nil, errors.Errorf("Expected a, x and y to have the same Dtype. Got %v, %v and %v instead", a.Dtype(), x.Dtype(), y.Dtype())
	}

	var dataA, dataB, dataReuse *storage.Header
	var ait, bit, iit Iterator
	var useIter bool
	if dataA, dataB, dataReuse, ait, bit, iit, useIter, _, err = prepDataVV(a, x, reuse); err != nil {
		return nil, errors.Wrapf(err, "StdEng.Mul")
	}
	if useIter {
		err = execution.MulIterIncrF64(dataA.Float64s(), dataB.Float64s(), dataReuse.Float64s(), ait, bit, iit)
		retVal = reuse
		return
	}

	mulIncrF64(dataA.Float64s(), dataB.Float64s(), dataReuse.Float64s())
	retVal = reuse
	return
}

func (e Float64Engine) FMAScalar(a Tensor, x interface{}, y Tensor) (retVal Tensor, err error) {
	var reuse DenseTensor
	var ok bool

	if !a.IsNativelyAccessible() {
		return nil, errors.Errorf(inaccessibleData, a)
	}
	if !y.IsNativelyAccessible() {
		return nil, errors.Errorf(inaccessibleData, y)
	}

	if reuse, ok = y.(DenseTensor); !ok {
		return nil, errors.New("y has to be a DenseTensor")
	}

	if a.Dtype() != Float64 {
		return nil, errors.Errorf("Expected a to be of Float64. Got %v instead", a.Dtype())
	}

	var ait, iit Iterator
	var dataTensor, dataReuse *storage.Header
	var scalar float64
	var useIter bool
	if dataTensor, scalar, dataReuse, ait, iit, useIter, err = prepDataVSF64(a, x, reuse); err != nil {
		return nil, errors.Wrapf(err, opFail, "StdEng.FMAScalar")
	}
	if useIter {
		err = execution.MulIterIncrVSF64(dataTensor.Float64s(), scalar, dataReuse.Float64s(), ait, iit)
		retVal = reuse
	}

	execution.MulIncrVSF64(dataTensor.Float64s(), scalar, dataReuse.Float64s())
	retVal = reuse
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
