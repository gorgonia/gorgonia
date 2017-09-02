package tensor

import (
	"unsafe"

	"github.com/chewxy/gorgonia/tensor/internal/execution"
	"github.com/chewxy/gorgonia/tensor/internal/storage"
	"github.com/pkg/errors"

	_ "github.com/chewxy/vecf32"
)

//go:linkname vecMulF32 github.com/chewxy/vecf32.Mul
func vecMulF32(a []float32, b []float32)

//go:linkname mulIncrF32 github.com/chewxy/vecf32.IncrMul
func mulIncrF32(a []float32, b []float32, incr []float32)

//go:linkname vecScaleF32 github.com/chewxy/vecf32.Scale
func vecScaleF32(a []float32, b float32)

// func handleFuncOptsF32(expShape Shape, opts ...FuncOpt) (reuse DenseTensor, safe, toReuse, incr bool, err error) {
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

func prepDataVSF32(a Tensor, b interface{}, reuse DenseTensor) (dataA *storage.Header, dataB float32, dataReuse *storage.Header, ait, iit Iterator, useIter bool, err error) {
	dataB = b.(float32)
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

// func prepDataSVF32(a interface{}, b Tensor, reuse DenseTensor) (dataA float32, dataB, dataReuse *storage.Header, bit, iit Iterator, useIter bool, err error) {
// 	dataA = a.(float32)
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

// Float32Engine is an execution engine that is optimized to only work with float32s. It assumes all data will are float32s.
//
// Use this engine only as form of optimization. You should probably be using the basic default engine for most cases.
type Float32Engine struct {
	StdEng
}

// makeArray allocates a slice for the array
func (e Float32Engine) makeArray(arr *array, t Dtype, size int) {
	if t != Float32 {
		panic("Float32Engine only creates float32s")
	}
	s := make([]float32, size)
	arr.t = t
	arr.L = size
	arr.C = size
	arr.Ptr = unsafe.Pointer(&s[0])
	arr.fix()
}

func (e Float32Engine) FMA(a, x, y Tensor) (retVal Tensor, err error) {
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

	if a.Dtype() != Float32 {
		return nil, errors.Errorf("Expected a to be of Float32. Got %v instead", a.Dtype())
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
		err = execution.MulIterIncrF32(dataA.Float32s(), dataB.Float32s(), dataReuse.Float32s(), ait, bit, iit)
		retVal = reuse
		return
	}

	mulIncrF32(dataA.Float32s(), dataB.Float32s(), dataReuse.Float32s())
	retVal = reuse
	return
}

func (e Float32Engine) FMAScalar(a Tensor, x interface{}, y Tensor) (retVal Tensor, err error) {
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

	if a.Dtype() != Float32 {
		return nil, errors.Errorf("Expected a to be of Float32. Got %v instead", a.Dtype())
	}

	var ait, iit Iterator
	var dataTensor, dataReuse *storage.Header
	var scalar float32
	var useIter bool
	if dataTensor, scalar, dataReuse, ait, iit, useIter, err = prepDataVSF32(a, x, reuse); err != nil {
		return nil, errors.Wrapf(err, opFail, "StdEng.FMAScalar")
	}
	if useIter {
		err = execution.MulIterIncrVSF32(dataTensor.Float32s(), scalar, dataReuse.Float32s(), ait, iit)
		retVal = reuse
	}

	execution.MulIncrVSF32(dataTensor.Float32s(), scalar, dataReuse.Float32s())
	retVal = reuse
	return
}

func (e Float32Engine) Inner(a, b Tensor) (retVal interface{}, err error) {
	var A, B []float32
	var ok bool
	if A, ok = a.Data().([]float32); !ok {
		return nil, errors.Errorf("A's data is not []float32")
	}
	if B, ok = b.Data().([]float32); !ok {
		return nil, errors.Errorf("B's data is not []float32")
	}
	retVal = whichblas.Sdot(len(A), A, 1, B, 1)
	return
}
