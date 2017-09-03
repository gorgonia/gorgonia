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

//go:linkname vecAddF64 github.com/chewxy/vecf64.Add
func vecAddF64(a []float64, b []float64)

//go:linkname addIncrF64 github.com/chewxy/vecf64.IncrAdd
func addIncrF64(a []float64, b []float64, incr []float64)

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
		if reuse.len() != expShape.TotalSize() && !expShape.IsScalar() {
			returnOpOpt(fo)
			err = errors.Errorf(shapeMismatch, reuse.Shape(), expShape)
			err = errors.Wrapf(err, "Cannot use reuse: shape mismatch")
			return
		}
	}
	returnOpOpt(fo)
	return
}

func prepDataVSF64(a Tensor, b interface{}, reuse Tensor) (dataA *storage.Header, dataB float64, dataReuse *storage.Header, ait, iit Iterator, useIter bool, err error) {
	// get data
	dataA = a.hdr()
	// if ah, ok := a.(headerer); ok {
	// 	dataA = ah.hdr()
	// } else {
	// 	err = errors.New("Unable to get data from a")
	// 	return
	// }

	dataB = b.(float64)
	if reuse != nil {
		dataReuse = reuse.hdr()
	}

	if a.RequiresIterator() || (reuse != nil && reuse.RequiresIterator()) {
		ait = a.Iterator()
		if reuse != nil {
			iit = reuse.Iterator()
		}
		useIter = true
	}
	return
}

func (e Float64Engine) checkThree(a, b Tensor, reuse Tensor) error {
	if !a.IsNativelyAccessible() {
		return errors.Errorf(inaccessibleData, a)
	}
	if !b.IsNativelyAccessible() {
		return errors.Errorf(inaccessibleData, b)
	}

	if reuse != nil && !reuse.IsNativelyAccessible() {
		return errors.Errorf(inaccessibleData, reuse)
	}

	if a.Dtype() != Float64 {
		return errors.Errorf("Expected a to be of Float64. Got %v instead", a.Dtype())
	}
	if a.Dtype() != b.Dtype() || (reuse != nil && b.Dtype() != reuse.Dtype()) {
		return errors.Errorf("Expected a, b and reuse to have the same Dtype. Got %v, %v and %v instead", a.Dtype(), b.Dtype(), reuse.Dtype())
	}
	return nil
}

func (e Float64Engine) checkTwo(a Tensor, reuse Tensor) error {
	if !a.IsNativelyAccessible() {
		return errors.Errorf(inaccessibleData, a)
	}
	if reuse != nil && !reuse.IsNativelyAccessible() {
		return errors.Errorf(inaccessibleData, reuse)
	}

	if a.Dtype() != Float64 {
		return errors.Errorf("Expected a to be of Float64. Got %v instead", a.Dtype())
	}

	if reuse != nil && reuse.Dtype() != a.Dtype() {
		return errors.Errorf("Expected reuse to be the same as a. Got %v instead", reuse.Dtype())
	}
	return nil
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

func (e Float64Engine) FMA(a, x, y Tensor) (retVal Tensor, err error) {
	reuse := y
	if err = e.checkThree(a, x, reuse); err != nil {
		return nil, errors.Wrap(err, "Failed checks")
	}

	var dataA, dataB, dataReuse *storage.Header
	var ait, bit, iit Iterator
	var useIter bool
	if dataA, dataB, dataReuse, ait, bit, iit, useIter, _, err = prepDataVV(a, x, reuse); err != nil {
		return nil, errors.Wrap(err, "Float64Engine.FMA")
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
	reuse := y
	if err = e.checkTwo(a, reuse); err != nil {
		return nil, errors.Wrap(err, "Failed checks")
	}

	var ait, iit Iterator
	var dataTensor, dataReuse *storage.Header
	var scalar float64
	var useIter bool
	if dataTensor, scalar, dataReuse, ait, iit, useIter, err = prepDataVSF64(a, x, reuse); err != nil {
		return nil, errors.Wrapf(err, opFail, "Float64Engine.FMAScalar")
	}
	if useIter {
		err = execution.MulIterIncrVSF64(dataTensor.Float64s(), scalar, dataReuse.Float64s(), ait, iit)
		retVal = reuse
	}

	execution.MulIncrVSF64(dataTensor.Float64s(), scalar, dataReuse.Float64s())
	retVal = reuse
	return
}

// Add performs a + b elementwise. Both a and b must have the same shape.
// Acceptable FuncOpts are: UseUnsafe(), WithReuse(T), WithIncr(T)
func (e Float64Engine) Add(a Tensor, b Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	if a.RequiresIterator() || b.RequiresIterator() {
		return e.StdEng.Add(a, b, opts...)
	}

	var reuse DenseTensor
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, err = handleFuncOptsF64(a.Shape(), opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}
	if err = e.checkThree(a, b, reuse); err != nil {
		return nil, errors.Wrap(err, "Failed checks")
	}

	var hdrA, hdrB, hdrReuse *storage.Header
	var dataA, dataB, dataReuse []float64

	if hdrA, hdrB, hdrReuse, _, _, _, _, _, err = prepDataVV(a, b, reuse); err != nil {
		return nil, errors.Wrapf(err, "Float64Engine.Add")
	}
	dataA = hdrA.Float64s()
	dataB = hdrB.Float64s()
	if hdrReuse != nil {
		dataReuse = hdrReuse.Float64s()
	}

	switch {
	case incr:
		addIncrF64(dataA, dataB, dataReuse)
		retVal = reuse
	case toReuse:
		copy(dataReuse, dataA)
		vecAddF64(dataReuse, dataB)
		retVal = reuse
	case !safe:
		vecAddF64(dataA, dataB)
		retVal = a
	default:
		ret := a.Clone().(headerer)
		vecAddF64(ret.hdr().Float64s(), dataB)
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
