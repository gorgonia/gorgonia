package tensor

import (
	"unsafe"

	"github.com/chewxy/gorgonia/tensor/internal/execution"
	"github.com/chewxy/gorgonia/tensor/internal/storage"
	"github.com/pkg/errors"

	"github.com/chewxy/vecf32"
)

func handleFuncOptsF32(expShape Shape, opts ...FuncOpt) (reuse DenseTensor, safe, toReuse, incr bool, err error) {
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

func prepDataVSF32(a Tensor, b interface{}, reuse Tensor) (dataA *storage.Header, dataB float32, dataReuse *storage.Header, ait, iit Iterator, useIter bool, err error) {
	// get data
	dataA = a.hdr()
	switch bt := b.(type) {
	case float32:
		dataB = bt
	case *float32:
		dataB = *bt
	default:
		err = errors.Errorf("b is not a float32: %T", b)
		return
	}
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

func (e Float32Engine) checkThree(a, b Tensor, reuse Tensor) error {
	if !a.IsNativelyAccessible() {
		return errors.Errorf(inaccessibleData, a)
	}
	if !b.IsNativelyAccessible() {
		return errors.Errorf(inaccessibleData, b)
	}

	if reuse != nil && !reuse.IsNativelyAccessible() {
		return errors.Errorf(inaccessibleData, reuse)
	}

	if a.Dtype() != Float32 {
		return errors.Errorf("Expected a to be of Float32. Got %v instead", a.Dtype())
	}
	if a.Dtype() != b.Dtype() || (reuse != nil && b.Dtype() != reuse.Dtype()) {
		return errors.Errorf("Expected a, b and reuse to have the same Dtype. Got %v, %v and %v instead", a.Dtype(), b.Dtype(), reuse.Dtype())
	}
	return nil
}

func (e Float32Engine) checkTwo(a Tensor, reuse Tensor) error {
	if !a.IsNativelyAccessible() {
		return errors.Errorf(inaccessibleData, a)
	}
	if reuse != nil && !reuse.IsNativelyAccessible() {
		return errors.Errorf(inaccessibleData, reuse)
	}

	if a.Dtype() != Float32 {
		return errors.Errorf("Expected a to be of Float32. Got %v instead", a.Dtype())
	}

	if reuse != nil && reuse.Dtype() != a.Dtype() {
		return errors.Errorf("Expected reuse to be the same as a. Got %v instead", reuse.Dtype())
	}
	return nil
}

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
	reuse := y
	if err = e.checkThree(a, x, reuse); err != nil {
		return nil, errors.Wrap(err, "Failed checks")
	}

	var dataA, dataB, dataReuse *storage.Header
	var ait, bit, iit Iterator
	var useIter bool
	if dataA, dataB, dataReuse, ait, bit, iit, useIter, _, err = prepDataVV(a, x, reuse); err != nil {
		return nil, errors.Wrap(err, "Float32Engine.FMA")
	}
	if useIter {
		err = execution.MulIterIncrF32(dataA.Float32s(), dataB.Float32s(), dataReuse.Float32s(), ait, bit, iit)
		retVal = reuse
		return
	}

	vecf32.IncrMul(dataA.Float32s(), dataB.Float32s(), dataReuse.Float32s())
	retVal = reuse
	return
}

func (e Float32Engine) FMAScalar(a Tensor, x interface{}, y Tensor) (retVal Tensor, err error) {
	reuse := y
	if err = e.checkTwo(a, reuse); err != nil {
		return nil, errors.Wrap(err, "Failed checks")
	}

	var ait, iit Iterator
	var dataTensor, dataReuse *storage.Header
	var scalar float32
	var useIter bool
	if dataTensor, scalar, dataReuse, ait, iit, useIter, err = prepDataVSF32(a, x, reuse); err != nil {
		return nil, errors.Wrapf(err, opFail, "Float32Engine.FMAScalar")
	}
	if useIter {
		err = execution.MulIterIncrVSF32(dataTensor.Float32s(), scalar, dataReuse.Float32s(), ait, iit)
		retVal = reuse
	}

	execution.MulIncrVSF32(dataTensor.Float32s(), scalar, dataReuse.Float32s())
	retVal = reuse
	return
}

// Add performs a + b elementwise. Both a and b must have the same shape.
// Acceptable FuncOpts are: UseUnsafe(), WithReuse(T), WithIncr(T)
func (e Float32Engine) Add(a Tensor, b Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	if a.RequiresIterator() || b.RequiresIterator() {
		return e.StdEng.Add(a, b, opts...)
	}

	var reuse DenseTensor
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, err = handleFuncOptsF32(a.Shape(), opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}
	if err = e.checkThree(a, b, reuse); err != nil {
		return nil, errors.Wrap(err, "Failed checks")
	}

	var hdrA, hdrB, hdrReuse *storage.Header
	var dataA, dataB, dataReuse []float32

	if hdrA, hdrB, hdrReuse, _, _, _, _, _, err = prepDataVV(a, b, reuse); err != nil {
		return nil, errors.Wrapf(err, "Float32Engine.Add")
	}
	dataA = hdrA.Float32s()
	dataB = hdrB.Float32s()
	if hdrReuse != nil {
		dataReuse = hdrReuse.Float32s()
	}

	switch {
	case incr:
		vecf32.IncrAdd(dataA, dataB, dataReuse)
		retVal = reuse
	case toReuse:
		copy(dataReuse, dataA)
		vecf32.Add(dataReuse, dataB)
		retVal = reuse
	case !safe:
		vecf32.Add(dataA, dataB)
		retVal = a
	default:
		ret := a.Clone().(headerer)
		vecf32.Add(ret.hdr().Float32s(), dataB)
		retVal = ret.(Tensor)
	}
	return
}

func (e Float32Engine) Inner(a, b Tensor) (retVal float32, err error) {
	var A, B []float32
	var AD, BD *Dense
	var ok bool

	if AD, ok = a.(*Dense); !ok {
		return 0, errors.Errorf("a is not a *Dense")
	}
	if BD, ok = b.(*Dense); !ok {
		return 0, errors.Errorf("b is not a *Dense")
	}

	A = AD.Float32s()
	B = BD.Float32s()
	retVal = whichblas.Sdot(len(A), A, 1, B, 1)
	return
}
