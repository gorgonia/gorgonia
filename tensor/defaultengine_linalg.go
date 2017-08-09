package tensor

import (
	"reflect"

	"github.com/pkg/errors"
)

func (e StdEng) Trace(t Tensor) (retVal interface{}, err error) {
	if t.Dims() != 2 {
		err = errors.Errorf(dimMismatch, 2, t.Dims())
		return
	}

	if err = typeclassCheck(t.Dtype(), numberTypes); err != nil {
		return nil, errors.Wrap(err, "Trace")
	}

	rstride := t.Strides()[0]
	cstride := t.Strides()[1]

	r := t.Shape()[0]
	c := t.Shape()[1]

	m := MinInt(r, c)
	stride := rstride + cstride

	switch data := t.Data().(type) {
	case []int:
		var trace int
		for i := 0; i < m; i++ {
			trace += data[i*stride]
		}
		retVal = trace
	case []int8:
		var trace int8
		for i := 0; i < m; i++ {
			trace += data[i*stride]
		}
		retVal = trace
	case []int16:
		var trace int16
		for i := 0; i < m; i++ {
			trace += data[i*stride]
		}
		retVal = trace
	case []int32:
		var trace int32
		for i := 0; i < m; i++ {
			trace += data[i*stride]
		}
		retVal = trace
	case []int64:
		var trace int64
		for i := 0; i < m; i++ {
			trace += data[i*stride]
		}
		retVal = trace
	case []uint:
		var trace uint
		for i := 0; i < m; i++ {
			trace += data[i*stride]
		}
		retVal = trace
	case []uint8:
		var trace uint8
		for i := 0; i < m; i++ {
			trace += data[i*stride]
		}
		retVal = trace
	case []uint16:
		var trace uint16
		for i := 0; i < m; i++ {
			trace += data[i*stride]
		}
		retVal = trace
	case []uint32:
		var trace uint32
		for i := 0; i < m; i++ {
			trace += data[i*stride]
		}
		retVal = trace
	case []uint64:
		var trace uint64
		for i := 0; i < m; i++ {
			trace += data[i*stride]
		}
		retVal = trace
	case []float32:
		var trace float32
		for i := 0; i < m; i++ {
			trace += data[i*stride]
		}
		retVal = trace
	case []float64:
		var trace float64
		for i := 0; i < m; i++ {
			trace += data[i*stride]
		}
		retVal = trace
	case []complex64:
		var trace complex64
		for i := 0; i < m; i++ {
			trace += data[i*stride]
		}
		retVal = trace
	case []complex128:
		var trace complex128
		for i := 0; i < m; i++ {
			trace += data[i*stride]
		}
		retVal = trace
	}
	return
}

func (e StdEng) Dot(x, y Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	if _, ok := x.(DenseTensor); !ok {
		err = errors.Errorf("Engine only supports working on x that is a DenseTensor. Got %T instead", x)
		return
	}

	if _, ok := y.(DenseTensor); !ok {
		err = errors.Errorf("Engine only supports working on y that is a DenseTensor. Got %T instead", y)
		return
	}

	var a, b *Dense
	if a, err = getFloatDense(x); err != nil {
		err = errors.Wrapf(err, opFail, "Dot")
		return
	}
	if b, err = getFloatDense(y); err != nil {
		err = errors.Wrapf(err, opFail, "Dot")
		return
	}

	fo := ParseFuncOpts(opts...)

	var reuse, incr *Dense
	if reuse, err = getFloatDense(fo.reuse); err != nil {
		err = errors.Wrapf(err, opFail, "Dot - reuse")
		return

	}

	if incr, err = getFloatDense(fo.incr); err != nil {
		err = errors.Wrapf(err, opFail, "Dot - incr")
		return
	}

	switch {
	case a.IsScalar() && b.IsScalar():
		var res interface{}
		switch a.t.Kind() {
		case reflect.Float64:
			res = a.GetF64(0) * b.GetF64(0)
		case reflect.Float32:
			res = a.GetF32(0) * b.GetF32(0)
		}

		switch {
		case incr != nil:
			if !incr.IsScalar() {
				err = errors.Errorf(shapeMismatch, ScalarShape(), incr.Shape())
				return
			}
			if err = e.E.MulIncr(a.Dtype().Type, a.hdr(), b.hdr(), incr.hdr()); err != nil {
				err = errors.Wrapf(err, opFail, "Dot scalar incr")
				return

			}
			retVal = incr
		case reuse != nil:
			reuse.Set(0, res)
			reuse.reshape()
			retVal = reuse
		default:
			retVal = New(FromScalar(res))
		}
		return
	case a.IsScalar():
		switch {
		case incr != nil:
			return Mul(a.ScalarValue(), b, WithIncr(incr))
		case reuse != nil:
			return Mul(a.ScalarValue(), b, WithReuse(reuse))
		}
		// default moved out
		return Mul(a.ScalarValue(), b)
	case b.IsScalar():
		switch {
		case incr != nil:
			return Mul(a, b.ScalarValue(), WithIncr(incr))
		case reuse != nil:
			return Mul(a, b.ScalarValue(), WithReuse(reuse))
		}
		return Mul(a, b.ScalarValue())
	}

	switch {
	case a.IsVector():
		switch {
		case b.IsVector():
			// check size
			if a.len() != b.len() {
				err = errors.Errorf(shapeMismatch, a.Shape(), b.Shape())
				return
			}
			return a.inner(b)
		case b.IsMatrix():
			b.T()
			defer b.UT()
			switch {
			case reuse != nil && incr != nil:
				return b.MatVecMul(a, WithReuse(reuse), WithIncr(incr))
			case reuse != nil:
				return b.MatVecMul(a, WithReuse(reuse))
			case incr != nil:
				return b.MatVecMul(a, WithIncr(incr))
			default:
			}
			return b.MatVecMul(a)
		default:

		}
	case a.IsMatrix():
		switch {
		case b.IsVector():
			switch {
			case reuse != nil && incr != nil:
				return a.MatVecMul(b, WithReuse(reuse), WithIncr(incr))
			case reuse != nil:
				return a.MatVecMul(b, WithReuse(reuse))
			case incr != nil:
				return a.MatVecMul(b, WithIncr(incr))
			default:
			}
			return a.MatVecMul(b)

		case b.IsMatrix():
			switch {
			case reuse != nil && incr != nil:
				return a.MatMul(b, WithReuse(reuse), WithIncr(incr))
			case reuse != nil:
				return a.MatMul(b, WithReuse(reuse))
			case incr != nil:
				return a.MatMul(b, WithIncr(incr))
			default:
			}
			return a.MatMul(b)
		default:
		}
	default:
	}
	as := a.Shape()
	bs := b.Shape()
	axesA := BorrowInts(1)
	axesB := BorrowInts(1)
	defer ReturnInts(axesA)
	defer ReturnInts(axesB)

	var lastA, secondLastB int

	lastA = len(as) - 1
	axesA[0] = lastA
	if len(bs) >= 2 {
		secondLastB = len(bs) - 2
	} else {
		secondLastB = 0
	}
	axesB[0] = secondLastB

	if as[lastA] != bs[secondLastB] {
		err = errors.Errorf(shapeMismatch, as, bs)
		return
	}

	var rd *Dense
	if rd, err = a.TensorMul(b, axesA, axesB); err != nil {
		return
	}

	if reuse != nil {
		copyDense(reuse, rd)
		ReturnAP(reuse.AP)
		reuse.AP = rd.AP.Clone()
		defer ReturnTensor(rd)
		// swap out the underlying data and metadata
		// reuse.data, rd.data = rd.data, reuse.data
		// reuse.AP, rd.AP = rd.AP, reuse.AP
		// defer ReturnTensor(rd)

		retVal = reuse
	} else {
		retVal = rd
	}
	return

}
