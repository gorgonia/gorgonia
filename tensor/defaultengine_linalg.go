package tensor

import (
	"reflect"

	"github.com/gonum/blas"
	"github.com/gonum/matrix"
	"github.com/gonum/matrix/mat64"
	"github.com/pkg/errors"
)

//  Trace returns the trace of a matrix (i.e. the sum of the diagonal elements). If the Tensor provided is not a matrix, it will return an error
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

	var a, b DenseTensor
	if a, err = getFloatDenseTensor(x); err != nil {
		err = errors.Wrapf(err, opFail, "Dot")
		return
	}
	if b, err = getFloatDenseTensor(y); err != nil {
		err = errors.Wrapf(err, opFail, "Dot")
		return
	}

	fo := ParseFuncOpts(opts...)

	var reuse, incr DenseTensor
	if reuse, err = getFloatDenseTensor(fo.reuse); err != nil {
		err = errors.Wrapf(err, opFail, "Dot - reuse")
		return

	}

	if incr, err = getFloatDenseTensor(fo.incr); err != nil {
		err = errors.Wrapf(err, opFail, "Dot - incr")
		return
	}

	switch {
	case a.IsScalar() && b.IsScalar():
		var res interface{}
		switch a.Dtype().Kind() {
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
			var ret interface{}
			if ret, err = e.Inner(a, b); err != nil {
				return nil, errors.Wrapf(err, opFail, "Dot")
			}
			return New(FromScalar(ret)), nil
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
		ReturnAP(reuse.Info())
		reuse.setAP(rd.Info().Clone())
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

// TODO: make it take DenseTensor
func (e StdEng) SVD(a Tensor, uv, full bool) (s, u, v Tensor, err error) {
	var t *Dense
	var ok bool
	if err = e.checkAccessible(a); err != nil {
		return nil, nil, nil, errors.Wrapf(err, "opFail", "SVD")
	}
	if t, ok = a.(*Dense); !ok {
		return nil, nil, nil, errors.Errorf("StdEng only performs SVDs for DenseTensors. Got %T instead", a)
	}
	if !isFloat(t.Dtype()) {
		return nil, nil, nil, errors.Errorf("StdEng can only perform SVDs for float64 and float32 type. Got tensor of %v instead", t.Dtype())
	}

	if !t.IsMatrix() {
		return nil, nil, nil, errors.Errorf(dimMismatch, 2, t.Dims())
	}

	var mat *mat64.Dense
	var svd mat64.SVD

	if mat, err = ToMat64(t, UseUnsafe()); err != nil {
		return
	}

	switch {
	case full && uv:
		ok = svd.Factorize(mat, matrix.SVDFull)
	case !full && uv:
		ok = svd.Factorize(mat, matrix.SVDThin)
	case full && !uv:
		// illogical state - if you specify "full", you WANT the UV matrices
		// error
		err = errors.Errorf("SVD requires computation of `u` and `v` matrices if `full` was specified.")
		return
	default:
		// by default, we return only the singular values
		ok = svd.Factorize(mat, matrix.SVDNone)
	}

	if !ok {
		// error
		err = errors.Errorf("Unable to compute SVD")
		return
	}

	// extract values
	var um, vm mat64.Dense
	s = recycledDense(Float64, Shape{MinInt(t.Shape()[0], t.Shape()[1])})
	svd.Values(s.Data().([]float64))
	if uv {
		um.UFromSVD(&svd)
		vm.VFromSVD(&svd)

		u = FromMat64(&um, UseUnsafe(), As(t.t))
		v = FromMat64(&vm, UseUnsafe(), As(t.t))
	}

	return
}

// Inner is a thin layer over BLAS's D/Sdot.
// It returns a scalar value, wrapped in an interface{}, which is not quite nice.
func (e StdEng) Inner(a, b Tensor) (retVal interface{}, err error) {
	var ad, bd DenseTensor
	if ad, bd, err = e.checkTwoFloatTensors(a, b); err != nil {
		return nil, errors.Wrapf(err, opFail, "StdEng.Inner")
	}

	switch A := ad.Data().(type) {
	case []float32:
		B := bd.Float32s()
		retVal = whichblas.Sdot(len(A), A, 1, B, 1)
	case []float64:
		B := bd.Float64s()
		retVal = whichblas.Ddot(len(A), A, 1, B, 1)
	}
	return
}

// MatVecMul is a thin layer over BLAS' DGEMV
// Because DGEMV computes:
// 		y = αA * x + βy
// we set beta to 0, so we don't have to manually zero out the reused/retval tensor data
func (e StdEng) MatVecMul(a, b, prealloc Tensor) (err error) {
	// check all are DenseTensors
	var ad, bd, pd DenseTensor
	if ad, bd, pd, err = e.checkThreeFloatTensors(a, b, prealloc); err != nil {
		return errors.Wrapf(err, opFail, "StdEng.MatVecMul")
	}

	m := ad.oshape()[0]
	n := ad.oshape()[1]

	tA := blas.NoTrans
	if ad.oldAP() != nil {
		tA = blas.Trans
	}
	lda := ad.ostrides()[0]
	incX, incY := 1, 1 // step size

	switch A := ad.Data().(type) {
	case []float64:
		x := bd.Float64s()
		y := pd.Float64s()
		alpha, beta := float64(1), float64(0)
		whichblas.Dgemv(tA, m, n, alpha, A, lda, x, incX, beta, y, incY)
	case []float32:
		x := bd.Float32s()
		y := pd.Float32s()
		alpha, beta := float32(1), float32(0)
		whichblas.Sgemv(tA, m, n, alpha, A, lda, x, incX, beta, y, incY)
	default:
		return errors.Errorf(typeNYI, "matVecMul", bd.Data())
	}

	return nil
}

// MatMul is a thin layer over DGEMM.
// DGEMM computes:
//		C = αA * B +  βC
// To prevent needless zeroing out of the slice, we just set β to 0
func (e StdEng) MatMul(a, b, prealloc Tensor) (err error) {
	// check all are DenseTensors
	var ad, bd, pd DenseTensor
	if ad, bd, pd, err = e.checkThreeFloatTensors(a, b, prealloc); err != nil {
		return errors.Wrapf(err, opFail, "StdEng.MatMul")
	}

	tA, tB := blas.NoTrans, blas.NoTrans
	if ad.oldAP() != nil {
		tA = blas.Trans
	}

	adp := ad.(*Dense)
	if adp.IsRowVec() {
		tA = blas.Trans
	}

	if bd.oldAP() != nil {
		tB = blas.Trans
	}

	bdp := bd.(*Dense)
	if bdp.IsRowVec() {
		tB = blas.Trans
	}

	var m, n, k int
	m = ad.Shape()[0]
	k = ad.Shape()[1]
	n = bd.Shape()[1]

	// wrt the strides, we use the original strides, because that's what BLAS needs, instead of calling .Strides()
	lda := ad.ostrides()[0]
	ldb := bd.ostrides()[0]
	ldc := pd.ostrides()[0]

	switch A := ad.Data().(type) {
	case []float64:
		B := bd.Float64s()
		C := pd.Float64s()
		alpha, beta := float64(1), float64(0)
		whichblas.Dgemm(tA, tB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
	case []float32:
		B := bd.Float32s()
		C := pd.Float32s()
		alpha, beta := float32(1), float32(0)
		whichblas.Sgemm(tA, tB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
	default:
		return errors.Errorf(typeNYI, "matVecMul", bd.Data())
	}
	return
}

// Outer is a thin wrapper over S/Dger
func (e StdEng) Outer(a, b, prealloc Tensor) (err error) {
	// check all are DenseTensors
	var ad, bd, pd DenseTensor
	if ad, bd, pd, err = e.checkThreeFloatTensors(a, b, prealloc); err != nil {
		return errors.Wrapf(err, opFail, "StdEng.Outer")
	}

	m := ad.Size()
	n := bd.Size()

	// the stride of a Vector is always going to be [1],
	// incX := t.Strides()[0]
	// incY := other.Strides()[0]
	incX, incY := 1, 1
	lda := pd.Strides()[0]

	switch x := ad.Data().(type) {
	case []float64:
		y := bd.Float64s()
		A := pd.Float64s()
		alpha := float64(1)
		whichblas.Dger(m, n, alpha, x, incX, y, incY, A, lda)
	case []float32:
		y := bd.Float32s()
		A := pd.Float32s()
		alpha := float32(1)
		whichblas.Sger(m, n, alpha, x, incX, y, incY, A, lda)
	default:
		return errors.Errorf(typeNYI, "outer", b.Data())
	}
	return nil
}

/* UNEXPORTED UTILITY FUNCTIONS */

func (e StdEng) checkTwoFloatTensors(a, b Tensor) (ad, bd DenseTensor, err error) {
	if err = e.checkAccessible(a); err != nil {
		return nil, nil, errors.Wrap(err, "checkTwoTensors: a is not accessible")
	}
	if err = e.checkAccessible(b); err != nil {
		return nil, nil, errors.Wrap(err, "checkTwoTensors: a is not accessible")
	}

	if a.Dtype() != b.Dtype() {
		return nil, nil, errors.New("Expected a and b to have the same Dtype")
	}

	if ad, err = getFloatDenseTensor(a); err != nil {
		return nil, nil, errors.Wrap(err, "checkTwoTensors expects a to be be a DenseTensor")
	}
	if bd, err = getFloatDenseTensor(b); err != nil {
		return nil, nil, errors.Wrap(err, "checkTwoTensors expects b to be be a DenseTensor")
	}
	return
}

func (e StdEng) checkThreeFloatTensors(a, b, ret Tensor) (ad, bd, retVal DenseTensor, err error) {
	if err = e.checkAccessible(a); err != nil {
		return nil, nil, nil, errors.Wrap(err, "checkTwoTensors: a is not accessible")
	}
	if err = e.checkAccessible(b); err != nil {
		return nil, nil, nil, errors.Wrap(err, "checkTwoTensors: a is not accessible")
	}
	if err = e.checkAccessible(ret); err != nil {
		return nil, nil, nil, errors.Wrap(err, "checkTwoTensors: ret is not accessible")
	}

	if a.Dtype() != b.Dtype() || b.Dtype() != ret.Dtype() {
		return nil, nil, nil, errors.New("Expected a and b and retVal all to have the same Dtype")
	}

	if ad, err = getFloatDenseTensor(a); err != nil {
		return nil, nil, nil, errors.Wrap(err, "checkTwoTensors expects a to be be a DenseTensor")
	}
	if bd, err = getFloatDenseTensor(b); err != nil {
		return nil, nil, nil, errors.Wrap(err, "checkTwoTensors expects b to be be a DenseTensor")
	}
	if retVal, err = getFloatDenseTensor(ret); err != nil {
		return nil, nil, nil, errors.Wrap(err, "checkTwoTensors expects retVal to be be a DenseTensor")
	}
	return
}
