package tensor

import (
	"github.com/gonum/blas"
	"github.com/pkg/errors"
)

// Trace returns the trace of the matrix (i.e. the sum of the diagonal elements). It only works for matrices
// TODO: support compatible Arrays
func (t *Dense) Trace() (retVal interface{}, err error) {
	if t.Dims() != 2 {
		err = errors.Errorf(dimMismatch, 2, t.Dims())
		// err = NewError(DimensionMismatch, "Trace() only works on matrices (i.e. only 2 dimensions. This has: %d dimensions", t.Dims())
		return
	}

	if _, ok := t.data.(Number); !ok {
		err = noopError{}
		return
	}

	rstride := t.Strides()[0]
	cstride := t.Strides()[1]

	r := t.Shape()[0]
	c := t.Shape()[1]

	m := MinInt(r, c)
	stride := rstride + cstride

	switch data := t.Data().(type) {
	case []float64:
		var trace float64
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
	case []int:
		var trace int
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
	case []int32:
		var trace int32
		for i := 0; i < m; i++ {
			trace += data[i*stride]
		}
		retVal = trace
	case []byte:
		var trace byte
		for i := 0; i < m; i++ {
			trace += data[i*stride]
		}
		retVal = trace
	default:
		if tracer, ok := t.data.(Tracer); ok {
			return tracer.Trace(rstride, cstride, m)
		}
		err = errors.Errorf(unsupportedDtype, t.data, "Trace")
	}
	return
}

// Inner performs a dot product on two vectors. If t or other are not vectors, it will return an error.
// It doesn't matter if the vectors are vertical-vertical (shape that looks like this: (x, 1)), or horizontal-horizontal (shapes that look like this: (1, x))
func (t *Dense) Inner(other Tensor) (retVal *Dense, err error) {
	// check that the data is a float
	if _, ok := t.data.(Float); !ok {
		err = errors.Errorf(unsupportedDtype, t.data, "Inner")
		return
	}

	// check both are vectors
	if !t.Shape().IsVector() || !other.Shape().IsVector() {
		err = errors.Errorf("Inner only works when there are two vectors. t's Shape: %v; other's Shape %v", t.Shape(), other.Shape())
		return
	}

	// we do this check instead of the more common t.Shape()[1] != other.Shape()[0],
	// basically to ensure a similarity with numpy's dot and vectors.
	if t.Size() != other.Size() {
		err = errors.Errorf(shapeMismatch, t.Shape(), other.Shape())
		return
	}

	return t.inner(other)
}

// inner is a thin layer over BLAS's Ddot.
// There is a slight difference in terms of API (you'll note that inner() returns a tensor and error)
// This is because the actual result of a  dot product is a scalar.
func (t *Dense) inner(other Tensor) (retVal *Dense, err error) {
	var ot *Dense
	if ot, err = getFloatDense(other); err != nil {
		err = errors.Wrapf(err, opFail, "inner")
		return
	}

	on := ot.data.(Float)
	switch a := t.Data().(type) {
	case []float64:
		var b []float64
		if b, err = getFloat64s(on); err != nil {
			return
		}

		ret := whichblas.Ddot(t.Size(), a, 1, b, 1)
		retVal = New(FromScalar(ret))
	case []float32:
		var b []float32
		if b, err = getFloat32s(on); err != nil {
			return
		}

		ret := whichblas.Sdot(t.Size(), a, 1, b, 1)
		retVal = New(FromScalar(ret))
	default:
		err = errors.Errorf(typeNYI, "Inner", other.Data())
	}
	return
}

func (t *Dense) MatVecMul(other Tensor, opts ...FuncOpt) (retVal *Dense, err error) {
	// check that it's a matrix x vector
	if t.Dims() != 2 || !other.Shape().IsVector() {
		err = errors.Errorf("MatVecMul requires t be a matrix and other to be a vector. Got t's shape: %v, other's shape: %v", t.Shape(), other.Shape())
		return
	}

	// checks that t is mxn matrix
	m := t.Shape()[0]
	n := t.Shape()[1]

	// check shape
	var odim int
	oshape := other.Shape()
	switch {
	case oshape.IsColVec():
		odim = oshape[0]
	case oshape.IsRowVec():
		odim = oshape[1]
	case oshape.IsVector():
		odim = oshape[0]
	default:
		err = errors.Errorf(shapeMismatch, t.Shape(), other.Shape()) // should be unreachable
		return
	}

	if odim != n {
		err = errors.Errorf(shapeMismatch, n, other.Shape())
		return
	}

	expectedShape := Shape{m}

	// check whether retVal has the same size as the resulting matrix would be: mx1
	reuse, incr := parseReuseIncr(opts...)

	if retVal, err = handleReuse(reuse, expectedShape); err != nil {
		err = errors.Wrapf(err, opFail, "MatVecMul")
		return
	}

	if retVal == nil {
		retVal = recycledDense(t.t, expectedShape)
	}

	var od *Dense
	if od, err = getFloatDense(other); err != nil {
		err = errors.Wrapf(err, typeNYI, "MatVecMul", other)
		return
	}

	if err = t.matVecMul(od, retVal); err != nil {
		return
	}

	return handleIncr(retVal, reuse, incr, expectedShape)
}

// matVecMul is a thin layer over BLAS' DGEMV
// Because DGEMV computes:
// 		y = αA * x + βy
// we set beta to 0, so we don't have to manually zero out the reused/retval tensor data
func (t *Dense) matVecMul(other *Dense, retVal *Dense) (err error) {
	// we use the pre-transpose shpes and strides, because BLAS handles the rest
	m := t.oshape()[0]
	n := t.oshape()[1]

	tA := blas.NoTrans
	if t.old != nil {
		tA = blas.Trans
	}
	lda := t.ostrides()[0]
	incX, incY := 1, 1 // step size

	switch A := t.Data().(type) {
	case []float64:
		var x, y []float64
		var ok bool
		if x, ok = other.Data().([]float64); !ok {
			err = errors.Errorf(dtypeMismatch, A, x)
			return
		}

		if y, ok = retVal.Data().([]float64); !ok {
			err = errors.Errorf(dtypeMismatch, A, y)
			return
		}

		alpha, beta := float64(1), float64(0)
		whichblas.Dgemv(tA, m, n, alpha, A, lda, x, incX, beta, y, incY)
	case []float32:
		var x, y []float32
		var ok bool
		if x, ok = other.Data().([]float32); !ok {
			err = errors.Errorf(dtypeMismatch, A, x)
			return
		}

		if y, ok = retVal.Data().([]float32); !ok {
			err = errors.Errorf(dtypeMismatch, A, y)
			return
		}

		alpha, beta := float32(1), float32(0)
		whichblas.Sgemv(tA, m, n, alpha, A, lda, x, incX, beta, y, incY)
	default:
		return errors.Errorf(typeNYI, "matVecMul", other.Data())
	}

	return nil
}

// MatMul is the basic matrix multiplication that you learned in high school. It takes an optional reuse ndarray, where the ndarray is reused as the result.
// If that isn't passed in,  a new ndarray will be created instead.
func (t *Dense) MatMul(other Tensor, opts ...FuncOpt) (retVal *Dense, err error) {
	// check that both are matrices
	if !t.Shape().IsMatrix() || !other.Shape().IsMatrix() {
		err = errors.Errorf("MatMul requires both operands to be matrices. Got t's shape: %v, other's shape: %v", t.Shape(), other.Shape())
		return
	}

	// checks that t is mxk matrix
	var m, n, k int
	m = t.Shape()[0]
	k = t.Shape()[1]
	n = other.Shape()[1]

	// check shape
	if k != other.Shape()[0] {
		err = errors.Errorf(shapeMismatch, t.Shape(), other.Shape())
		return
	}

	// check whether retVal has the same size as the resulting matrix would be: mxn
	expectedShape := Shape{m, n}

	reuse, incr := parseReuseIncr(opts...)
	if retVal, err = handleReuse(reuse, expectedShape); err != nil {
		err = errors.Wrapf(err, opFail, "MatMul")
		return
	}

	if retVal == nil {
		retVal = recycledDense(t.t, expectedShape)
	}

	var od *Dense
	if od, err = getFloatDense(other); err != nil {
		err = errors.Wrapf(err, typeNYI, "MatMul", other)
		return
	}

	if err = t.matMul(od, retVal); err != nil {
		return
	}

	return handleIncr(retVal, reuse, incr, expectedShape)
}

// matMul is a thin layer over DGEMM.
// DGEMM computes:
//		C = αA * B +  βC
// To prevent needless zeroing out of the slice, we just set β to 0
func (t *Dense) matMul(other, retVal *Dense) (err error) {
	tA, tB := blas.NoTrans, blas.NoTrans
	if t.old != nil {
		tA = blas.Trans
	}

	if other.old != nil {
		tB = blas.Trans
	}

	var m, n, k int
	m = t.Shape()[0]
	k = t.Shape()[1]
	n = other.Shape()[1]

	// wrt the strides, we use the original strides, because that's what BLAS needs, instead of calling .Strides()
	lda := t.ostrides()[0]
	ldb := other.ostrides()[0]
	ldc := retVal.ostrides()[0]

	var ok bool
	switch a := t.Data().(type) {
	case []float64:
		var b, c []float64
		if b, ok = other.Data().([]float64); !ok {
			err = errors.Errorf(dtypeMismatch, a, b)
			return
		}

		if c, ok = retVal.Data().([]float64); !ok {
			err = errors.Errorf(dtypeMismatch, a, c)
			return
		}

		alpha, beta := float64(1), float64(0)
		whichblas.Dgemm(tA, tB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
	case []float32:
		var b, c []float32
		if b, ok = other.Data().([]float32); !ok {
			err = errors.Errorf(dtypeMismatch, a, b)
			return
		}

		if c, ok = retVal.Data().([]float32); !ok {
			err = errors.Errorf(dtypeMismatch, a, c)
			return
		}

		alpha, beta := float32(1), float32(0)
		whichblas.Sgemm(tA, tB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
	default:
		return errors.Errorf(typeNYI, "matVecMul", other.Data())
	}
	return
}

/* UTILITY FUNCTIONS */

// getFloatDense extracts a *Dense from a Tensor and ensures that the .data is a Array that implements Float
func getFloatDense(a Tensor) (retVal *Dense, err error) {
	switch at := a.(type) {
	case *Dense:
		if f, ok := at.data.(Float); !ok {
			err = errors.Errorf(dtypeMismatch, f, at.data)
			return
		}
		return at, nil
	default:
		err = errors.Errorf(extractionFail, "*Dense", a)
		return
	}
	panic("unreachable")
}

// handleReuse extracts a *Dense from Tensor, and checks the shape of the reuse Tensor
func handleReuse(reuse Tensor, expectedShape Shape) (retVal *Dense, err error) {
	if reuse != nil {
		var rd *Dense
		var ok bool
		if rd, ok = reuse.(*Dense); !ok {
			err = errors.Errorf(extractionFail, "*Dense", reuse)
			return
		}
		if err = reuseCheckShape(rd, expectedShape); err != nil {
			err = errors.Wrapf(err, "Unable to process reuse *Dense Tensor. Shape error.")
			return
		}
		retVal = rd
		return
	}
	return
}

// handleIncr is the cleanup step for when there is an Tensor to increment. If the result tensor is the same as the reuse Tensor, the result tensor gets returned to the pool
func handleIncr(res *Dense, reuse, incr Tensor, expectedShape Shape) (retVal *Dense, err error) {
	// handle increments
	if incr != nil {
		if !expectedShape.Eq(incr.Shape()) {
			err = errors.Errorf(shapeMismatch, expectedShape, incr.Shape())
			return
		}
		var incrD *Dense
		var ok bool
		if incrD, ok = incr.(*Dense); !ok {
			err = errors.Errorf(extractionFail, "*Dense", incr)
			return
		}

		var incrN Number
		if incrN, ok = incrD.data.(Number); !ok {
			err = errors.Errorf(unsupportedDtype, incrD.data, "MatVecMul as incr")
			return
		}

		if err = incrN.Add(res.data.(Number)); err != nil {
			return
		}
		// vecAdd(incr.data, retVal.data)

		// return retVal to pool - if and only if retVal is not reuse
		// reuse indicates that someone else also has the reference to the *Dense
		if res != reuse {
			ReturnTensor(res)
		}

		// then
		retVal = incrD
		return
	}

	return res, nil
}
