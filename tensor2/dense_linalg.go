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

	switch data := t.data.(type) {
	case f64s:
		var trace float64
		for i := 0; i < m; i++ {
			trace += data[i*(rstride+cstride)]
		}
		retVal = trace

	case f32s:
		var trace float32
		for i := 0; i < m; i++ {
			trace += data[i*(rstride+cstride)]
		}
		retVal = trace
	case ints:
		var trace int
		for i := 0; i < m; i++ {
			trace += data[i*(rstride+cstride)]
		}
		retVal = trace
	case i64s:
		var trace int64
		for i := 0; i < m; i++ {
			trace += data[i*(rstride+cstride)]
		}
		retVal = trace
	case i32s:
		var trace int32
		for i := 0; i < m; i++ {
			trace += data[i*(rstride+cstride)]
		}
		retVal = trace
	case u8s:
		var trace byte
		for i := 0; i < m; i++ {
			trace += data[i*(rstride+cstride)]
		}
		retVal = trace
	default:
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
	ot, ok := other.(*Dense)
	if !ok {
		err = errors.Errorf(typeNYI, "inner", other)
		return
	}

	on, ok := ot.data.(Float)
	if !ok {
		err = errors.Errorf(unsupportedDtype, ot.data, "Inner")
		return
	}

	switch data := t.data.(type) {
	case f64s:
		a := []float64(data)
		var b []float64
		if b, err = getFloat64s(on); err != nil {
			return
		}

		ret := whichblas.Ddot(t.Size(), a, 1, b, 1)
		retVal = New(FromScalar(ret))
	case f32s:
		a := []float32(data)
		var b []float32
		if b, err = getFloat32s(on); err != nil {
			return
		}

		ret := whichblas.Sdot(t.Size(), a, 1, b, 1)
		retVal = New(FromScalar(ret))
	case Float32ser:
		a := data.Float32s()
		var b []float32
		if b, err = getFloat32s(on); err != nil {
			return
		}

		ret := whichblas.Sdot(t.Size(), a, 1, b, 1)
		retVal = New(FromScalar(ret))
	case Float64ser:
		a := data.Float64s()
		var b []float64
		if b, err = getFloat64s(on); err != nil {
			return
		}

		ret := whichblas.Ddot(t.Size(), a, 1, b, 1)
		retVal = New(FromScalar(ret))
	default:
		panic("Unreachable")
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
	if reuse != nil {
		var rd *Dense
		var ok bool
		if rd, ok = reuse.(*Dense); !ok {
			err = errors.Errorf(typeNYI, "MatVecMul", reuse)
		}
		if err = reuseCheckShape(rd, expectedShape); err != nil {
			return
		}
		retVal = rd
	}

	if retVal == nil {
		retVal = recycledDense(t.t, expectedShape)
	}

	switch ot := other.(type) {
	case *Dense:
		if _, ok := ot.data.(Float); !ok {
			err = errors.Errorf(unsupportedDtype, ot.data, "MatVecMul")
			return
		}

		if err = t.matVecMul(ot, retVal); err != nil {
			return
		}
	default:
		err = errors.Errorf(typeNYI, "MatVecMul", other)
		return
	}

	// handle increments
	if incr != nil {
		if !expectedShape.Eq(incr.Shape()) {
			err = errors.Errorf(shapeMismatch, expectedShape, incr.Shape())
			return
		}
		var incrD *Dense
		var ok bool
		if incrD, ok = incr.(*Dense); !ok {
			err = errors.Errorf(typeNYI, "MatVecMul", reuse)
			return
		}

		var incrN Number
		if incrN, ok = incrD.data.(Number); !ok {
			err = errors.Errorf(unsupportedDtype, incrD.data, "MatVecMul as incr")
			return
		}

		incrN.Add(retVal.data.(Number))
		// vecAdd(incr.data, retVal.data)

		// return retVal to pool - if and only if retVal is not reuse
		// reuse indicates that someone else also has the reference to the *Tensor
		if retVal != reuse {
			ReturnTensor(retVal)
		}

		// then
		retVal = incrD
	}
	return
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

	switch data := t.data.(type) {
	case f64s:
		A := []float64(data)
		var x, y []float64
		var ok bool
		if x, ok = other.Data().([]float64); !ok {
			err = errors.Errorf(dtypeMismatch, data, x)
			return
		}

		if y, ok = retVal.Data().([]float64); !ok {
			err = errors.Errorf(dtypeMismatch, data, y)
			return
		}

		alpha, beta := float64(1), float64(0)
		whichblas.Dgemv(tA, m, n, alpha, A, lda, x, incX, beta, y, incY)
	case f32s:
		A := []float32(data)
		var x, y []float32
		var ok bool
		if x, ok = other.Data().([]float32); !ok {
			err = errors.Errorf(dtypeMismatch, data, x)
			return
		}

		if y, ok = retVal.Data().([]float32); !ok {
			err = errors.Errorf(dtypeMismatch, data, y)
			return
		}

		alpha, beta := float32(1), float32(0)
		whichblas.Sgemv(tA, m, n, alpha, A, lda, x, incX, beta, y, incY)
	case Float64ser:
		A := data.Float64s()
		var x, y []float64
		var ok bool
		if x, ok = other.Data().([]float64); !ok {
			err = errors.Errorf(dtypeMismatch, data, x)
			return
		}

		if y, ok = retVal.Data().([]float64); !ok {
			err = errors.Errorf(dtypeMismatch, data, y)
			return
		}

		alpha, beta := float64(1), float64(0)
		whichblas.Dgemv(tA, m, n, alpha, A, lda, x, incX, beta, y, incY)
	case Float32ser:
		A := data.Float32s()
		var x, y []float32
		var ok bool
		if x, ok = other.Data().([]float32); !ok {
			err = errors.Errorf(dtypeMismatch, data, x)
			return
		}

		if y, ok = retVal.Data().([]float32); !ok {
			err = errors.Errorf(dtypeMismatch, t, y)
			return
		}

		alpha, beta := float32(1), float32(0)
		whichblas.Sgemv(tA, m, n, alpha, A, lda, x, incX, beta, y, incY)
	default:
		panic("Unreachable")
	}

	return nil
}
