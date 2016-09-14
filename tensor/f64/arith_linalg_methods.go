package tensorf64

import (
	"log"

	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/gonum/blas"
	"github.com/pkg/errors"
)

/*
The methods in this file are methods of the *Tensor struct. The file is laid out as such:
	func ExportedMethod1(){}
	func unexportedMethod1(){}

	func ExportedMethod2(){}
	func unexportedMethod2(){}

The general pattern is that ExportedMethod performs all the checks before calling unexportedMethod.
*/

// Trace returns the trace of the matrix (i.e. the sum of the diagonal elements). It only works for matrices
func (t *Tensor) Trace() (retVal float64, err error) {
	if t.Dims() != 2 {
		err = types.NewError(types.DimensionMismatch, "Trace() only works on matrices (i.e. only 2 dimensions. This has: %d dimensions", t.Dims())
		return
	}

	rstride := t.Strides()[0]
	cstride := t.Strides()[1]

	r := t.Shape()[0]
	c := t.Shape()[1]

	m := types.MinInt(r, c)

	for i := 0; i < m; i++ {
		retVal += t.data[i*(rstride+cstride)]
	}
	return
}

// DotProd performs a dot product on two vectors. If t or other are not vectors, it will return an error.
// It doesn't matter if the vectors are vertical-vertical (shape that looks like this: (x, 1)), or horizontal-horizontal (shapes that look like this: (1, x))
func (t *Tensor) Inner(other *Tensor) (retVal *Tensor, err error) {
	// check both are vectors
	if !t.IsVector() || !other.IsVector() {
		err = types.NewError(types.OpError, "DotProd only works when there are two vectors. t has %v; other has %v", t.Shape(), other.Shape())
		return
	}

	// we do this check instead of the more common t.Shape()[1] != other.Shape()[0],
	// basically to ensure a similarity with numpy's dot and vectors.
	if t.Size() != other.Size() {
		err = shapeMismatchError(t.Shape(), other.Shape())
		return
	}

	return t.inner(other)
}

// dotProd is a thin layer over BLAS's Ddot.
// There is a slight difference in terms of API (you'll note that dotProd() returns a tensor and error)
// This is because the actual result of a  dot product is a scalar float64.
func (t *Tensor) inner(other *Tensor) (retVal *Tensor, err error) {
	ret := whichblas.Ddot(t.Size(), t.data, 1, other.data, 1)
	retVal = NewTensor(AsScalar(ret))
	return
}

// MatVecMul multiplies a matrix and a vector together. t must be a Matrix, and other must be a vector otherwise it will error out
func (t *Tensor) MatVecMul(other *Tensor, opts ...types.FuncOpt) (retVal *Tensor, err error) {
	// check that it's a matrix x vector
	if t.Dims() != 2 || !other.IsVector() {
		err = types.NewError(types.OpError, "matVecMul requires t to be a matrix, and other to be a vector. Got %v and %v instead", t.Shape(), other.Shape())
		return
	}

	// checks that t is mxn matrix
	m := t.Shape()[0]
	n := t.Shape()[1]

	// check shape
	if !other.Shape().IsVector() || (other.Shape().IsColVec() && n != other.Shape()[0]) || (other.Shape().IsRowVec() && n != other.Shape()[1]) {
		err = shapeMismatchError(t.Shape(), other.Shape())
		return
	}

	expectedShape := types.Shape{m}

	// check whether retVal has the same size as the resulting matrix would be: mx1
	reuse, incr := parseReuseIncr(opts...)
	if reuse != nil {
		if reuse.Size() != m {
			err = shapeMismatchError(expectedShape, reuse.Shape())
			return
		}
		if err = reuse.Reshape(m); err != nil {
			err = errors.Wrapf(err, reuseReshapeErr, expectedShape, reuse.DataSize())
			return
		}
		retVal = reuse
	}

	if retVal == nil {
		retVal = BorrowTensor(m)
		if err = retVal.Reshape(expectedShape...); err != nil {
			err = errors.Wrapf(err, retValReshapeErr, expectedShape, retVal.DataSize())
			return
		}
		// retVal = NewTensor(WithShape(m, 1))
	}

	t.matVecMul(other, retVal)

	// handle increments
	if incr != nil {
		if incr.Size() != m {
			err = shapeMismatchError(expectedShape, incr.Shape())
			return
		}
		if err = incr.Reshape(expectedShape...); err != nil {
			err = errors.Wrapf(err, incrReshapeErr, expectedShape, reuse.DataSize())
			return
		}

		vecAdd(incr.data, retVal.data)

		// return retVal to pool - if and only if retVal is not reuse
		// reuse indicates that someone else also has the reference to the *Tensor
		if retVal != reuse {
			ReturnTensor(retVal)
		}

		// then
		retVal = incr
	}
	return
}

// matVecMul is a thin layer over BLAS' DGEMV
// Because DGEMV computes:
// 		y = αA * x + βy
// we set beta to 0, so we don't have to manually zero out the reused/retval tensor data
func (t *Tensor) matVecMul(other *Tensor, retVal *Tensor) {
	// we use the pre-transpose shpes and strides, because BLAS handles the rest
	m := t.oshape()[0]
	n := t.oshape()[1]

	tA := blas.NoTrans
	if t.old != nil {
		tA = blas.Trans
	}

	A := t.data
	x := other.data
	y := retVal.data
	lda := t.ostrides()[0]

	alpha, beta := float64(1), float64(0)
	incX, incY := 1, 1 // step size
	whichblas.Dgemv(tA, m, n, alpha, A, lda, x, incX, beta, y, incY)
	return
}

// MatMul is the basic matrix multiplication that you learned in high school. It takes an optional reuse ndarray, where the ndarray is reused as the result.
// If that isn't passed in,  a new ndarray will be created instead.
func (t *Tensor) MatMul(other *Tensor, opts ...types.FuncOpt) (retVal *Tensor, err error) {
	// check that both are matrices
	if t.Dims() != 2 || other.Dims() != 2 {
		err = types.NewError(types.OpError, "MatMul only works when there are two matrices. t has %v; other has %v", t.Shape(), other.Shape())
		return
	}

	// checks that t is mxk matrix
	var m, n, k int
	m = t.Shape()[0]
	k = t.Shape()[1]
	n = other.Shape()[1]

	// check shape
	if k != other.Shape()[0] {
		err = shapeMismatchError(t.Shape(), other.Shape())
		return
	}

	// check whether retVal has the same size as the resulting matrix would be: mxn
	expectedShape := types.Shape{m, n}
	expectedSize := expectedShape.TotalSize()

	reuse, incr := parseReuseIncr(opts...)
	if reuse != nil {
		if reuse.Size() != expectedSize {
			err = shapeMismatchError(expectedShape, reuse.Shape())
			return
		}
		if err = reuse.Reshape(m, n); err != nil {
			err = errors.Wrapf(err, reuseReshapeErr, expectedShape, reuse.DataSize())
			return
		}
		retVal = reuse
	}

	if retVal == nil {
		retVal = BorrowTensor(expectedSize)
		if err = retVal.Reshape(m, n); err != nil {
			err = errors.Wrapf(err, retValReshapeErr, expectedShape, retVal.DataSize())
			return
		}

		// retVal = NewTensor(WithShape(m, n))
	}

	t.matMul(other, retVal)

	// handle increments
	if incr != nil {
		if incr.Size() != expectedSize {
			err = shapeMismatchError(types.Shape{m, n}, incr.Shape())
			return
		}
		if err = incr.Reshape(m, n); err != nil {
			err = errors.Wrapf(err, incrReshapeErr, expectedShape, incr.DataSize())
			return
		}

		vecAdd(incr.data, retVal.data)

		// return retVal to pool - if and only if retVal is not reuse
		// reuse indicates that someone else also has the reference to the *Tensor
		if retVal != reuse {
			ReturnTensor(retVal)
		}

		// then
		retVal = incr
	}
	return
}

// matMul is a thin layer over DGEMM.
// DGEMM computes:
//		C = αA * B +  βC
// To prevent needless zeroing out of the slice, we just set β to 0
func (t *Tensor) matMul(other, retVal *Tensor) {
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

	a := t.data
	b := other.data
	c := retVal.data

	// wrt the strides, we use the original strides, because that's what BLAS needs, instead of calling .Strides()
	lda := t.ostrides()[0]
	ldb := other.ostrides()[0]
	ldc := retVal.ostrides()[0]

	alpha, beta := float64(1), float64(0)
	whichblas.Dgemm(tA, tB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
	return
}

func (t *Tensor) Outer(other *Tensor, opts ...types.FuncOpt) (retVal *Tensor, err error) {
	// check both are vectors
	if !t.IsVector() || !other.IsVector() {
		err = types.NewError(types.OpError, "DotProd only works when there are two vectors. t has %v; other has %v", t.Shape(), other.Shape())
		return
	}

	m := t.Size()
	n := other.Size()

	// check whether retVal has the same size as the resulting matrix would be: mxn
	expectedShape := types.Shape{m, n}
	expectedSize := expectedShape.TotalSize()

	reuse, incr := parseReuseIncr(opts...)
	if reuse != nil {
		if reuse.Size() != expectedSize {
			err = shapeMismatchError(types.Shape{m, n}, reuse.Shape())
			return
		}
		if err = reuse.Reshape(m, n); err != nil {
			err = errors.Wrapf(err, reuseReshapeErr, expectedShape, reuse.DataSize())
			return
		}

		retVal = reuse
	}

	if retVal == nil {
		retVal = BorrowTensor(expectedSize)
		if err = retVal.Reshape(m, n); err != nil {
			err = errors.Wrapf(err, retValReshapeErr, expectedShape, retVal.DataSize())
			return
		}
		// retVal = NewTensor(WithShape(m, n))
	}
	// DGER does not have any beta. So the values have to be zeroed first if the tensor is to be reused
	zeroAll(retVal.data)
	t.outer(other, retVal)

	// handle increments
	if incr != nil {
		if incr.Size() != expectedSize {
			err = shapeMismatchError(types.Shape{m, n}, incr.Shape())
			return
		}
		if err = incr.Reshape(m, n); err != nil {
			err = errors.Wrapf(err, incrReshapeErr, expectedShape, incr.DataSize())
			return
		}

		vecAdd(incr.data, retVal.data)

		// return retVal to pool - if and only if retVal is not reuse
		// reuse indicates that someone else also has the reference to the *Tensor
		if retVal != reuse {
			ReturnTensor(retVal)
		}

		// then
		retVal = incr
	}
	return
}

func (t *Tensor) outer(other, retVal *Tensor) {
	m := t.Size()
	n := other.Size()

	x := t.data
	y := other.data
	A := retVal.data

	alpha := float64(1)
	incX, incY := 1, 1
	// the stride of a Vector is always going to be [1], so above
	// incX := t.Strides()[0]
	// incY := other.Strides()[0]
	lda := retVal.Strides()[0]

	whichblas.Dger(m, n, alpha, x, incX, y, incY, A, lda)
	return
}

// TensorMul is for multiplying Tensors with more than 2 dimensions. It exploits a trick in D/SGEMM, and reshaping using the fortran order
// BROKEN. DO NOT USE UNTIL FIXED.
func (t *Tensor) TensorMul(other *Tensor, axesA, axesB []int) (retVal *Tensor, err error) {
	ts := t.Shape()
	td := len(ts)

	os := other.Shape()
	od := len(os)

	na := len(axesA)
	nb := len(axesB)
	sameLength := na == nb
	if sameLength {
		for i := 0; i < na; i++ {
			if ts[axesA[i]] != os[axesB[i]] {
				sameLength = false
				break
			}
			if axesA[i] < 0 {
				axesA[i] += td
			}

			if axesB[i] < 0 {
				axesB[i] += od
			}
		}
	}
	if !sameLength {
		err = shapeMismatchError(ts, os)
		return
	}

	var notins []int
	for i := 0; i < td; i++ {
		notin := true
		for _, a := range axesA {
			if i == a {
				notin = false
				break
			}
		}
		if notin {
			notins = append(notins, i)
		}
	}

	newAxesA := types.BorrowInts(len(notins) + len(axesA))
	defer types.ReturnInts(newAxesA)
	newAxesA = newAxesA[:0]
	newAxesA = append(notins, axesA...)
	n2 := 1
	for _, a := range axesA {
		n2 *= ts[a]
	}

	newShapeT := types.Shape(types.BorrowInts(2))
	defer types.ReturnInts(newShapeT)
	newShapeT[0] = ts.TotalSize() / n2
	newShapeT[1] = n2

	retShape1 := types.BorrowInts(len(ts))
	defer types.ReturnInts(retShape1)
	retShape1 = retShape1[:0]
	log.Printf("len(retShape %d", len(retShape1))
	for _, ni := range notins {
		retShape1 = append(retShape1, ts[ni])
	}
	log.Printf("NewAxesA: %v | notins: %v | newShapeT: %v | retShape:%v", newAxesA, notins, newShapeT, retShape1)

	// work on other now
	notins = notins[:0]
	for i := 0; i < od; i++ {
		notin := true
		for _, a := range axesB {
			if i == a {
				notin = false
				break
			}
		}
		if notin {
			notins = append(notins, i)
		}
	}
	newAxesB := types.BorrowInts(len(notins) + len(axesB))
	defer types.ReturnInts(newAxesB)
	newAxesB = newAxesB[:0]
	newAxesB = append(notins, axesB...)

	newShapeO := types.Shape(types.BorrowInts(2))
	defer types.ReturnInts(newShapeO)
	newShapeO[0] = n2
	newShapeO[1] = os.TotalSize() / n2

	retShape2 := types.BorrowInts(len(ts))
	retShape2 = retShape2[:0]
	for _, ni := range notins {
		retShape2 = append(retShape2, os[ni])
	}
	log.Printf("NewAxesB: %v | notins: %v | newShapeO: %v | retShape:%v", newAxesB, notins, newShapeO, retShape2)

	if err = t.T(newAxesA...); err != nil {
		return
	}
	log.Printf("T.strides: %v", t.Strides())
	log.Printf("%v", t)

	if err = t.Reshape(newShapeT...); err != nil {
		return
	}

	if err = other.T(newAxesB...); err != nil {
		return
	}

	if err = other.Reshape(newShapeO...); err != nil {
		return
	}

	if retVal, err = t.MatMul(other); err != nil {
		return
	}

	retShape := types.BorrowInts(len(retShape1) + len(retShape2))
	defer types.ReturnInts(retShape)
	retShape = retShape[:0]
	retShape = append(retShape, retShape1...)
	retShape = append(retShape, retShape2...)

	if err = retVal.Reshape(retShape...); err != nil {
		return
	}

	// now reset everything
	types.ReturnAP(t.AP)
	t.AP = t.old
	t.old = nil

	types.ReturnAP(other.AP)
	other.AP = other.old
	other.old = nil

	return
}
