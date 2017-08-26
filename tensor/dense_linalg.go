package tensor

import "github.com/pkg/errors"

// Trace returns the trace of the matrix (i.e. the sum of the diagonal elements). It only works for matrices
func (t *Dense) Trace() (retVal interface{}, err error) {
	e := t.e
	if e == nil {
		e = StdEng{}
	}
	if tracer, ok := e.(Tracer); ok {
		return tracer.Trace(t)
	}
	return nil, errors.Errorf("Engine %T does not support Trace", e)
}

// Inner performs a dot product on two vectors. If t or other are not vectors, it will return an error.
func (t *Dense) Inner(other Tensor) (retVal interface{}, err error) {
	// check that the data is a float
	if !isFloat(t.t) {
		return nil, errors.Errorf(unsupportedDtype, t.t, "Inner")
	}

	// check both are vectors
	if !t.Shape().IsVector() || !other.Shape().IsVector() {
		return nil, errors.Errorf("Inner only works when there are two vectors. t's Shape: %v; other's Shape %v", t.Shape(), other.Shape())
	}

	// we do this check instead of the more common t.Shape()[1] != other.Shape()[0],
	// basically to ensure a similarity with numpy's dot and vectors.
	if t.len() != other.DataSize() {
		return nil, errors.Errorf(shapeMismatch, t.Shape(), other.Shape())
	}

	e := t.e
	if e == nil {
		e = StdEng{}
	}

	if ip, ok := e.(InnerProder); ok {
		return ip.Inner(t, other)
	}
	return nil, errors.Errorf("Engine does not support Inner()")
}

// MatVecMul performs a matrix-vector multiplication.
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
	fo := ParseFuncOpts(opts...)
	defer returnOpOpt(fo)
	if retVal, err = handleReuse(fo.Reuse(), expectedShape); err != nil {
		err = errors.Wrapf(err, opFail, "MatVecMul")
		return
	}

	if retVal == nil {
		retVal = recycledDense(t.t, expectedShape)
	}

	e := t.e
	if e == nil {
		e = StdEng{}
	}

	if mvm, ok := e.(MatVecMuler); ok {
		if err = mvm.MatVecMul(t, other, retVal); err != nil {
			return nil, errors.Wrapf(err, opFail, "MatVecMul")
		}
		return handleIncr(retVal, fo.Reuse(), fo.Incr(), expectedShape)
	}
	return nil, errors.New("engine does not support MatVecMul")
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

	fo := ParseFuncOpts(opts...)
	defer returnOpOpt(fo)
	if retVal, err = handleReuse(fo.Reuse(), expectedShape); err != nil {
		err = errors.Wrapf(err, opFail, "MatMul")
		return
	}

	if retVal == nil {
		retVal = recycledDense(t.t, expectedShape)
	}

	e := t.e
	if e == nil {
		e = StdEng{}
	}

	if mm, ok := e.(MatMuler); ok {
		if err = mm.MatMul(t, other, retVal); err != nil {
			return
		}
		return handleIncr(retVal, fo.Reuse(), fo.Incr(), expectedShape)
	}

	return nil, errors.New("engine does not support MatMul")
}

// Outer finds the outer product of two vectors
func (t *Dense) Outer(other Tensor, opts ...FuncOpt) (retVal *Dense, err error) {
	// check both are vectors
	if !t.Shape().IsVector() || !other.Shape().IsVector() {
		err = errors.Errorf("Outer only works when there are two vectors. t's shape: %v. other's shape: %v", t.Shape(), other.Shape())
		return
	}

	m := t.Size()
	n := other.Size()

	// check whether retVal has the same size as the resulting matrix would be: mxn
	expectedShape := Shape{m, n}

	fo := ParseFuncOpts(opts...)
	defer returnOpOpt(fo)
	if retVal, err = handleReuse(fo.Reuse(), expectedShape); err != nil {
		err = errors.Wrapf(err, opFail, "Outer")
		return
	}

	if retVal == nil {
		retVal = recycledDense(t.t, expectedShape)
	}

	e := t.e
	if e == nil {
		e = StdEng{}
	}
	// DGER does not have any beta. So the values have to be zeroed first if the tensor is to be reused
	retVal.Zero()
	if op, ok := e.(OuterProder); ok {
		if err = op.Outer(t, other, retVal); err != nil {
			return nil, errors.Wrapf(err, opFail, "engine.uter")
		}
		return handleIncr(retVal, fo.Reuse(), fo.Incr(), expectedShape)
	}
	return nil, errors.New("engine does not support Outer")
}

// TensorMul is for multiplying Tensors with more than 2 dimensions.
//
// The algorithm is conceptually simple (but tricky to get right):
// 		1. Transpose and reshape the Tensors in such a way that both t and other are 2D matrices
//		2. Use DGEMM to multiply them
//		3. Reshape the results to be the new expected result
//
// This function is a Go implementation of Numpy's tensordot method. It simplifies a lot of what Numpy does.
func (t *Dense) TensorMul(other Tensor, axesA, axesB []int) (retVal *Dense, err error) {
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
		err = errors.Errorf(shapeMismatch, ts, os)
		return
	}

	// handle shapes
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

	newAxesA := BorrowInts(len(notins) + len(axesA))
	defer ReturnInts(newAxesA)
	newAxesA = newAxesA[:0]
	newAxesA = append(notins, axesA...)
	n2 := 1
	for _, a := range axesA {
		n2 *= ts[a]
	}

	newShapeT := Shape(BorrowInts(2))
	defer ReturnInts(newShapeT)
	newShapeT[0] = ts.TotalSize() / n2
	newShapeT[1] = n2

	retShape1 := BorrowInts(len(ts))
	defer ReturnInts(retShape1)
	retShape1 = retShape1[:0]
	for _, ni := range notins {
		retShape1 = append(retShape1, ts[ni])
	}

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

	newAxesB := BorrowInts(len(notins) + len(axesB))
	defer ReturnInts(newAxesB)
	newAxesB = newAxesB[:0]
	newAxesB = append(axesB, notins...)

	newShapeO := Shape(BorrowInts(2))
	defer ReturnInts(newShapeO)
	newShapeO[0] = n2
	newShapeO[1] = os.TotalSize() / n2

	retShape2 := BorrowInts(len(ts))
	retShape2 = retShape2[:0]
	for _, ni := range notins {
		retShape2 = append(retShape2, os[ni])
	}

	// we borrowClone because we don't want to touch the original Tensors
	doT := t.Clone().(*Dense)
	doOther := other.Clone().(*Dense)
	defer ReturnTensor(doT)
	defer ReturnTensor(doOther)

	if err = doT.T(newAxesA...); err != nil {
		return
	}
	doT.Transpose() // we have to materialize the transpose first or the underlying data won't be changed and the reshape that follows would be meaningless

	if err = doT.Reshape(newShapeT...); err != nil {
		return
	}

	if err = doOther.T(newAxesB...); err != nil {
		return
	}
	doOther.Transpose()

	if err = doOther.Reshape(newShapeO...); err != nil {
		return
	}

	// the magic happens here
	var rt Tensor
	if rt, err = Dot(doT, doOther); err != nil {
		return
	}
	retVal = rt.(*Dense)

	retShape := BorrowInts(len(retShape1) + len(retShape2))
	defer ReturnInts(retShape)

	retShape = retShape[:0]
	retShape = append(retShape, retShape1...)
	retShape = append(retShape, retShape2...)

	if err = retVal.Reshape(retShape...); err != nil {
		return
	}

	return
}

// SVD does the Single Value Decomposition for the *Dense.
//
// How it works is it temporarily converts the *Dense into a gonum/mat64 matrix, and uses Gonum's SVD function to perform the SVD.
// In the future, when gonum/lapack fully supports float32, we'll look into rewriting this
func (t *Dense) SVD(uv, full bool) (s, u, v *Dense, err error) {
	e := t.Engine()
	if e == nil {
		e = StdEng{}
	}
	if svder, ok := e.(SVDer); ok {
		var sT, uT, vT Tensor
		if sT, uT, vT, err = svder.SVD(t, uv, full); err != nil {
			return nil, nil, nil, errors.Wrap(err, "Error while performing *Dense.SVD")
		}
		if s, err = assertDense(sT); err != nil {
			return nil, nil, nil, errors.Wrapf(err, "sT is not *Dense (uv %t full %t). Got %T instead", uv, full, sT)
		}
		// if not uv and not full, u can be nil
		if u, err = assertDense(uT); err != nil && !(!uv && !full) {
			return nil, nil, nil, errors.Wrapf(err, "uT is not *Dense (uv %t full %t). Got %T instead", uv, full, uT)
		}
		// if not uv and not full, v can be nil
		if v, err = assertDense(vT); err != nil && !(!uv && !full) {
			return nil, nil, nil, errors.Wrapf(err, "vT is not *Dense (uv %t full %t). Got %T instead", uv, full, vT)
		}
		return s, u, v, nil
	}
	return nil, nil, nil, errors.New("Engine does not support SVD")
}

/* UTILITY FUNCTIONS */

// handleReuse extracts a *Dense from Tensor, and checks the shape of the reuse Tensor
func handleReuse(reuse Tensor, expectedShape Shape) (retVal *Dense, err error) {
	if reuse != nil {
		if retVal, err = assertDense(reuse); err != nil {
			err = errors.Wrapf(err, opFail, "handling reuse")
			return
		}

		if err = reuseCheckShape(retVal, expectedShape); err != nil {
			err = errors.Wrapf(err, "Unable to process reuse *Dense Tensor. Shape error.")
			return
		}
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

		if err = typeclassCheck(incrD.t, numberTypes); err != nil {
			err = errors.Wrapf(err, "handleIncr only handles Number types. Got %v instead", incrD.t)
			return
		}

		if _, err = incrD.Add(res, UseUnsafe()); err != nil {
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
