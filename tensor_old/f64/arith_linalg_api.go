package tensorf64

import "github.com/chewxy/gorgonia/tensor/types"

// Dot is a highly opinionated API for performing dot product operations on two *Tensors, a and b.
// This function is opinionated with regard to the vector operations because of how it treats operations with vectors.
// Vectors in this package comes in two flavours - column or row vectors. Column vectors have shape (x, 1), while row vectors have shape (1, x).
//
// As such, it is easy to assume that performing a linalg operation on vectors would follow the same rules (i.e shapes have to be aligned for things to work).
// For the most part in this package, this is true. This function is one of the few notable exceptions.
//
// Here I give three specific examples of how the expectations of vector operations will differ.
// 		Given two vectors, a, b with shapes (4, 1) and (4, 1), Dot() will perform an inner product as if the shapes were (1, 4) and (4, 1). This will result in a scalar value
// 		Given matrix A and vector b with shapes (2, 4) and (1, 4), Dot() will perform a matrix-vector multiplication as if the shapes were (2,4) and (4,1). This will result in a column vector with shape (2,1)
//		Given vector a and matrix B with shapes (3, 1) and (3, 2), Dot() will perform a matrix-vector multiplication as if it were Bᵀ * a
//
// The main reason why this opinionated route was taken was due to the author's familiarity with NumPy, and general laziness in translating existing machine learning algorithms
// to fit the API of the package.
func Dot(a, b *Tensor, opts ...types.FuncOpt) (retVal *Tensor, err error) {
	// safe, incr, reuse := parseSafeReuse(opts...)
	reuse, incr := parseReuseIncr(opts...)

	switch {
	case a.IsScalar() && b.IsScalar():
		res := a.data[0] * b.data[0]
		switch {
		case incr != nil:
			if !incr.IsScalar() {
				err = shapeMismatchError(types.ScalarShape(), incr.Shape())
				return
			}
			incr.data[0] += res
			retVal = incr
		case reuse != nil:
			retVal = reuse
			retVal.reshape()
			retVal.data[0] = res
		default:
			retVal = NewTensor(AsScalar(res))
		}
		return
	case a.IsScalar():
		switch {
		case incr != nil:
			return PointwiseMul(a.ScalarValue(), b, types.WithIncr(incr))
		case reuse != nil:
			return PointwiseMul(a.ScalarValue(), b, types.WithReuse(reuse))
		}
		// default moved out
		return PointwiseMul(a.ScalarValue(), b)
	case b.IsScalar():
		switch {
		case incr != nil:
			return PointwiseMul(a, b.ScalarValue(), types.WithIncr(incr))
		case reuse != nil:
			return PointwiseMul(a, b.ScalarValue(), types.WithReuse(reuse))
		}
		return PointwiseMul(a, b.ScalarValue())
	}
	// now the stupid scaling stuff is out of the way, let's do some linear algebra

	if incr != nil {
		defer func() {
			if !retVal.Shape().Eq(incr.Shape()) {
				err = shapeMismatchError(retVal.Shape(), incr.Shape())
				return
			}
			vecAdd(incr.data, retVal.data)
			retVal = incr
		}()
	}

	switch {
	case a.IsVector():
		switch {
		case b.IsVector():
			// check size
			if a.Size() != b.Size() {
				err = shapeMismatchError(a.Shape(), b.Shape())
				return
			}
			return a.inner(b)
		case b.IsMatrix():
			if b.Shape()[0] != a.Size() {
				err = shapeMismatchError(a.Shape(), b.Shape())
				return
			}

			expectedShape := types.Shape{b.Shape()[1]}
			if reuse != nil {
				if err = reuseCheckShape(reuse, expectedShape); err != nil {
					return
				}
				retVal = reuse

			} else {
				retVal = newBorrowedTensor(expectedShape.TotalSize(), WithShape(expectedShape...))
			}
			b.T()
			b.matVecMul(a, retVal)
			b.T()
			return
		default:

		}
	case a.IsMatrix():
		switch {
		case b.IsVector():
			if a.Shape()[1] != b.Size() {
				err = shapeMismatchError(a.Shape(), b.Shape())
				return
			}
			expectedShape := types.Shape{a.Shape()[0]}
			if reuse != nil {
				if err = reuseCheckShape(reuse, expectedShape); err != nil {
					return
				}
				retVal = reuse
			} else {
				retVal = newBorrowedTensor(expectedShape.TotalSize(), WithShape(expectedShape...))
			}
			a.matVecMul(b, retVal)
			return
		case b.IsMatrix():
			if a.Shape()[1] != b.Shape()[0] {
				err = shapeMismatchError(a.Shape(), b.Shape())
				return
			}
			expectedShape := types.Shape{a.Shape()[0], b.Shape()[1]}
			if reuse != nil {
				retVal = reuse
				// check that retVal has the correct shape
				if !retVal.Shape().Eq(expectedShape) {
					err = shapeMismatchError(expectedShape, retVal.Shape())
				}
			} else {
				retVal = NewTensor(WithShape(expectedShape...))
			}

			a.matMul(b, retVal)
			return
		default:
		}
	default:
	}
	as := a.Shape()
	bs := b.Shape()
	axesA := types.BorrowInts(1)
	axesB := types.BorrowInts(1)
	defer types.ReturnInts(axesA)
	defer types.ReturnInts(axesB)

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
		err = shapeMismatchError(as, bs)
		return
	}

	if retVal, err = a.TensorMul(b, axesA, axesB); err != nil {
		return
	}

	if reuse != nil {
		// swap out the underlying data and metadata
		reuse.data, retVal.data = retVal.data, reuse.data
		reuse.AP, retVal.AP = retVal.AP, reuse.AP
		defer ReturnTensor(retVal)

		retVal = reuse
	}
	return
}
