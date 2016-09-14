package tensorf32

import (
	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/pkg/errors"
)

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

	if incr != nil {
		defer func() {
			if incr.Size() != retVal.Size() {
				err = shapeMismatchError(retVal.Shape(), incr.Shape())
				return
			}

			if err = incr.Reshape(retVal.Shape()...); err != nil {
				err = errors.Wrapf(err, incrReshapeErr, retVal.Shape(), reuse.DataSize())
				return
			}

			vecAdd(incr.data, retVal.data)

			// release retVal

			retVal = incr
		}()
	}

	if a.IsVector() {
		if b.IsVector() {
			// check size
			if a.Size() != b.Size() {
				err = shapeMismatchError(a.Shape(), b.Shape())
				return
			}
			return a.inner(b)
		} else if b.Dims() == 2 {
			if b.Shape()[0] != a.Size() {
				err = shapeMismatchError(a.Shape(), b.Shape())
				return
			}

			expectedShape := types.Shape{b.Shape()[1], 1}
			if reuse != nil {
				retVal = reuse
				// check that retVal has the correct shape
				if !retVal.Shape().Eq(expectedShape) {
					err = shapeMismatchError(expectedShape, retVal.Shape())
				}
			} else {
				retVal = NewTensor(WithShape(expectedShape...))
			}

			b.T()
			b.matVecMul(a, retVal)
			return
		}
	}

	if a.Dims() == 2 {
		as := a.Shape()
		bs := b.Shape()
		ar, ac := as[0], as[1]
		br, bc := bs[0], bs[1]

		if ac != br && !b.IsVector() {
			err = shapeMismatchError(a.Shape(), b.Shape())
			return
		} else if b.IsVector() && ac != br {
			if err = b.T(); err != nil {
				err = errors.Wrap(err, "Error while transposing")
				return
			}
			bs = b.Shape()
			br, bc = bs[0], bs[1]
		}

		expectedShape := types.Shape{ar, bc}
		if reuse != nil {
			retVal = reuse
			// check that retVal has the correct shape
			if !retVal.Shape().Eq(expectedShape) {
				err = shapeMismatchError(expectedShape, retVal.Shape())
			}
		} else {
			retVal = NewTensor(WithShape(expectedShape...))
		}

		if b.IsVector() {
			a.matVecMul(b, retVal)
		} else if b.Dims() == 2 {
			a.matMul(b, retVal)
		}
		return
	}
	// TODO: redoing using TensorDot. SumProd is as of now broken
	return nil, notyetimplemented("sum prod not yet implemented")
}
