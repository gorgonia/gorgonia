package tensor

import "github.com/pkg/errors"

// exported API for arithmetics and the stupidly crazy amount of overloaded semantics
// Add performs a pointwise a+b. a and b can either be float64 or Tensor
//
// If both operands are Tensor, shape is checked first.
// Even though the underlying data may have the same size (say (2,2) vs (4,1)), if they have different shapes, it will error out.
//
// If the Unsafe flag is passed in, the data of the first tensor will be overwritten

// Add performs elementwise addition on the Tensor(s). These operations are supported:
//		Add(Tensor, scalar)
//		Add(scalar, Tensor)
//		Add(Tensor, Tensor)
func Add(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	ad, adok := a.(*Dense)
	bd, bdok := b.(*Dense)

	switch {
	case adok && bdok:
		return addDD(ad, bd, opts...)
	case adok && !bdok:
		return addDS(ad, b, opts...)
	case !adok && bdok:
		return addSD(a, bd, opts...)
	}
	panic("Unreachable")
}

// Sub performs elementwise subtraction on the Tensor(s). These operations are supported:
//		Sub(Tensor, scalar)
//		Sub(scalar, Tensor)
//		Sub(Tensor, Tensor)
func Sub(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	ad, adok := a.(*Dense)
	bd, bdok := b.(*Dense)

	switch {
	case adok && bdok:
		return subDD(ad, bd, opts...)
	case adok && !bdok:
		return subDS(ad, b, opts...)
	case !adok && bdok:
		return subSD(a, bd, opts...)
	}
	panic("Unreachable")
}

// Mul performs elementwise multiplication on the Tensor(s). These operations are supported:
//		Mul(Tensor, scalar)
//		Mul(scalar, Tensor)
//		Mul(Tensor, Tensor)
func Mul(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	ad, adok := a.(*Dense)
	bd, bdok := b.(*Dense)

	switch {
	case adok && bdok:
		return mulDD(ad, bd, opts...)
	case adok && !bdok:
		return mulDS(ad, b, opts...)
	case !adok && bdok:
		return mulSD(a, bd, opts...)
	}
	panic("Unreachable")
}

// Div performs elementwise division on the Tensor(s). These operations are supported:
//		Div(Tensor, scalar)
//		Div(scalar, Tensor)
//		Div(Tensor, Tensor)
func Div(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	ad, adok := a.(*Dense)
	bd, bdok := b.(*Dense)

	switch {
	case adok && bdok:
		return divDD(ad, bd, opts...)
	case adok && !bdok:
		return divDS(ad, b, opts...)
	case !adok && bdok:
		return divSD(a, bd, opts...)
	}
	panic("Unreachable")
}

// Pow performs elementwise exponentiation on the Tensor(s). These operations are supported:
//		Pow(Tensor, scalar)
//		Pow(scalar, Tensor)
//		Pow(Tensor, Tensor)
func Pow(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	ad, adok := a.(*Dense)
	bd, bdok := b.(*Dense)

	switch {
	case adok && bdok:
		return powDD(ad, bd, opts...)
	case adok && !bdok:
		return powDS(ad, b, opts...)
	case !adok && bdok:
		return powSD(a, bd, opts...)
	}
	panic("Unreachable")
}

// Dot is a highly opinionated API for performing dot product operations on two *Denses, a and b.
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
func Dot(x, y Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	// sparse tensors will have their own methods to handle Dot
	if xdottir, ok := x.(Dotter); ok {
		return xdottir.Dot(y)
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

	reuseT, incrT := parseReuseIncr(opts...)
	var reuse, incr *Dense

	if reuseT != nil {
		if reuse, err = getFloatDense(reuseT); err != nil {
			err = errors.Wrapf(err, opFail, "Dot - reuse")
			return
		}
	}
	if incrT != nil {
		if incr, err = getFloatDense(incrT); err != nil {
			err = errors.Wrapf(err, opFail, "Dot - incr")
		}
	}

	var af, bf, incrF Float
	af = a.data.(Float)
	bf = a.data.(Float)

	if incr != nil {
		incrF = incr.data.(Float)
	}

	var ok bool
	switch {
	case a.IsScalar() && b.IsScalar():
		var ret Number
		if ret, err = safeMul(af, bf); err != nil {
			err = errors.Wrapf(err, opFail, "Dot - is Scalar")
			return
		}

		switch {
		case incr != nil:
			if !incr.IsScalar() {
				err = errors.Errorf(shapeMismatch, ScalarShape(), incr.Shape())
				return
			}
			if err = af.IncrMul(bf, incrF); err != nil {
				err = errors.Wrapf(err, opFail, "Dot scalar incr")
			}

			retVal = incr
		case reuse != nil:
			if err = reuse.data.Set(0, ret.Get(0)); err != nil {
				err = errors.Wrapf(err, opFail, "Dot - scalar reuse")
				return
			}
			reuse.reshape()
			retVal = reuse
		default:
			retVal = New(FromScalar(ret.Get(0)))
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
	// now the stupid scaling stuff is out of the way, let's do some linear algebra

	if incr != nil {
		defer func() {
			if !retVal.Shape().Eq(incr.Shape()) {
				err = errors.Errorf(shapeMismatch, retVal.Shape(), incr.Shape())
				return
			}
			switch rt := retVal.(type) {
			case *Dense:
				var rn Number
				if rn, ok = rt.data.(Number); !ok {
					err = errors.Errorf(extractionFail)
					return
				}
				incrF.Add(rn)
				retVal = incr
			default:
				err = errors.Errorf(typeNYI, "Dot", retVal)
				return
			}

		}()
	}

	switch {
	case a.IsVector():
		switch {
		case b.IsVector():
			// check size
			if a.Size() != b.Size() {
				err = errors.Errorf(shapeMismatch, a.Shape(), b.Shape())
				return
			}
			return a.inner(b)
		case b.IsMatrix():
			if b.Shape()[0] != a.Size() {
				err = errors.Errorf(shapeMismatch, a.Shape(), b.Shape())
				return
			}

			var rd *Dense
			expectedShape := Shape{b.Shape()[1]}
			if reuse != nil {
				if err = reuseCheckShape(reuse, expectedShape); err != nil {
					return
				}
				rd = reuse

			} else {
				rd = recycledDense(a.t, expectedShape)
			}
			b.T()
			if err = b.matVecMul(a, rd); err != nil {
				err = errors.Wrapf(err, opFail, "Dot - a is vector, b is matrix")
				return
			}
			b.T()

			retVal = rd
			return
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

			// if a.Shape()[1] != b.Size() {
			// 	err = errors.Errorf(shapeMismatch, a.Shape(), b.Shape())
			// 	return
			// }

			// var rd *Dense
			// expectedShape := Shape{a.Shape()[0]}
			// if reuse != nil {
			// 	if err = reuseCheckShape(reuse, expectedShape); err != nil {
			// 		return
			// 	}
			// 	rd = reuse
			// } else {
			// 	rd = recycledDense(a.t, expectedShape)
			// }

			// if err = a.matVecMul(b, rd); err != nil {
			// 	err = errors.Wrapf(err, opFail, "Dot - a is matrix, b is vector")
			// 	return
			// }

			// retVal = rd
			// return
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
			// if a.Shape()[1] != b.Shape()[0] {
			// 	err = errors.Errorf(shapeMismatch, a.Shape(), b.Shape())
			// 	return
			// }
			// var rd *Dense
			// expectedShape := Shape{a.Shape()[0], b.Shape()[1]}
			// if reuse != nil {
			// 	rd = reuse
			// 	// check that retVal has the correct shape
			// 	if !retVal.Shape().Eq(expectedShape) {
			// 		err = errors.Errorf(shapeMismatch, expectedShape, retVal.Shape())
			// 		return
			// 	}
			// } else {
			// 	rd = New(Of(a.t), WithShape(expectedShape...))
			// }
			// a.matMul(b, rd)
			// retVal = rd
			// return
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
		// swap out the underlying data and metadata
		reuse.data, rd.data = rd.data, reuse.data
		reuse.AP, rd.AP = rd.AP, reuse.AP
		defer ReturnTensor(rd)

		retVal = reuse
	} else {
		retVal = rd
	}
	return
}
