package tensor

import (
	"log"
	"reflect"

	"github.com/pkg/errors"
)

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
		return ad.Add(bd, opts...)
	case adok && !bdok:
		return ad.Trans(b, opts...)
	case !adok && bdok:
		return bd.Trans(a, opts...)
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
		return ad.Sub(bd, opts...)
	case adok && !bdok:
		return ad.TransInv(b, opts...)
	case !adok && bdok:
		return bd.TransInvR(a, opts...)
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
		return ad.Mul(bd, opts...)
	case adok && !bdok:
		log.Printf("TransInv: a: %v, b %v", a, b)
		defer log.Printf("a %v", retVal)
		return ad.Scale(b, opts...)
	case !adok && bdok:
		return bd.Scale(a, opts...)
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
		return ad.Div(bd, opts...)
	case adok && !bdok:
		return ad.ScaleInv(b, opts...)
	case !adok && bdok:
		return bd.ScaleInvR(a, opts...)
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
		return ad.Pow(bd, opts...)
	case adok && !bdok:
		return ad.PowOf(b, opts...)
	case !adok && bdok:
		return bd.PowOfR(a, opts...)
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

	fo := parseFuncOpts(opts...)

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
			res = a.getF64(0) * b.getF64(0)
		case reflect.Float32:
			res = a.getF32(0) * b.getF32(0)
		}

		switch {
		case incr != nil:
			if !incr.IsScalar() {
				err = errors.Errorf(shapeMismatch, ScalarShape(), incr.Shape())
				return
			}
			log.Printf("use Incr : %v", incr)
			if _, err = incr.Trans(res, UseUnsafe()); err != nil {
				err = errors.Wrapf(err, opFail, "Dot scalar incr")
				return
			}
			retVal = incr
			log.Printf("retVal %v", retVal)
		case reuse != nil:
			reuse.set(0, res)
			reuse.reshape()
			retVal = reuse
		default:
			retVal = New(FromScalar(res))
		}
		log.Printf("Returning %v", retVal)
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

	// now that the stupid scalar stuff is out of the way, let's do some linear algebra
	// if incr != nil {
	// 	defer func() {
	// 		if !retVal.Shape().Eq(incr.Shape()) {
	// 			err = errors.Errorf(shapeMismatch, retVal.Shape(), incr.Shape())
	// 			return
	// 		}
	// 		retD := retVal.(*Dense)
	// 		retVal, err = incr.Add(retD, UseUnsafe())
	// 	}()
	// }

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
