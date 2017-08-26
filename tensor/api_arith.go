package tensor

import "github.com/pkg/errors"

// exported API for arithmetics and the stupidly crazy amount of overloaded semantics

// Add performs a pointwise a+b. a and b can either be float64 or Tensor
//
// If both operands are Tensor, shape is checked first.
// Even though the underlying data may have the same size (say (2,2) vs (4,1)), if they have different shapes, it will error out.
//

// Add performs elementwise addition on the Tensor(s). These operations are supported:
//		Add(*Dense, scalar)
//		Add(scalar, *Dense)
//		Add(*Dense, *Dense)
// If the Unsafe flag is passed in, the data of the first tensor will be overwritten
func Add(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	var adder Adder
	var ok bool
	switch at := a.(type) {
	case Tensor:
		if adder, ok = at.Engine().(Adder); !ok {
			return nil, errors.Errorf("Engine does not support Add")
		}
		switch bt := b.(type) {
		case Tensor:
			return adder.Add(at, bt, opts...)
		default:
			return adder.AddScalar(at, b, true, opts...)
		}
	default:
		switch bt := b.(type) {
		case Tensor:
			if adder, ok = bt.Engine().(Adder); !ok {
				return nil, errors.Errorf("Engine does not support AddScalar")
			}
			return adder.AddScalar(bt, a, false, opts...)
		default:
			// error
		}
	}
	panic("Unreachable")
}

// Sub performs elementwise subtraction on the Tensor(s). These operations are supported:
//		Sub(*Dense, scalar)
//		Sub(scalar, *Dense)
//		Sub(*Dense, *Dense)
// If the Unsafe flag is passed in, the data of the first tensor will be overwritten
func Sub(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	var suber Suber
	var ok bool
	switch at := a.(type) {
	case Tensor:
		if suber, ok = at.Engine().(Suber); !ok {
			return nil, errors.Errorf("Engine does not support Sub")
		}
		switch bt := b.(type) {
		case Tensor:
			return suber.Sub(at, bt, opts...)
		default:
			return suber.SubScalar(at, b, true, opts...)
		}
	default:
		switch bt := b.(type) {
		case Tensor:
			if suber, ok = bt.Engine().(Suber); !ok {
				return nil, errors.Errorf("Engine does not support SubScalar")
			}
			return suber.SubScalar(bt, a, false, opts...)
		default:
			// error
		}
	}
	panic("Unreachable")
}

// Mul performs elementwise multiplication on the Tensor(s). These operations are supported:
//		Mul(*Dense, scalar)
//		Mul(scalar, *Dense)
//		Mul(*Dense, *Dense)
// If the Unsafe flag is passed in, the data of the first tensor will be overwritten
func Mul(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	switch at := a.(type) {
	case Tensor:
		switch bt := b.(type) {
		case Tensor:
			switch e := at.Engine().(type) {
			case Float64Muler:
				return e.Float64Mul(at, bt, opts...)
			case Float32Muler:
				return e.Float32Mul(at, bt, opts...)
			case Muler:
				return e.Mul(at, bt, opts...)
			default:
				switch e := bt.Engine().(type) {
				case Float64Muler:
					return e.Float64Mul(at, bt, opts...)
				case Float32Muler:
					return e.Float32Mul(at, bt, opts...)
				case Muler:
					return e.Mul(at, bt, opts...)
				default:
					return nil, errors.New("Neither engines support Mul")
				}
			}

		default:
			switch e := at.Engine().(type) {
			case Float64Muler:
				btf := bt.(float64)
				return e.Float64MulScalar(at, btf, true, opts...)
			case Float32Muler:
				btf := bt.(float32)
				return e.Float32MulScalar(at, btf, true, opts...)
			case Muler:
				return e.MulScalar(at, bt, true, opts...)
			default:
				return nil, errors.New("Engine does not support MulScalar")
			}
		}
	default:
		switch bt := b.(type) {
		case Tensor:
			switch e := bt.Engine().(type) {
			case Float64Muler:
				atf := at.(float64)
				return e.Float64MulScalar(bt, atf, false, opts...)
			case Float32Muler:
				atf := at.(float32)
				return e.Float32MulScalar(bt, atf, false, opts...)
			case Muler:
				return e.MulScalar(bt, at, false, opts...)
			default:
				return nil, errors.New("Engine does not support MulScalar")
			}
		default:
			return nil, errors.Errorf("Cannot perform Mul of %T and %T", a, b)
		}
	}
	panic("Unreachable")
}

// Div performs elementwise division on the Tensor(s). These operations are supported:
//		Div(*Dense, scalar)
//		Div(scalar, *Dense)
//		Div(*Dense, *Dense)
// If the Unsafe flag is passed in, the data of the first tensor will be overwritten
func Div(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	var diver Diver
	var ok bool
	switch at := a.(type) {
	case Tensor:
		if diver, ok = at.Engine().(Diver); !ok {
			// error
		}
		switch bt := b.(type) {
		case Tensor:
			return diver.Div(at, bt, opts...)
		default:
			return diver.DivScalar(at, b, true, opts...)
		}
	default:
		switch bt := b.(type) {
		case Tensor:
			if diver, ok = bt.Engine().(Diver); !ok {
				// error
			}
			return diver.DivScalar(bt, a, false, opts...)
		default:
			// error
		}
	}
	panic("Unreachable")
}

// Pow performs elementwise exponentiation on the Tensor(s). These operations are supported:
//		Pow(*Dense, scalar)
//		Pow(scalar, *Dense)
//		Pow(*Dense, *Dense)
// If the Unsafe flag is passed in, the data of the first tensor will be overwritten
func Pow(a, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	var power Power
	var ok bool
	switch at := a.(type) {
	case Tensor:
		if power, ok = at.Engine().(Power); !ok {
			// error
		}
		switch bt := b.(type) {
		case Tensor:
			return power.Pow(at, bt, opts...)
		default:
			return power.PowScalar(at, b, true, opts...)
		}
	default:
		switch bt := b.(type) {
		case Tensor:
			if power, ok = bt.Engine().(Power); !ok {
				// error
			}
			return power.PowScalar(bt, a, false, opts...)
		default:
			// error
		}
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
	if xdottir, ok := x.Engine().(Dotter); ok {
		return xdottir.Dot(x, y, opts...)
	}
	if ydottir, ok := y.Engine().(Dotter); ok {
		return ydottir.Dot(x, y, opts...)
	}
	return nil, errors.New("Neither x's nor y's engines support Dot")
}

// MatMul performs matrix-matrix multiplication between two Tensors
func MatMul(a, b Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	if a.Dtype() != b.Dtype() {
		err = errors.Errorf(dtypeMismatch, a.Dtype(), b.Dtype())
		return
	}

	switch at := a.(type) {
	case *Dense:
		bt := b.(*Dense)
		return at.MatMul(bt, opts...)
	}
	panic("Unreachable")
}

// MatVecMul performs matrix-vector multiplication between two Tensors. `a` is expected to be a matrix, and `b` is expected to be a vector
func MatVecMul(a, b Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	if a.Dtype() != b.Dtype() {
		err = errors.Errorf(dtypeMismatch, a.Dtype(), b.Dtype())
		return
	}

	switch at := a.(type) {
	case *Dense:
		bt := b.(*Dense)
		return at.MatVecMul(bt, opts...)
	}
	panic("Unreachable")
}

// Inner finds the inner products of two vector Tensors. Both arguments to the functions are eexpected to be vectors.
func Inner(a, b Tensor) (retVal interface{}, err error) {
	if a.Dtype() != b.Dtype() {
		err = errors.Errorf(dtypeMismatch, a.Dtype(), b.Dtype())
		return
	}

	switch at := a.(type) {
	case *Dense:
		bt := b.(*Dense)
		return at.Inner(bt)
	}
	panic("Unreachable")
}

// Outer performs the outer product of two vector Tensors. Both arguments to the functions are expected to be vectors.
func Outer(a, b Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	if a.Dtype() != b.Dtype() {
		err = errors.Errorf(dtypeMismatch, a.Dtype(), b.Dtype())
		return
	}

	switch at := a.(type) {
	case *Dense:
		bt := b.(*Dense)
		return at.Outer(bt, opts...)
	}
	panic("Unreachable")
}
