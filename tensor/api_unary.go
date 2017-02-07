package tensor

import (
	"math"
	"reflect"

	"github.com/chewxy/math32"
	"github.com/chewxy/vecf32"
	"github.com/chewxy/vecf64"
	"github.com/pkg/errors"
)

// Square squares the elements of the Tensor. This function used to be called PointwiseSquare instead of Square.
// If you want to achieve a Matrix Square as defined:
// 		A^2 = A · A,
// You should call this function instead:
//		A.MatMul(A)
//
// This function is a convenience function. It is no different from A.Mul(A).
func Square(a Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	switch at := a.(type) {
	case *Dense:
		var reuse *Dense
		var safe, toReuse, incr bool
		if reuse, safe, toReuse, incr, err = prepUnaryDense(at, opts...); err != nil {
			err = errors.Wrapf(err, opFail, "PointwiseSquare")
			return
		}

		if !isNumber(at.t) {
			err = errors.Errorf("PointwiseSquare only works on numbers")
			return
		}

		switch {
		case incr:
			var ret *Dense
			if ret, err = at.Mul(at); err != nil {
				err = errors.Wrapf(err, opFail, "Mul")
				return
			}
			return reuse.Add(ret, UseUnsafe())
		case toReuse:
			copyDense(reuse, at)
			return reuse.Mul(at, UseUnsafe())
		case safe:
			return at.Mul(at)
		case !safe:
			return at.Mul(at, UseUnsafe())
		}
		return

	default:
		panic("NYI - not yet implemented")
	}
	return
}

// Sqrt calculates the square root of each elements of the Tensor.
func Sqrt(a Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	switch t := a.(type) {
	case *Dense:
		if t.IsMaterializable() {
			var f interface{}
			switch t.t.Kind() {
			case reflect.Float64:
				f = math.Sqrt
			case reflect.Float32:
				f = math32.Sqrt
			default:
				err = errors.Errorf("Sqrt only works on floats")
				return
			}
			return t.Apply(f, opts...)
		}
		if !isFloat(t.t) {
			err = errors.Errorf("Sqrt only works on floats")
			return
		}

		// otherwise, we have optimizations for this
		var reuse *Dense
		var safe, toReuse, incr bool
		if reuse, safe, toReuse, incr, err = prepUnaryDense(t, opts...); err != nil {
			err = errors.Wrapf(err, opFail, "PointwiseSquare")
			return
		}

		switch {
		case incr:
			cloned := t.Clone().(*Dense)
			switch t.t.Kind() {
			case reflect.Float64:
				vecf64.Sqrt(cloned.float64s())
			case reflect.Float32:
				vecf32.Sqrt(cloned.float32s())
			}
			_, err = reuse.Add(cloned, UseUnsafe())
			retVal = reuse
		case toReuse:
			copyDense(reuse, t)
			switch t.t.Kind() {
			case reflect.Float64:
				vecf64.Sqrt(reuse.float64s())
			case reflect.Float32:
				vecf32.Sqrt(reuse.float32s())
			}
			retVal = reuse
		case safe:
			cloned := t.Clone().(*Dense)
			switch t.t.Kind() {
			case reflect.Float64:
				vecf64.Sqrt(cloned.float64s())
			case reflect.Float32:
				vecf32.Sqrt(cloned.float32s())
			}
			retVal = cloned
		case !safe:
			switch t.t.Kind() {
			case reflect.Float64:
				vecf64.Sqrt(t.float64s())
			case reflect.Float32:
				vecf32.Sqrt(t.float32s())
			}
			retVal = t
		}
	default:
		panic("NYI - not yet implemented")
	}
	return
}

// InvSqrt calculates 1/sqrt(v) of each element in the *Tensor. Does not support incr option yet
func InvSqrt(a Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	switch t := a.(type) {
	case *Dense:
		if t.IsMaterializable() {
			var f interface{}
			switch t.t.Kind() {
			case reflect.Float64:
				f = func(a float64) float64 { return float64(1) / math.Sqrt(a) }
			case reflect.Float32:
				f = func(a float32) float32 { return float32(1) / math32.Sqrt(a) }
			default:
				err = errors.Errorf("InvSqrt only works on floats")
				return
			}
			return t.Apply(f, opts...)
		}
		if !isFloat(t.t) {
			err = errors.Errorf("InvSqrt only works on floats")
			return
		}

		// otherwise, we have optimizations for this
		var reuse *Dense
		var safe, toReuse, incr bool
		if reuse, safe, toReuse, incr, err = prepUnaryDense(t, opts...); err != nil {
			err = errors.Wrapf(err, opFail, "PointwiseSquare")
			return
		}

		switch {
		case incr:
			cloned := t.Clone().(*Dense)
			switch t.t.Kind() {
			case reflect.Float64:
				vecf64.InvSqrt(cloned.float64s())
			case reflect.Float32:
				vecf32.InvSqrt(cloned.float32s())
			}
			_, err = reuse.Add(cloned, UseUnsafe())
			retVal = reuse
		case toReuse:
			copyDense(reuse, t)
			switch t.t.Kind() {
			case reflect.Float64:
				vecf64.InvSqrt(reuse.float64s())
			case reflect.Float32:
				vecf32.InvSqrt(reuse.float32s())
			}
			retVal = reuse
		case safe:
			cloned := t.Clone().(*Dense)
			switch t.t.Kind() {
			case reflect.Float64:
				vecf64.InvSqrt(cloned.float64s())
			case reflect.Float32:
				vecf32.InvSqrt(cloned.float32s())
			}
			retVal = cloned
		case !safe:
			switch t.t.Kind() {
			case reflect.Float64:
				vecf64.InvSqrt(t.float64s())
			case reflect.Float32:
				vecf32.InvSqrt(t.float32s())
			}
			retVal = t
		}
	default:
		panic("NYI - not yet implemented")
	}
	return
}
