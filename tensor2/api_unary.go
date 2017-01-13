package tensor

import (
	"math"

	"github.com/chewxy/math32"
	"github.com/chewxy/vecf32"
	"github.com/chewxy/vecf64"
	"github.com/pkg/errors"
)

// PointwiseSquare squares the elements of the ndarray. The reason why it's called PointwiseSquare instead of Square is because
// A^2 = A · A, and is a valid linalg operation for square matrices (ndarrays with dims() of 2, and both shapes (m, m)).
//
// This function is a convenience function. It is no different from A.PointwiseMul(A). It does not support the incr option yet
func PointwiseSquare(a Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	switch at := a.(type) {
	case *Dense:
		var an, rn Number
		var reuse *Dense
		var safe, toReuse, incr bool
		if an, rn, reuse, safe, toReuse, incr, err = prepUnaryDense(at, opts...); err != nil {
			err = errors.Wrapf(err, opFail, "PointwiseSquare")
			return
		}

		switch {
		case incr:
			fallthrough
		case toReuse:
			_, err = safeMul(an, an, rn)
			retVal = reuse
		case safe:
			var backing Number
			if backing, err = safeMul(an, an); err == nil {
				retVal = New(Of(at.t), WithBacking(backing), WithShape(a.Shape().Clone()...))
			}

		case !safe:
			if err = an.Mul(an); err != nil {
				return
			}
			retVal = a
		}
		return

	default:
		panic("NYI - not yet implemented")
	}
	return
}

// InvSqrt calculates 1/sqrt(v) of each element in the *Tensor. Does not support incr option yet
func InvSqrt(a Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	switch at := a.(type) {
	case *Dense:
		var an, rn Number
		var reuse *Dense
		var safe, toReuse, incr bool
		if an, rn, reuse, safe, toReuse, incr, err = prepUnaryDense(at, opts...); err != nil {
			err = errors.Wrapf(err, opFail, "InvSqrt")
			return
		}

		switch {
		case incr:
			fallthrough
		case toReuse:
			_, err = safeInvSqrt(an, rn)
			retVal = reuse
		case safe:
			var backing Number
			if backing, err = safeInvSqrt(an); err != nil {
				retVal = New(Of(at.t), WithBacking(backing), WithShape(a.Shape().Clone()...))
			}
		case !safe:
			switch arr := an.(type) {
			case Float64ser:
				vecf64.InvSqrt(arr.Float64s())
			case Float32ser:
				vecf32.InvSqrt(arr.Float32s())
			default:
				err = errors.Errorf(unsupportedDtype, an, "InvSqrt")
			}
			retVal = a
		}
		return

	default:
		panic("NYI")
	}
}

// Clamp clamps the values in the *Tensor to the min and max provided. Does not support incr option yet.
func Clamp(a Tensor, min, max interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	switch at := a.(type) {
	case *Dense:
		// var an, rn Number
		var reuse *Dense
		var safe, toReuse, incr bool
		if _, _, reuse, safe, toReuse, incr, err = prepUnaryDense(at, opts...); err != nil {
			err = errors.Wrapf(err, opFail, "InvSqrt")
			return
		}

		var retD *Dense
		switch {
		case incr:
			fallthrough
		case toReuse:
			copyArray(reuse.data, at.data)
			retD = reuse
		case safe:
			retD = at.Clone().(*Dense)
		case !safe:
			retD = at
		}

		switch dt := retD.data.(type) {
		case Float64ser:
			var minF, maxF float64
			if minF, err = getFloat64(min); err != nil {
				err = errors.Wrapf(err, opFail, "Clamp min")
			}
			if maxF, err = getFloat64(max); err != nil {
				err = errors.Wrapf(err, opFail, "Clamp min")
			}
			data := dt.Float64s()
			for i, v := range data {
				if v < minF || math.IsInf(v, -1) {
					data[i] = minF
					continue
				}
				if v > maxF || math.IsInf(v, 1) {
					data[i] = maxF
				}
			}

		case Float32ser:
			var minF, maxF float32
			if minF, err = getFloat32(min); err != nil {
				err = errors.Wrapf(err, opFail, "Clamp min")
				return
			}
			if maxF, err = getFloat32(max); err != nil {
				err = errors.Wrapf(err, opFail, "Clamp min")
				return
			}
			data := dt.Float32s()
			for i, v := range data {
				if v < minF || math32.IsInf(v, -1) {
					data[i] = minF
					continue
				}
				if v > maxF || math32.IsInf(v, 1) {
					data[i] = maxF
				}
			}

		case Intser:
			var minF, maxF int
			if minF, err = getInt(min); err != nil {
				err = errors.Wrapf(err, opFail, "Clamp min")
				return
			}
			if maxF, err = getInt(max); err != nil {
				err = errors.Wrapf(err, opFail, "Clamp min")
				return
			}
			data := dt.Ints()
			for i, v := range data {
				if v < minF {
					data[i] = minF
					continue
				}
				if v > maxF {
					data[i] = maxF
				}
			}
		case Int64ser:
			var minF, maxF int64
			if minF, err = getInt64(min); err != nil {
				err = errors.Wrapf(err, opFail, "Clamp min")
				return
			}
			if maxF, err = getInt64(max); err != nil {
				err = errors.Wrapf(err, opFail, "Clamp min")
				return
			}
			data := dt.Int64s()
			for i, v := range data {
				if v < minF {
					data[i] = minF
					continue
				}
				if v > maxF {
					data[i] = maxF
				}
			}
		case Int32ser:
			var minF, maxF int32
			if minF, err = getInt32(min); err != nil {
				err = errors.Wrapf(err, opFail, "Clamp min")
				return
			}
			if maxF, err = getInt32(max); err != nil {
				err = errors.Wrapf(err, opFail, "Clamp min")
				return
			}
			data := dt.Int32s()
			for i, v := range data {
				if v < minF {
					data[i] = minF
					continue
				}
				if v > maxF {
					data[i] = maxF
				}
			}
		case Byteser:
			var minF, maxF byte
			if minF, err = getByte(min); err != nil {
				err = errors.Wrapf(err, opFail, "Clamp min")
				return
			}
			if maxF, err = getByte(max); err != nil {
				err = errors.Wrapf(err, opFail, "Clamp min")
				return
			}
			data := dt.Bytes()
			for i, v := range data {
				if v < minF {
					data[i] = minF
					continue
				}
				if v > maxF {
					data[i] = maxF
				}
			}
		case Boolser:
			return nil, errors.Errorf(unsupportedDtype, retD.data, "Clamp")
		}
	default:
		panic("NYI")
	}
	return
}
