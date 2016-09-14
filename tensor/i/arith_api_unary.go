package tensori

import (
	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/pkg/errors"
)

// PointwiseSquare squares the elements of the ndarray. The reason why it's called PointwiseSquare instead of Square is because
// A^2 = A · A, and is a valid linalg operation for square matrices (ndarrays with dims() of 2, and both shapes (m, m)).
//
// This function is a convenience function. It is no different from A.PointwiseMul(A). It does not support the incr option yet
func PointwiseSquare(a *Tensor, opts ...types.FuncOpt) (retVal *Tensor, err error) {
	safe, incr, reuse := parseSafeReuse(opts...)
	toReuse := reuse != nil

	if toReuse {
		if a.Size() != reuse.Size() {
			err = types.NewError(types.SizeMismatch, "Cannot reuse %v. Expected size of %v. Got %v instead", reuse, a.Size(), reuse.Size())
			return
		}

		if !a.Shape().Eq(reuse.Shape()) {
			if err = reuse.Reshape(a.Shape()...); err != nil {
				err = errors.Wrapf(err, reuseReshapeErr, a.Shape(), reuse.DataSize())
				return
			}
		}
	}

	switch {
	case incr:
		fallthrough
	case toReuse:
		safeVecMul(a.data, a.data, reuse.data)
		retVal = reuse
	case safe:
		backing := safeVecMul(a.data, a.data)
		retVal = NewTensor(WithBacking(backing), WithShape(a.Shape()...))
	case !safe:
		vecMul(a.data, a.data)
		retVal = a
	}
	return
}

// Sqrt calculates the square root of each elements of the ndarray. Does not support incr option yet
func Sqrt(a *Tensor, opts ...types.FuncOpt) (retVal *Tensor, err error) {
	safe, incr, reuse := parseSafeReuse(opts...)
	toReuse := reuse != nil

	if toReuse {
		if a.Size() != reuse.Size() {
			err = types.NewError(types.SizeMismatch, "Cannot reuse %v. Expected size of %v. Got %v instead", reuse, a.Size(), reuse.Size())
			return
		}

		if !a.Shape().Eq(reuse.Shape()) {
			if err = reuse.Reshape(a.Shape()...); err != nil {
				err = errors.Wrapf(err, reuseReshapeErr, a.Shape(), reuse.DataSize())
				return
			}
		}
	}

	switch {
	case incr:
		fallthrough
	case toReuse:
		safeVecSqrt(a.data, reuse.data)
		retVal = reuse
	case safe:
		backing := safeVecSqrt(a.data)
		retVal = NewTensor(WithBacking(backing), WithShape(a.Shape()...))
	case !safe:
		vecSqrt(a.data)
		retVal = a
	}
	return
}

// InvSqrt calculates 1/sqrt(v) of each element in the *Tensor. Does not support incr option yet
func InvSqrt(a *Tensor, opts ...types.FuncOpt) (retVal *Tensor, err error) {
	safe, incr, reuse := parseSafeReuse(opts...)
	toReuse := reuse != nil

	if toReuse {
		if a.Size() != reuse.Size() {
			err = types.NewError(types.SizeMismatch, "Cannot reuse %v. Expected size of %v. Got %v instead", reuse, a.Size(), reuse.Size())
			return
		}

		if !a.Shape().Eq(reuse.Shape()) {
			if err = reuse.Reshape(a.Shape()...); err != nil {
				err = errors.Wrapf(err, reuseReshapeErr, a.Shape(), reuse.DataSize())
				return
			}
		}
	}

	switch {
	case incr:
		fallthrough
	case toReuse:
		safeVecInvSqrt(a.data, reuse.data)
		retVal = reuse
	case safe:
		backing := safeVecInvSqrt(a.data)
		retVal = NewTensor(WithBacking(backing), WithShape(a.Shape()...))
	case !safe:
		vecInvSqrt(a.data)
		retVal = a
	}
	return
}

// Clamp clamps the values in the *Tensor to the min and max provided. Does not support incr option yet.
func Clamp(a *Tensor, min, max int, opts ...types.FuncOpt) (retVal *Tensor, err error) {
	safe, incr, reuse := parseSafeReuse(opts...)
	toReuse := reuse != nil

	if toReuse {
		if a.Size() != reuse.Size() {
			err = types.NewError(types.SizeMismatch, "Cannot reuse %v. Expected size of %v. Got %v instead", reuse, a.Size(), reuse.Size())
			return
		}

		if !a.Shape().Eq(reuse.Shape()) {
			if err = reuse.Reshape(a.Shape()...); err != nil {
				err = errors.Wrapf(err, reuseReshapeErr, a.Shape(), reuse.DataSize())
				return
			}
		}
	}

	// TODO: meditate on this
	if min >= max {
		// err?

		// This?
		// min, max = max, min
	}

	switch {
	case incr:
		fallthrough
	case toReuse:
		copy(reuse.data, a.data)
		retVal = reuse
	case safe:
		retVal = a.Clone()
	case !safe:
		retVal = a
	}

	for i, v := range retVal.data {
		if v < min {
			retVal.data[i] = min
			continue
		}
		if v > max {
			retVal.data[i] = max
		}
	}
	return
}

// Sign returns the sign function as applied to each element in the ndarray. It does not yet support the incr option
func Sign(a *Tensor, opts ...types.FuncOpt) (retVal *Tensor, err error) {
	safe, incr, reuse := parseSafeReuse(opts...)
	toReuse := reuse != nil

	if toReuse {
		if a.Size() != reuse.Size() {
			err = types.NewError(types.SizeMismatch, "Cannot reuse %v. Expected size of %v. Got %v instead", reuse, a.Size(), reuse.Size())
			return
		}

		if !a.Shape().Eq(reuse.Shape()) {
			if err = reuse.Reshape(a.Shape()...); err != nil {
				err = errors.Wrapf(err, reuseReshapeErr, a.Shape(), reuse.DataSize())
				return
			}
		}
	}

	switch {
	case incr:
		fallthrough
	case toReuse:
		copy(reuse.data, a.data)
		retVal = reuse
	case safe:
		retVal = BorrowTensor(len(a.data))
		copy(retVal.data, a.data)
	case !safe:
		retVal = a
	}

	for i, v := range retVal.data {
		if v < 0 {
			retVal.data[i] = int(-1)
			continue
		}
		if v > 0 {
			retVal.data[i] = int(1)
			continue
		}
	}
	return
}
