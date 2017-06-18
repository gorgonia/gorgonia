package tensor

import (
	"math"
	"math/cmplx"

	"github.com/chewxy/vecf32"
	"github.com/chewxy/vecf64"
	"github.com/pkg/errors"
)

/*
GENERATED FILE. DO NOT EDIT
*/

/* Add */

func vecAddI(a, b []int) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] += v
	}
	return nil
}

func vecAddIMasked(a, b []int, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] += v
		}
	}

	return nil
}

func vecAddI8(a, b []int8) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] += v
	}
	return nil
}

func vecAddI8Masked(a, b []int8, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] += v
		}
	}

	return nil
}

func vecAddI16(a, b []int16) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] += v
	}
	return nil
}

func vecAddI16Masked(a, b []int16, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] += v
		}
	}

	return nil
}

func vecAddI32(a, b []int32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] += v
	}
	return nil
}

func vecAddI32Masked(a, b []int32, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] += v
		}
	}

	return nil
}

func vecAddI64(a, b []int64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] += v
	}
	return nil
}

func vecAddI64Masked(a, b []int64, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] += v
		}
	}

	return nil
}

func vecAddU(a, b []uint) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] += v
	}
	return nil
}

func vecAddUMasked(a, b []uint, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] += v
		}
	}

	return nil
}

func vecAddU8(a, b []uint8) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] += v
	}
	return nil
}

func vecAddU8Masked(a, b []uint8, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] += v
		}
	}

	return nil
}

func vecAddU16(a, b []uint16) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] += v
	}
	return nil
}

func vecAddU16Masked(a, b []uint16, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] += v
		}
	}

	return nil
}

func vecAddU32(a, b []uint32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] += v
	}
	return nil
}

func vecAddU32Masked(a, b []uint32, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] += v
		}
	}

	return nil
}

func vecAddU64(a, b []uint64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] += v
	}
	return nil
}

func vecAddU64Masked(a, b []uint64, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] += v
		}
	}

	return nil
}

func vecAddF32(a, b []float32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	vecf32.Add(a, b)
	return nil
}

func vecAddF32Masked(a, b []float32, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] += v
		}
	}

	return nil
}

func vecAddF64(a, b []float64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	vecf64.Add(a, b)
	return nil
}

func vecAddF64Masked(a, b []float64, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] += v
		}
	}

	return nil
}

func vecAddC64(a, b []complex64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] += v
	}
	return nil
}

func vecAddC64Masked(a, b []complex64, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] += v
		}
	}
	return nil
}

func vecAddC128(a, b []complex128) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] += v
	}
	return nil
}

func vecAddC128Masked(a, b []complex128, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] += v
		}
	}
	return nil
}

/* Sub */

func vecSubI(a, b []int) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] -= v
	}
	return nil
}

func vecSubIMasked(a, b []int, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] -= v
		}
	}

	return nil
}

func vecSubI8(a, b []int8) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] -= v
	}
	return nil
}

func vecSubI8Masked(a, b []int8, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] -= v
		}
	}

	return nil
}

func vecSubI16(a, b []int16) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] -= v
	}
	return nil
}

func vecSubI16Masked(a, b []int16, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] -= v
		}
	}

	return nil
}

func vecSubI32(a, b []int32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] -= v
	}
	return nil
}

func vecSubI32Masked(a, b []int32, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] -= v
		}
	}

	return nil
}

func vecSubI64(a, b []int64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] -= v
	}
	return nil
}

func vecSubI64Masked(a, b []int64, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] -= v
		}
	}

	return nil
}

func vecSubU(a, b []uint) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] -= v
	}
	return nil
}

func vecSubUMasked(a, b []uint, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] -= v
		}
	}

	return nil
}

func vecSubU8(a, b []uint8) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] -= v
	}
	return nil
}

func vecSubU8Masked(a, b []uint8, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] -= v
		}
	}

	return nil
}

func vecSubU16(a, b []uint16) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] -= v
	}
	return nil
}

func vecSubU16Masked(a, b []uint16, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] -= v
		}
	}

	return nil
}

func vecSubU32(a, b []uint32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] -= v
	}
	return nil
}

func vecSubU32Masked(a, b []uint32, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] -= v
		}
	}

	return nil
}

func vecSubU64(a, b []uint64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] -= v
	}
	return nil
}

func vecSubU64Masked(a, b []uint64, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] -= v
		}
	}

	return nil
}

func vecSubF32(a, b []float32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	vecf32.Sub(a, b)
	return nil
}

func vecSubF32Masked(a, b []float32, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] -= v
		}
	}

	return nil
}

func vecSubF64(a, b []float64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	vecf64.Sub(a, b)
	return nil
}

func vecSubF64Masked(a, b []float64, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] -= v
		}
	}

	return nil
}

func vecSubC64(a, b []complex64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] -= v
	}
	return nil
}

func vecSubC64Masked(a, b []complex64, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] -= v
		}
	}
	return nil
}

func vecSubC128(a, b []complex128) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] -= v
	}
	return nil
}

func vecSubC128Masked(a, b []complex128, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] -= v
		}
	}
	return nil
}

/* Mul */

func vecMulI(a, b []int) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] *= v
	}
	return nil
}

func vecMulIMasked(a, b []int, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] *= v
		}
	}

	return nil
}

func vecMulI8(a, b []int8) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] *= v
	}
	return nil
}

func vecMulI8Masked(a, b []int8, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] *= v
		}
	}

	return nil
}

func vecMulI16(a, b []int16) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] *= v
	}
	return nil
}

func vecMulI16Masked(a, b []int16, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] *= v
		}
	}

	return nil
}

func vecMulI32(a, b []int32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] *= v
	}
	return nil
}

func vecMulI32Masked(a, b []int32, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] *= v
		}
	}

	return nil
}

func vecMulI64(a, b []int64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] *= v
	}
	return nil
}

func vecMulI64Masked(a, b []int64, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] *= v
		}
	}

	return nil
}

func vecMulU(a, b []uint) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] *= v
	}
	return nil
}

func vecMulUMasked(a, b []uint, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] *= v
		}
	}

	return nil
}

func vecMulU8(a, b []uint8) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] *= v
	}
	return nil
}

func vecMulU8Masked(a, b []uint8, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] *= v
		}
	}

	return nil
}

func vecMulU16(a, b []uint16) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] *= v
	}
	return nil
}

func vecMulU16Masked(a, b []uint16, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] *= v
		}
	}

	return nil
}

func vecMulU32(a, b []uint32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] *= v
	}
	return nil
}

func vecMulU32Masked(a, b []uint32, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] *= v
		}
	}

	return nil
}

func vecMulU64(a, b []uint64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] *= v
	}
	return nil
}

func vecMulU64Masked(a, b []uint64, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] *= v
		}
	}

	return nil
}

func vecMulF32(a, b []float32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	vecf32.Mul(a, b)
	return nil
}

func vecMulF32Masked(a, b []float32, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] *= v
		}
	}

	return nil
}

func vecMulF64(a, b []float64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	vecf64.Mul(a, b)
	return nil
}

func vecMulF64Masked(a, b []float64, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] *= v
		}
	}

	return nil
}

func vecMulC64(a, b []complex64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] *= v
	}
	return nil
}

func vecMulC64Masked(a, b []complex64, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] *= v
		}
	}
	return nil
}

func vecMulC128(a, b []complex128) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] *= v
	}
	return nil
}

func vecMulC128Masked(a, b []complex128, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] *= v
		}
	}
	return nil
}

/* Div */

func vecDivI(a, b []int) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	var errs errorIndices
	for i, v := range b {
		if v == int(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}
		a[i] /= v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func vecDivIMasked(a, b []int, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	var errs errorIndices
	for i, v := range b {
		if !mask[i] {
			if v == int(0) {
				errs = append(errs, i)
				a[i] = 0
				continue
			}
			a[i] /= v
		}
	}

	if errs != nil {
		return errs
	}
	return nil
}

func vecDivI8(a, b []int8) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	var errs errorIndices
	for i, v := range b {
		if v == int8(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}
		a[i] /= v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func vecDivI8Masked(a, b []int8, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	var errs errorIndices
	for i, v := range b {
		if !mask[i] {
			if v == int8(0) {
				errs = append(errs, i)
				a[i] = 0
				continue
			}
			a[i] /= v
		}
	}

	if errs != nil {
		return errs
	}
	return nil
}

func vecDivI16(a, b []int16) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	var errs errorIndices
	for i, v := range b {
		if v == int16(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}
		a[i] /= v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func vecDivI16Masked(a, b []int16, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	var errs errorIndices
	for i, v := range b {
		if !mask[i] {
			if v == int16(0) {
				errs = append(errs, i)
				a[i] = 0
				continue
			}
			a[i] /= v
		}
	}

	if errs != nil {
		return errs
	}
	return nil
}

func vecDivI32(a, b []int32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	var errs errorIndices
	for i, v := range b {
		if v == int32(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}
		a[i] /= v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func vecDivI32Masked(a, b []int32, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	var errs errorIndices
	for i, v := range b {
		if !mask[i] {
			if v == int32(0) {
				errs = append(errs, i)
				a[i] = 0
				continue
			}
			a[i] /= v
		}
	}

	if errs != nil {
		return errs
	}
	return nil
}

func vecDivI64(a, b []int64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	var errs errorIndices
	for i, v := range b {
		if v == int64(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}
		a[i] /= v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func vecDivI64Masked(a, b []int64, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	var errs errorIndices
	for i, v := range b {
		if !mask[i] {
			if v == int64(0) {
				errs = append(errs, i)
				a[i] = 0
				continue
			}
			a[i] /= v
		}
	}

	if errs != nil {
		return errs
	}
	return nil
}

func vecDivU(a, b []uint) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	var errs errorIndices
	for i, v := range b {
		if v == uint(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}
		a[i] /= v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func vecDivUMasked(a, b []uint, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	var errs errorIndices
	for i, v := range b {
		if !mask[i] {
			if v == uint(0) {
				errs = append(errs, i)
				a[i] = 0
				continue
			}
			a[i] /= v
		}
	}

	if errs != nil {
		return errs
	}
	return nil
}

func vecDivU8(a, b []uint8) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	var errs errorIndices
	for i, v := range b {
		if v == uint8(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}
		a[i] /= v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func vecDivU8Masked(a, b []uint8, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	var errs errorIndices
	for i, v := range b {
		if !mask[i] {
			if v == uint8(0) {
				errs = append(errs, i)
				a[i] = 0
				continue
			}
			a[i] /= v
		}
	}

	if errs != nil {
		return errs
	}
	return nil
}

func vecDivU16(a, b []uint16) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	var errs errorIndices
	for i, v := range b {
		if v == uint16(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}
		a[i] /= v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func vecDivU16Masked(a, b []uint16, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	var errs errorIndices
	for i, v := range b {
		if !mask[i] {
			if v == uint16(0) {
				errs = append(errs, i)
				a[i] = 0
				continue
			}
			a[i] /= v
		}
	}

	if errs != nil {
		return errs
	}
	return nil
}

func vecDivU32(a, b []uint32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	var errs errorIndices
	for i, v := range b {
		if v == uint32(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}
		a[i] /= v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func vecDivU32Masked(a, b []uint32, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	var errs errorIndices
	for i, v := range b {
		if !mask[i] {
			if v == uint32(0) {
				errs = append(errs, i)
				a[i] = 0
				continue
			}
			a[i] /= v
		}
	}

	if errs != nil {
		return errs
	}
	return nil
}

func vecDivU64(a, b []uint64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	var errs errorIndices
	for i, v := range b {
		if v == uint64(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}
		a[i] /= v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func vecDivU64Masked(a, b []uint64, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	var errs errorIndices
	for i, v := range b {
		if !mask[i] {
			if v == uint64(0) {
				errs = append(errs, i)
				a[i] = 0
				continue
			}
			a[i] /= v
		}
	}

	if errs != nil {
		return errs
	}
	return nil
}

func vecDivF32(a, b []float32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	vecf32.Div(a, b)
	return nil
}

func vecDivF32Masked(a, b []float32, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	var errs errorIndices
	for i, v := range b {
		if !mask[i] {
			if v == float32(0) {
				errs = append(errs, i)
				a[i] = 0
				continue
			}
			a[i] /= v
		}
	}

	if errs != nil {
		return errs
	}
	return nil
}

func vecDivF64(a, b []float64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	vecf64.Div(a, b)
	return nil
}

func vecDivF64Masked(a, b []float64, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	var errs errorIndices
	for i, v := range b {
		if !mask[i] {
			if v == float64(0) {
				errs = append(errs, i)
				a[i] = 0
				continue
			}
			a[i] /= v
		}
	}

	if errs != nil {
		return errs
	}
	return nil
}

func vecDivC64(a, b []complex64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] /= v
	}
	return nil
}

func vecDivC64Masked(a, b []complex64, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] /= v
		}
	}
	return nil
}

func vecDivC128(a, b []complex128) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] /= v
	}
	return nil
}

func vecDivC128Masked(a, b []complex128, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] /= v
		}
	}
	return nil
}

/* Pow */

func vecPowI(a, b []int) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] = int(math.Pow(float64(a[i]), float64(v)))
	}
	return nil
}

func vecPowIMasked(a, b []int, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] = int(math.Pow(float64(a[i]), float64(v)))
		}
	}

	return nil
}

func vecPowI8(a, b []int8) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] = int8(math.Pow(float64(a[i]), float64(v)))
	}
	return nil
}

func vecPowI8Masked(a, b []int8, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] = int8(math.Pow(float64(a[i]), float64(v)))
		}
	}

	return nil
}

func vecPowI16(a, b []int16) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] = int16(math.Pow(float64(a[i]), float64(v)))
	}
	return nil
}

func vecPowI16Masked(a, b []int16, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] = int16(math.Pow(float64(a[i]), float64(v)))
		}
	}

	return nil
}

func vecPowI32(a, b []int32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] = int32(math.Pow(float64(a[i]), float64(v)))
	}
	return nil
}

func vecPowI32Masked(a, b []int32, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] = int32(math.Pow(float64(a[i]), float64(v)))
		}
	}

	return nil
}

func vecPowI64(a, b []int64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] = int64(math.Pow(float64(a[i]), float64(v)))
	}
	return nil
}

func vecPowI64Masked(a, b []int64, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] = int64(math.Pow(float64(a[i]), float64(v)))
		}
	}

	return nil
}

func vecPowU(a, b []uint) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] = uint(math.Pow(float64(a[i]), float64(v)))
	}
	return nil
}

func vecPowUMasked(a, b []uint, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] = uint(math.Pow(float64(a[i]), float64(v)))
		}
	}

	return nil
}

func vecPowU8(a, b []uint8) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] = uint8(math.Pow(float64(a[i]), float64(v)))
	}
	return nil
}

func vecPowU8Masked(a, b []uint8, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] = uint8(math.Pow(float64(a[i]), float64(v)))
		}
	}

	return nil
}

func vecPowU16(a, b []uint16) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] = uint16(math.Pow(float64(a[i]), float64(v)))
	}
	return nil
}

func vecPowU16Masked(a, b []uint16, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] = uint16(math.Pow(float64(a[i]), float64(v)))
		}
	}

	return nil
}

func vecPowU32(a, b []uint32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] = uint32(math.Pow(float64(a[i]), float64(v)))
	}
	return nil
}

func vecPowU32Masked(a, b []uint32, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] = uint32(math.Pow(float64(a[i]), float64(v)))
		}
	}

	return nil
}

func vecPowU64(a, b []uint64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] = uint64(math.Pow(float64(a[i]), float64(v)))
	}
	return nil
}

func vecPowU64Masked(a, b []uint64, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] = uint64(math.Pow(float64(a[i]), float64(v)))
		}
	}

	return nil
}

func vecPowF32(a, b []float32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	vecf32.Pow(a, b)
	return nil
}

func vecPowF32Masked(a, b []float32, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] = float32(math.Pow(float64(a[i]), float64(v)))
		}
	}

	return nil
}

func vecPowF64(a, b []float64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	vecf64.Pow(a, b)
	return nil
}

func vecPowF64Masked(a, b []float64, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] = float64(math.Pow(float64(a[i]), float64(v)))
		}
	}

	return nil
}

func vecPowC64(a, b []complex64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] = complex64(cmplx.Pow(complex128(a[i]), complex128(v)))
	}
	return nil
}

func vecPowC64Masked(a, b []complex64, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] = complex64(cmplx.Pow(complex128(a[i]), complex128(v)))
		}
	}
	return nil
}

func vecPowC128(a, b []complex128) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		a[i] = cmplx.Pow(a[i], v)
	}
	return nil
}

func vecPowC128Masked(a, b []complex128, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	for i, v := range b {
		if !mask[i] {
			a[i] = cmplx.Pow(a[i], v)
		}
	}
	return nil
}

/* incr Add */

func incrVecAddI(a, b, incr []int) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += a[i] + v
	}
	return nil
}

func incrVecAddIMasked(a, b, incr []int, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] + v
		}
	}

	return nil
}

func incrVecAddI8(a, b, incr []int8) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += a[i] + v
	}
	return nil
}

func incrVecAddI8Masked(a, b, incr []int8, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] + v
		}
	}

	return nil
}

func incrVecAddI16(a, b, incr []int16) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += a[i] + v
	}
	return nil
}

func incrVecAddI16Masked(a, b, incr []int16, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] + v
		}
	}

	return nil
}

func incrVecAddI32(a, b, incr []int32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += a[i] + v
	}
	return nil
}

func incrVecAddI32Masked(a, b, incr []int32, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] + v
		}
	}

	return nil
}

func incrVecAddI64(a, b, incr []int64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += a[i] + v
	}
	return nil
}

func incrVecAddI64Masked(a, b, incr []int64, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] + v
		}
	}

	return nil
}

func incrVecAddU(a, b, incr []uint) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += a[i] + v
	}
	return nil
}

func incrVecAddUMasked(a, b, incr []uint, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] + v
		}
	}

	return nil
}

func incrVecAddU8(a, b, incr []uint8) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += a[i] + v
	}
	return nil
}

func incrVecAddU8Masked(a, b, incr []uint8, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] + v
		}
	}

	return nil
}

func incrVecAddU16(a, b, incr []uint16) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += a[i] + v
	}
	return nil
}

func incrVecAddU16Masked(a, b, incr []uint16, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] + v
		}
	}

	return nil
}

func incrVecAddU32(a, b, incr []uint32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += a[i] + v
	}
	return nil
}

func incrVecAddU32Masked(a, b, incr []uint32, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] + v
		}
	}

	return nil
}

func incrVecAddU64(a, b, incr []uint64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += a[i] + v
	}
	return nil
}

func incrVecAddU64Masked(a, b, incr []uint64, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] + v
		}
	}

	return nil
}

func incrVecAddF32(a, b, incr []float32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	vecf32.IncrAdd(a, b, incr)
	return nil
}

func incrVecAddF32Masked(a, b, incr []float32, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] + v
		}
	}

	return nil
}

func incrVecAddF64(a, b, incr []float64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	vecf64.IncrAdd(a, b, incr)
	return nil
}

func incrVecAddF64Masked(a, b, incr []float64, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] + v
		}
	}

	return nil
}

func incrVecAddC64(a, b, incr []complex64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += a[i] + v
	}
	return nil
}

func incrVecAddC64Masked(a, b, incr []complex64, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] + v
		}
	}
	return nil
}

func incrVecAddC128(a, b, incr []complex128) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += a[i] + v
	}
	return nil
}

func incrVecAddC128Masked(a, b, incr []complex128, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] + v
		}
	}
	return nil
}

/* incr Sub */

func incrVecSubI(a, b, incr []int) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += a[i] - v
	}
	return nil
}

func incrVecSubIMasked(a, b, incr []int, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] - v
		}
	}

	return nil
}

func incrVecSubI8(a, b, incr []int8) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += a[i] - v
	}
	return nil
}

func incrVecSubI8Masked(a, b, incr []int8, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] - v
		}
	}

	return nil
}

func incrVecSubI16(a, b, incr []int16) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += a[i] - v
	}
	return nil
}

func incrVecSubI16Masked(a, b, incr []int16, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] - v
		}
	}

	return nil
}

func incrVecSubI32(a, b, incr []int32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += a[i] - v
	}
	return nil
}

func incrVecSubI32Masked(a, b, incr []int32, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] - v
		}
	}

	return nil
}

func incrVecSubI64(a, b, incr []int64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += a[i] - v
	}
	return nil
}

func incrVecSubI64Masked(a, b, incr []int64, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] - v
		}
	}

	return nil
}

func incrVecSubU(a, b, incr []uint) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += a[i] - v
	}
	return nil
}

func incrVecSubUMasked(a, b, incr []uint, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] - v
		}
	}

	return nil
}

func incrVecSubU8(a, b, incr []uint8) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += a[i] - v
	}
	return nil
}

func incrVecSubU8Masked(a, b, incr []uint8, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] - v
		}
	}

	return nil
}

func incrVecSubU16(a, b, incr []uint16) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += a[i] - v
	}
	return nil
}

func incrVecSubU16Masked(a, b, incr []uint16, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] - v
		}
	}

	return nil
}

func incrVecSubU32(a, b, incr []uint32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += a[i] - v
	}
	return nil
}

func incrVecSubU32Masked(a, b, incr []uint32, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] - v
		}
	}

	return nil
}

func incrVecSubU64(a, b, incr []uint64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += a[i] - v
	}
	return nil
}

func incrVecSubU64Masked(a, b, incr []uint64, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] - v
		}
	}

	return nil
}

func incrVecSubF32(a, b, incr []float32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	vecf32.IncrSub(a, b, incr)
	return nil
}

func incrVecSubF32Masked(a, b, incr []float32, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] - v
		}
	}

	return nil
}

func incrVecSubF64(a, b, incr []float64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	vecf64.IncrSub(a, b, incr)
	return nil
}

func incrVecSubF64Masked(a, b, incr []float64, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] - v
		}
	}

	return nil
}

func incrVecSubC64(a, b, incr []complex64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += a[i] - v
	}
	return nil
}

func incrVecSubC64Masked(a, b, incr []complex64, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] - v
		}
	}
	return nil
}

func incrVecSubC128(a, b, incr []complex128) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += a[i] - v
	}
	return nil
}

func incrVecSubC128Masked(a, b, incr []complex128, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] - v
		}
	}
	return nil
}

/* incr Mul */

func incrVecMulI(a, b, incr []int) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += a[i] * v
	}
	return nil
}

func incrVecMulIMasked(a, b, incr []int, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] * v
		}
	}

	return nil
}

func incrVecMulI8(a, b, incr []int8) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += a[i] * v
	}
	return nil
}

func incrVecMulI8Masked(a, b, incr []int8, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] * v
		}
	}

	return nil
}

func incrVecMulI16(a, b, incr []int16) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += a[i] * v
	}
	return nil
}

func incrVecMulI16Masked(a, b, incr []int16, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] * v
		}
	}

	return nil
}

func incrVecMulI32(a, b, incr []int32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += a[i] * v
	}
	return nil
}

func incrVecMulI32Masked(a, b, incr []int32, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] * v
		}
	}

	return nil
}

func incrVecMulI64(a, b, incr []int64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += a[i] * v
	}
	return nil
}

func incrVecMulI64Masked(a, b, incr []int64, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] * v
		}
	}

	return nil
}

func incrVecMulU(a, b, incr []uint) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += a[i] * v
	}
	return nil
}

func incrVecMulUMasked(a, b, incr []uint, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] * v
		}
	}

	return nil
}

func incrVecMulU8(a, b, incr []uint8) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += a[i] * v
	}
	return nil
}

func incrVecMulU8Masked(a, b, incr []uint8, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] * v
		}
	}

	return nil
}

func incrVecMulU16(a, b, incr []uint16) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += a[i] * v
	}
	return nil
}

func incrVecMulU16Masked(a, b, incr []uint16, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] * v
		}
	}

	return nil
}

func incrVecMulU32(a, b, incr []uint32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += a[i] * v
	}
	return nil
}

func incrVecMulU32Masked(a, b, incr []uint32, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] * v
		}
	}

	return nil
}

func incrVecMulU64(a, b, incr []uint64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += a[i] * v
	}
	return nil
}

func incrVecMulU64Masked(a, b, incr []uint64, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] * v
		}
	}

	return nil
}

func incrVecMulF32(a, b, incr []float32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	vecf32.IncrMul(a, b, incr)
	return nil
}

func incrVecMulF32Masked(a, b, incr []float32, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] * v
		}
	}

	return nil
}

func incrVecMulF64(a, b, incr []float64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	vecf64.IncrMul(a, b, incr)
	return nil
}

func incrVecMulF64Masked(a, b, incr []float64, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] * v
		}
	}

	return nil
}

func incrVecMulC64(a, b, incr []complex64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += a[i] * v
	}
	return nil
}

func incrVecMulC64Masked(a, b, incr []complex64, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] * v
		}
	}
	return nil
}

func incrVecMulC128(a, b, incr []complex128) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += a[i] * v
	}
	return nil
}

func incrVecMulC128Masked(a, b, incr []complex128, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] * v
		}
	}
	return nil
}

/* incr Div */

func incrVecDivI(a, b, incr []int) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	var errs errorIndices
	for i, v := range b {
		if v == int(0) {
			errs = append(errs, i)
			incr[i] = 0
			continue
		}
		incr[i] += a[i] / v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrVecDivIMasked(a, b, incr []int, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	var errs errorIndices
	for i, v := range b {
		if !mask[i] {
			if v == int(0) {
				errs = append(errs, i)
				incr[i] = 0
				continue
			}
			incr[i] += a[i] / v
		}
	}

	if errs != nil {
		return errs
	}
	return nil
}

func incrVecDivI8(a, b, incr []int8) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	var errs errorIndices
	for i, v := range b {
		if v == int8(0) {
			errs = append(errs, i)
			incr[i] = 0
			continue
		}
		incr[i] += a[i] / v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrVecDivI8Masked(a, b, incr []int8, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	var errs errorIndices
	for i, v := range b {
		if !mask[i] {
			if v == int8(0) {
				errs = append(errs, i)
				incr[i] = 0
				continue
			}
			incr[i] += a[i] / v
		}
	}

	if errs != nil {
		return errs
	}
	return nil
}

func incrVecDivI16(a, b, incr []int16) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	var errs errorIndices
	for i, v := range b {
		if v == int16(0) {
			errs = append(errs, i)
			incr[i] = 0
			continue
		}
		incr[i] += a[i] / v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrVecDivI16Masked(a, b, incr []int16, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	var errs errorIndices
	for i, v := range b {
		if !mask[i] {
			if v == int16(0) {
				errs = append(errs, i)
				incr[i] = 0
				continue
			}
			incr[i] += a[i] / v
		}
	}

	if errs != nil {
		return errs
	}
	return nil
}

func incrVecDivI32(a, b, incr []int32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	var errs errorIndices
	for i, v := range b {
		if v == int32(0) {
			errs = append(errs, i)
			incr[i] = 0
			continue
		}
		incr[i] += a[i] / v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrVecDivI32Masked(a, b, incr []int32, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	var errs errorIndices
	for i, v := range b {
		if !mask[i] {
			if v == int32(0) {
				errs = append(errs, i)
				incr[i] = 0
				continue
			}
			incr[i] += a[i] / v
		}
	}

	if errs != nil {
		return errs
	}
	return nil
}

func incrVecDivI64(a, b, incr []int64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	var errs errorIndices
	for i, v := range b {
		if v == int64(0) {
			errs = append(errs, i)
			incr[i] = 0
			continue
		}
		incr[i] += a[i] / v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrVecDivI64Masked(a, b, incr []int64, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	var errs errorIndices
	for i, v := range b {
		if !mask[i] {
			if v == int64(0) {
				errs = append(errs, i)
				incr[i] = 0
				continue
			}
			incr[i] += a[i] / v
		}
	}

	if errs != nil {
		return errs
	}
	return nil
}

func incrVecDivU(a, b, incr []uint) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	var errs errorIndices
	for i, v := range b {
		if v == uint(0) {
			errs = append(errs, i)
			incr[i] = 0
			continue
		}
		incr[i] += a[i] / v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrVecDivUMasked(a, b, incr []uint, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	var errs errorIndices
	for i, v := range b {
		if !mask[i] {
			if v == uint(0) {
				errs = append(errs, i)
				incr[i] = 0
				continue
			}
			incr[i] += a[i] / v
		}
	}

	if errs != nil {
		return errs
	}
	return nil
}

func incrVecDivU8(a, b, incr []uint8) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	var errs errorIndices
	for i, v := range b {
		if v == uint8(0) {
			errs = append(errs, i)
			incr[i] = 0
			continue
		}
		incr[i] += a[i] / v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrVecDivU8Masked(a, b, incr []uint8, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	var errs errorIndices
	for i, v := range b {
		if !mask[i] {
			if v == uint8(0) {
				errs = append(errs, i)
				incr[i] = 0
				continue
			}
			incr[i] += a[i] / v
		}
	}

	if errs != nil {
		return errs
	}
	return nil
}

func incrVecDivU16(a, b, incr []uint16) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	var errs errorIndices
	for i, v := range b {
		if v == uint16(0) {
			errs = append(errs, i)
			incr[i] = 0
			continue
		}
		incr[i] += a[i] / v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrVecDivU16Masked(a, b, incr []uint16, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	var errs errorIndices
	for i, v := range b {
		if !mask[i] {
			if v == uint16(0) {
				errs = append(errs, i)
				incr[i] = 0
				continue
			}
			incr[i] += a[i] / v
		}
	}

	if errs != nil {
		return errs
	}
	return nil
}

func incrVecDivU32(a, b, incr []uint32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	var errs errorIndices
	for i, v := range b {
		if v == uint32(0) {
			errs = append(errs, i)
			incr[i] = 0
			continue
		}
		incr[i] += a[i] / v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrVecDivU32Masked(a, b, incr []uint32, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	var errs errorIndices
	for i, v := range b {
		if !mask[i] {
			if v == uint32(0) {
				errs = append(errs, i)
				incr[i] = 0
				continue
			}
			incr[i] += a[i] / v
		}
	}

	if errs != nil {
		return errs
	}
	return nil
}

func incrVecDivU64(a, b, incr []uint64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	var errs errorIndices
	for i, v := range b {
		if v == uint64(0) {
			errs = append(errs, i)
			incr[i] = 0
			continue
		}
		incr[i] += a[i] / v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrVecDivU64Masked(a, b, incr []uint64, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	var errs errorIndices
	for i, v := range b {
		if !mask[i] {
			if v == uint64(0) {
				errs = append(errs, i)
				incr[i] = 0
				continue
			}
			incr[i] += a[i] / v
		}
	}

	if errs != nil {
		return errs
	}
	return nil
}

func incrVecDivF32(a, b, incr []float32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	vecf32.IncrDiv(a, b, incr)
	return nil
}

func incrVecDivF32Masked(a, b, incr []float32, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	var errs errorIndices
	for i, v := range b {
		if !mask[i] {
			if v == float32(0) {
				errs = append(errs, i)
				incr[i] = 0
				continue
			}
			incr[i] += a[i] / v
		}
	}

	if errs != nil {
		return errs
	}
	return nil
}

func incrVecDivF64(a, b, incr []float64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	vecf64.IncrDiv(a, b, incr)
	return nil
}

func incrVecDivF64Masked(a, b, incr []float64, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	var errs errorIndices
	for i, v := range b {
		if !mask[i] {
			if v == float64(0) {
				errs = append(errs, i)
				incr[i] = 0
				continue
			}
			incr[i] += a[i] / v
		}
	}

	if errs != nil {
		return errs
	}
	return nil
}

func incrVecDivC64(a, b, incr []complex64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += a[i] / v
	}
	return nil
}

func incrVecDivC64Masked(a, b, incr []complex64, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] / v
		}
	}
	return nil
}

func incrVecDivC128(a, b, incr []complex128) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += a[i] / v
	}
	return nil
}

func incrVecDivC128Masked(a, b, incr []complex128, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += a[i] / v
		}
	}
	return nil
}

/* incr Pow */

func incrVecPowI(a, b, incr []int) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += int(math.Pow(float64(a[i]), float64(v)))
	}
	return nil
}

func incrVecPowIMasked(a, b, incr []int, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += int(math.Pow(float64(a[i]), float64(v)))
		}
	}

	return nil
}

func incrVecPowI8(a, b, incr []int8) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += int8(math.Pow(float64(a[i]), float64(v)))
	}
	return nil
}

func incrVecPowI8Masked(a, b, incr []int8, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += int8(math.Pow(float64(a[i]), float64(v)))
		}
	}

	return nil
}

func incrVecPowI16(a, b, incr []int16) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += int16(math.Pow(float64(a[i]), float64(v)))
	}
	return nil
}

func incrVecPowI16Masked(a, b, incr []int16, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += int16(math.Pow(float64(a[i]), float64(v)))
		}
	}

	return nil
}

func incrVecPowI32(a, b, incr []int32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += int32(math.Pow(float64(a[i]), float64(v)))
	}
	return nil
}

func incrVecPowI32Masked(a, b, incr []int32, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += int32(math.Pow(float64(a[i]), float64(v)))
		}
	}

	return nil
}

func incrVecPowI64(a, b, incr []int64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += int64(math.Pow(float64(a[i]), float64(v)))
	}
	return nil
}

func incrVecPowI64Masked(a, b, incr []int64, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += int64(math.Pow(float64(a[i]), float64(v)))
		}
	}

	return nil
}

func incrVecPowU(a, b, incr []uint) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += uint(math.Pow(float64(a[i]), float64(v)))
	}
	return nil
}

func incrVecPowUMasked(a, b, incr []uint, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += uint(math.Pow(float64(a[i]), float64(v)))
		}
	}

	return nil
}

func incrVecPowU8(a, b, incr []uint8) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += uint8(math.Pow(float64(a[i]), float64(v)))
	}
	return nil
}

func incrVecPowU8Masked(a, b, incr []uint8, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += uint8(math.Pow(float64(a[i]), float64(v)))
		}
	}

	return nil
}

func incrVecPowU16(a, b, incr []uint16) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += uint16(math.Pow(float64(a[i]), float64(v)))
	}
	return nil
}

func incrVecPowU16Masked(a, b, incr []uint16, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += uint16(math.Pow(float64(a[i]), float64(v)))
		}
	}

	return nil
}

func incrVecPowU32(a, b, incr []uint32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += uint32(math.Pow(float64(a[i]), float64(v)))
	}
	return nil
}

func incrVecPowU32Masked(a, b, incr []uint32, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += uint32(math.Pow(float64(a[i]), float64(v)))
		}
	}

	return nil
}

func incrVecPowU64(a, b, incr []uint64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += uint64(math.Pow(float64(a[i]), float64(v)))
	}
	return nil
}

func incrVecPowU64Masked(a, b, incr []uint64, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += uint64(math.Pow(float64(a[i]), float64(v)))
		}
	}

	return nil
}

func incrVecPowF32(a, b, incr []float32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	vecf32.IncrPow(a, b, incr)
	return nil
}

func incrVecPowF32Masked(a, b, incr []float32, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += float32(math.Pow(float64(a[i]), float64(v)))
		}
	}

	return nil
}

func incrVecPowF64(a, b, incr []float64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	vecf64.IncrPow(a, b, incr)
	return nil
}

func incrVecPowF64Masked(a, b, incr []float64, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += float64(math.Pow(float64(a[i]), float64(v)))
		}
	}

	return nil
}

func incrVecPowC64(a, b, incr []complex64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += complex64(cmplx.Pow(complex128(a[i]), complex128(v)))
	}
	return nil
}

func incrVecPowC64Masked(a, b, incr []complex64, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += complex64(cmplx.Pow(complex128(a[i]), complex128(v)))
		}
	}
	return nil
}

func incrVecPowC128(a, b, incr []complex128) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		incr[i] += cmplx.Pow(a[i], v)
	}
	return nil
}

func incrVecPowC128Masked(a, b, incr []complex128, mask []bool) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b), len(mask))
	}

	a = a[:]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	for i, v := range b {
		if !mask[i] {
			incr[i] += cmplx.Pow(a[i], v)
		}
	}
	return nil
}

/* Trans */

func transI(a []int, b int) error {
	for i, v := range a {
		a[i] = v + b
	}
	return nil
}

func transIMasked(a []int, b int, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v + b
		}
	}
	return nil
}

func transI8(a []int8, b int8) error {
	for i, v := range a {
		a[i] = v + b
	}
	return nil
}

func transI8Masked(a []int8, b int8, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v + b
		}
	}
	return nil
}

func transI16(a []int16, b int16) error {
	for i, v := range a {
		a[i] = v + b
	}
	return nil
}

func transI16Masked(a []int16, b int16, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v + b
		}
	}
	return nil
}

func transI32(a []int32, b int32) error {
	for i, v := range a {
		a[i] = v + b
	}
	return nil
}

func transI32Masked(a []int32, b int32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v + b
		}
	}
	return nil
}

func transI64(a []int64, b int64) error {
	for i, v := range a {
		a[i] = v + b
	}
	return nil
}

func transI64Masked(a []int64, b int64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v + b
		}
	}
	return nil
}

func transU(a []uint, b uint) error {
	for i, v := range a {
		a[i] = v + b
	}
	return nil
}

func transUMasked(a []uint, b uint, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v + b
		}
	}
	return nil
}

func transU8(a []uint8, b uint8) error {
	for i, v := range a {
		a[i] = v + b
	}
	return nil
}

func transU8Masked(a []uint8, b uint8, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v + b
		}
	}
	return nil
}

func transU16(a []uint16, b uint16) error {
	for i, v := range a {
		a[i] = v + b
	}
	return nil
}

func transU16Masked(a []uint16, b uint16, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v + b
		}
	}
	return nil
}

func transU32(a []uint32, b uint32) error {
	for i, v := range a {
		a[i] = v + b
	}
	return nil
}

func transU32Masked(a []uint32, b uint32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v + b
		}
	}
	return nil
}

func transU64(a []uint64, b uint64) error {
	for i, v := range a {
		a[i] = v + b
	}
	return nil
}

func transU64Masked(a []uint64, b uint64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v + b
		}
	}
	return nil
}

func transF32(a []float32, b float32) error {
	vecf32.Trans(a, b)
	return nil
}

func transF32Masked(a []float32, b float32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v + b
		}
	}
	return nil
}

func transF64(a []float64, b float64) error {
	vecf64.Trans(a, b)
	return nil
}

func transF64Masked(a []float64, b float64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v + b
		}
	}
	return nil
}

func transC64(a []complex64, b complex64) error {
	for i, v := range a {
		a[i] = v + b
	}
	return nil
}

func transC64Masked(a []complex64, b complex64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v + b
		}
	}
	return nil
}

func transC128(a []complex128, b complex128) error {
	for i, v := range a {
		a[i] = v + b
	}
	return nil
}

func transC128Masked(a []complex128, b complex128, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v + b
		}
	}
	return nil
}

/* TransInv */

func transinvI(a []int, b int) error {
	for i, v := range a {
		a[i] = v - b
	}
	return nil
}

func transinvIMasked(a []int, b int, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v - b
		}
	}
	return nil
}

func transinvI8(a []int8, b int8) error {
	for i, v := range a {
		a[i] = v - b
	}
	return nil
}

func transinvI8Masked(a []int8, b int8, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v - b
		}
	}
	return nil
}

func transinvI16(a []int16, b int16) error {
	for i, v := range a {
		a[i] = v - b
	}
	return nil
}

func transinvI16Masked(a []int16, b int16, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v - b
		}
	}
	return nil
}

func transinvI32(a []int32, b int32) error {
	for i, v := range a {
		a[i] = v - b
	}
	return nil
}

func transinvI32Masked(a []int32, b int32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v - b
		}
	}
	return nil
}

func transinvI64(a []int64, b int64) error {
	for i, v := range a {
		a[i] = v - b
	}
	return nil
}

func transinvI64Masked(a []int64, b int64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v - b
		}
	}
	return nil
}

func transinvU(a []uint, b uint) error {
	for i, v := range a {
		a[i] = v - b
	}
	return nil
}

func transinvUMasked(a []uint, b uint, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v - b
		}
	}
	return nil
}

func transinvU8(a []uint8, b uint8) error {
	for i, v := range a {
		a[i] = v - b
	}
	return nil
}

func transinvU8Masked(a []uint8, b uint8, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v - b
		}
	}
	return nil
}

func transinvU16(a []uint16, b uint16) error {
	for i, v := range a {
		a[i] = v - b
	}
	return nil
}

func transinvU16Masked(a []uint16, b uint16, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v - b
		}
	}
	return nil
}

func transinvU32(a []uint32, b uint32) error {
	for i, v := range a {
		a[i] = v - b
	}
	return nil
}

func transinvU32Masked(a []uint32, b uint32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v - b
		}
	}
	return nil
}

func transinvU64(a []uint64, b uint64) error {
	for i, v := range a {
		a[i] = v - b
	}
	return nil
}

func transinvU64Masked(a []uint64, b uint64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v - b
		}
	}
	return nil
}

func transinvF32(a []float32, b float32) error {
	vecf32.TransInv(a, b)
	return nil
}

func transinvF32Masked(a []float32, b float32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v - b
		}
	}
	return nil
}

func transinvF64(a []float64, b float64) error {
	vecf64.TransInv(a, b)
	return nil
}

func transinvF64Masked(a []float64, b float64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v - b
		}
	}
	return nil
}

func transinvC64(a []complex64, b complex64) error {
	for i, v := range a {
		a[i] = v - b
	}
	return nil
}

func transinvC64Masked(a []complex64, b complex64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v - b
		}
	}
	return nil
}

func transinvC128(a []complex128, b complex128) error {
	for i, v := range a {
		a[i] = v - b
	}
	return nil
}

func transinvC128Masked(a []complex128, b complex128, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v - b
		}
	}
	return nil
}

/* TransInvR */

func transinvrI(a []int, b int) error {
	for i, v := range a {
		a[i] = b - v
	}
	return nil
}

func transinvrIMasked(a []int, b int, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = b - v
		}
	}
	return nil
}

func transinvrI8(a []int8, b int8) error {
	for i, v := range a {
		a[i] = b - v
	}
	return nil
}

func transinvrI8Masked(a []int8, b int8, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = b - v
		}
	}
	return nil
}

func transinvrI16(a []int16, b int16) error {
	for i, v := range a {
		a[i] = b - v
	}
	return nil
}

func transinvrI16Masked(a []int16, b int16, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = b - v
		}
	}
	return nil
}

func transinvrI32(a []int32, b int32) error {
	for i, v := range a {
		a[i] = b - v
	}
	return nil
}

func transinvrI32Masked(a []int32, b int32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = b - v
		}
	}
	return nil
}

func transinvrI64(a []int64, b int64) error {
	for i, v := range a {
		a[i] = b - v
	}
	return nil
}

func transinvrI64Masked(a []int64, b int64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = b - v
		}
	}
	return nil
}

func transinvrU(a []uint, b uint) error {
	for i, v := range a {
		a[i] = b - v
	}
	return nil
}

func transinvrUMasked(a []uint, b uint, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = b - v
		}
	}
	return nil
}

func transinvrU8(a []uint8, b uint8) error {
	for i, v := range a {
		a[i] = b - v
	}
	return nil
}

func transinvrU8Masked(a []uint8, b uint8, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = b - v
		}
	}
	return nil
}

func transinvrU16(a []uint16, b uint16) error {
	for i, v := range a {
		a[i] = b - v
	}
	return nil
}

func transinvrU16Masked(a []uint16, b uint16, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = b - v
		}
	}
	return nil
}

func transinvrU32(a []uint32, b uint32) error {
	for i, v := range a {
		a[i] = b - v
	}
	return nil
}

func transinvrU32Masked(a []uint32, b uint32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = b - v
		}
	}
	return nil
}

func transinvrU64(a []uint64, b uint64) error {
	for i, v := range a {
		a[i] = b - v
	}
	return nil
}

func transinvrU64Masked(a []uint64, b uint64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = b - v
		}
	}
	return nil
}

func transinvrF32(a []float32, b float32) error {
	vecf32.TransInvR(a, b)
	return nil
}

func transinvrF32Masked(a []float32, b float32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = b - v
		}
	}
	return nil
}

func transinvrF64(a []float64, b float64) error {
	vecf64.TransInvR(a, b)
	return nil
}

func transinvrF64Masked(a []float64, b float64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = b - v
		}
	}
	return nil
}

func transinvrC64(a []complex64, b complex64) error {
	for i, v := range a {
		a[i] = b - v
	}
	return nil
}

func transinvrC64Masked(a []complex64, b complex64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = b - v
		}
	}
	return nil
}

func transinvrC128(a []complex128, b complex128) error {
	for i, v := range a {
		a[i] = b - v
	}
	return nil
}

func transinvrC128Masked(a []complex128, b complex128, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = b - v
		}
	}
	return nil
}

/* Scale */

func scaleI(a []int, b int) error {
	for i, v := range a {
		a[i] = v * b
	}
	return nil
}

func scaleIMasked(a []int, b int, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v * b
		}
	}
	return nil
}

func scaleI8(a []int8, b int8) error {
	for i, v := range a {
		a[i] = v * b
	}
	return nil
}

func scaleI8Masked(a []int8, b int8, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v * b
		}
	}
	return nil
}

func scaleI16(a []int16, b int16) error {
	for i, v := range a {
		a[i] = v * b
	}
	return nil
}

func scaleI16Masked(a []int16, b int16, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v * b
		}
	}
	return nil
}

func scaleI32(a []int32, b int32) error {
	for i, v := range a {
		a[i] = v * b
	}
	return nil
}

func scaleI32Masked(a []int32, b int32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v * b
		}
	}
	return nil
}

func scaleI64(a []int64, b int64) error {
	for i, v := range a {
		a[i] = v * b
	}
	return nil
}

func scaleI64Masked(a []int64, b int64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v * b
		}
	}
	return nil
}

func scaleU(a []uint, b uint) error {
	for i, v := range a {
		a[i] = v * b
	}
	return nil
}

func scaleUMasked(a []uint, b uint, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v * b
		}
	}
	return nil
}

func scaleU8(a []uint8, b uint8) error {
	for i, v := range a {
		a[i] = v * b
	}
	return nil
}

func scaleU8Masked(a []uint8, b uint8, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v * b
		}
	}
	return nil
}

func scaleU16(a []uint16, b uint16) error {
	for i, v := range a {
		a[i] = v * b
	}
	return nil
}

func scaleU16Masked(a []uint16, b uint16, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v * b
		}
	}
	return nil
}

func scaleU32(a []uint32, b uint32) error {
	for i, v := range a {
		a[i] = v * b
	}
	return nil
}

func scaleU32Masked(a []uint32, b uint32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v * b
		}
	}
	return nil
}

func scaleU64(a []uint64, b uint64) error {
	for i, v := range a {
		a[i] = v * b
	}
	return nil
}

func scaleU64Masked(a []uint64, b uint64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v * b
		}
	}
	return nil
}

func scaleF32(a []float32, b float32) error {
	vecf32.Scale(a, b)
	return nil
}

func scaleF32Masked(a []float32, b float32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v * b
		}
	}
	return nil
}

func scaleF64(a []float64, b float64) error {
	vecf64.Scale(a, b)
	return nil
}

func scaleF64Masked(a []float64, b float64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v * b
		}
	}
	return nil
}

func scaleC64(a []complex64, b complex64) error {
	for i, v := range a {
		a[i] = v * b
	}
	return nil
}

func scaleC64Masked(a []complex64, b complex64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v * b
		}
	}
	return nil
}

func scaleC128(a []complex128, b complex128) error {
	for i, v := range a {
		a[i] = v * b
	}
	return nil
}

func scaleC128Masked(a []complex128, b complex128, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v * b
		}
	}
	return nil
}

/* ScaleInv */

func scaleinvI(a []int, b int) error {
	if b == int(0) {
		return errors.New(div0General)
	}
	for i, v := range a {
		a[i] = v / b
	}
	return nil
}

func scaleinvIMasked(a []int, b int, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	if b == int(0) {
		return errors.New(div0General)
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v / b
		}
	}
	return nil
}

func scaleinvI8(a []int8, b int8) error {
	if b == int8(0) {
		return errors.New(div0General)
	}
	for i, v := range a {
		a[i] = v / b
	}
	return nil
}

func scaleinvI8Masked(a []int8, b int8, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	if b == int8(0) {
		return errors.New(div0General)
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v / b
		}
	}
	return nil
}

func scaleinvI16(a []int16, b int16) error {
	if b == int16(0) {
		return errors.New(div0General)
	}
	for i, v := range a {
		a[i] = v / b
	}
	return nil
}

func scaleinvI16Masked(a []int16, b int16, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	if b == int16(0) {
		return errors.New(div0General)
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v / b
		}
	}
	return nil
}

func scaleinvI32(a []int32, b int32) error {
	if b == int32(0) {
		return errors.New(div0General)
	}
	for i, v := range a {
		a[i] = v / b
	}
	return nil
}

func scaleinvI32Masked(a []int32, b int32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	if b == int32(0) {
		return errors.New(div0General)
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v / b
		}
	}
	return nil
}

func scaleinvI64(a []int64, b int64) error {
	if b == int64(0) {
		return errors.New(div0General)
	}
	for i, v := range a {
		a[i] = v / b
	}
	return nil
}

func scaleinvI64Masked(a []int64, b int64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	if b == int64(0) {
		return errors.New(div0General)
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v / b
		}
	}
	return nil
}

func scaleinvU(a []uint, b uint) error {
	if b == uint(0) {
		return errors.New(div0General)
	}
	for i, v := range a {
		a[i] = v / b
	}
	return nil
}

func scaleinvUMasked(a []uint, b uint, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	if b == uint(0) {
		return errors.New(div0General)
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v / b
		}
	}
	return nil
}

func scaleinvU8(a []uint8, b uint8) error {
	if b == uint8(0) {
		return errors.New(div0General)
	}
	for i, v := range a {
		a[i] = v / b
	}
	return nil
}

func scaleinvU8Masked(a []uint8, b uint8, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	if b == uint8(0) {
		return errors.New(div0General)
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v / b
		}
	}
	return nil
}

func scaleinvU16(a []uint16, b uint16) error {
	if b == uint16(0) {
		return errors.New(div0General)
	}
	for i, v := range a {
		a[i] = v / b
	}
	return nil
}

func scaleinvU16Masked(a []uint16, b uint16, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	if b == uint16(0) {
		return errors.New(div0General)
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v / b
		}
	}
	return nil
}

func scaleinvU32(a []uint32, b uint32) error {
	if b == uint32(0) {
		return errors.New(div0General)
	}
	for i, v := range a {
		a[i] = v / b
	}
	return nil
}

func scaleinvU32Masked(a []uint32, b uint32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	if b == uint32(0) {
		return errors.New(div0General)
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v / b
		}
	}
	return nil
}

func scaleinvU64(a []uint64, b uint64) error {
	if b == uint64(0) {
		return errors.New(div0General)
	}
	for i, v := range a {
		a[i] = v / b
	}
	return nil
}

func scaleinvU64Masked(a []uint64, b uint64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	if b == uint64(0) {
		return errors.New(div0General)
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v / b
		}
	}
	return nil
}

func scaleinvF32(a []float32, b float32) error {
	vecf32.ScaleInv(a, b)
	return nil
}

func scaleinvF32Masked(a []float32, b float32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v / b
		}
	}
	return nil
}

func scaleinvF64(a []float64, b float64) error {
	vecf64.ScaleInv(a, b)
	return nil
}

func scaleinvF64Masked(a []float64, b float64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v / b
		}
	}
	return nil
}

func scaleinvC64(a []complex64, b complex64) error {
	for i, v := range a {
		a[i] = v / b
	}
	return nil
}

func scaleinvC64Masked(a []complex64, b complex64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v / b
		}
	}
	return nil
}

func scaleinvC128(a []complex128, b complex128) error {
	for i, v := range a {
		a[i] = v / b
	}
	return nil
}

func scaleinvC128Masked(a []complex128, b complex128, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = v / b
		}
	}
	return nil
}

/* ScaleInvR */

func scaleinvrI(a []int, b int) error {
	if b == int(0) {
		return errors.New(div0General)
	}
	var errs errorIndices
	for i, v := range a {
		if v == int(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}
		a[i] = b / v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func scaleinvrIMasked(a []int, b int, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	var errs errorIndices
	for i, v := range a {
		if !mask[i] {
			if v == int(0) {
				errs = append(errs, i)
				a[i] = 0
				continue
			}
			a[i] = b / v
		}
	}
	if errs != nil {
		return errs
	}
	return nil
}

func scaleinvrI8(a []int8, b int8) error {
	if b == int8(0) {
		return errors.New(div0General)
	}
	var errs errorIndices
	for i, v := range a {
		if v == int8(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}
		a[i] = b / v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func scaleinvrI8Masked(a []int8, b int8, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	var errs errorIndices
	for i, v := range a {
		if !mask[i] {
			if v == int8(0) {
				errs = append(errs, i)
				a[i] = 0
				continue
			}
			a[i] = b / v
		}
	}
	if errs != nil {
		return errs
	}
	return nil
}

func scaleinvrI16(a []int16, b int16) error {
	if b == int16(0) {
		return errors.New(div0General)
	}
	var errs errorIndices
	for i, v := range a {
		if v == int16(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}
		a[i] = b / v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func scaleinvrI16Masked(a []int16, b int16, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	var errs errorIndices
	for i, v := range a {
		if !mask[i] {
			if v == int16(0) {
				errs = append(errs, i)
				a[i] = 0
				continue
			}
			a[i] = b / v
		}
	}
	if errs != nil {
		return errs
	}
	return nil
}

func scaleinvrI32(a []int32, b int32) error {
	if b == int32(0) {
		return errors.New(div0General)
	}
	var errs errorIndices
	for i, v := range a {
		if v == int32(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}
		a[i] = b / v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func scaleinvrI32Masked(a []int32, b int32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	var errs errorIndices
	for i, v := range a {
		if !mask[i] {
			if v == int32(0) {
				errs = append(errs, i)
				a[i] = 0
				continue
			}
			a[i] = b / v
		}
	}
	if errs != nil {
		return errs
	}
	return nil
}

func scaleinvrI64(a []int64, b int64) error {
	if b == int64(0) {
		return errors.New(div0General)
	}
	var errs errorIndices
	for i, v := range a {
		if v == int64(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}
		a[i] = b / v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func scaleinvrI64Masked(a []int64, b int64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	var errs errorIndices
	for i, v := range a {
		if !mask[i] {
			if v == int64(0) {
				errs = append(errs, i)
				a[i] = 0
				continue
			}
			a[i] = b / v
		}
	}
	if errs != nil {
		return errs
	}
	return nil
}

func scaleinvrU(a []uint, b uint) error {
	if b == uint(0) {
		return errors.New(div0General)
	}
	var errs errorIndices
	for i, v := range a {
		if v == uint(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}
		a[i] = b / v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func scaleinvrUMasked(a []uint, b uint, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	var errs errorIndices
	for i, v := range a {
		if !mask[i] {
			if v == uint(0) {
				errs = append(errs, i)
				a[i] = 0
				continue
			}
			a[i] = b / v
		}
	}
	if errs != nil {
		return errs
	}
	return nil
}

func scaleinvrU8(a []uint8, b uint8) error {
	if b == uint8(0) {
		return errors.New(div0General)
	}
	var errs errorIndices
	for i, v := range a {
		if v == uint8(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}
		a[i] = b / v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func scaleinvrU8Masked(a []uint8, b uint8, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	var errs errorIndices
	for i, v := range a {
		if !mask[i] {
			if v == uint8(0) {
				errs = append(errs, i)
				a[i] = 0
				continue
			}
			a[i] = b / v
		}
	}
	if errs != nil {
		return errs
	}
	return nil
}

func scaleinvrU16(a []uint16, b uint16) error {
	if b == uint16(0) {
		return errors.New(div0General)
	}
	var errs errorIndices
	for i, v := range a {
		if v == uint16(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}
		a[i] = b / v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func scaleinvrU16Masked(a []uint16, b uint16, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	var errs errorIndices
	for i, v := range a {
		if !mask[i] {
			if v == uint16(0) {
				errs = append(errs, i)
				a[i] = 0
				continue
			}
			a[i] = b / v
		}
	}
	if errs != nil {
		return errs
	}
	return nil
}

func scaleinvrU32(a []uint32, b uint32) error {
	if b == uint32(0) {
		return errors.New(div0General)
	}
	var errs errorIndices
	for i, v := range a {
		if v == uint32(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}
		a[i] = b / v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func scaleinvrU32Masked(a []uint32, b uint32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	var errs errorIndices
	for i, v := range a {
		if !mask[i] {
			if v == uint32(0) {
				errs = append(errs, i)
				a[i] = 0
				continue
			}
			a[i] = b / v
		}
	}
	if errs != nil {
		return errs
	}
	return nil
}

func scaleinvrU64(a []uint64, b uint64) error {
	if b == uint64(0) {
		return errors.New(div0General)
	}
	var errs errorIndices
	for i, v := range a {
		if v == uint64(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}
		a[i] = b / v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func scaleinvrU64Masked(a []uint64, b uint64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	var errs errorIndices
	for i, v := range a {
		if !mask[i] {
			if v == uint64(0) {
				errs = append(errs, i)
				a[i] = 0
				continue
			}
			a[i] = b / v
		}
	}
	if errs != nil {
		return errs
	}
	return nil
}

func scaleinvrF32(a []float32, b float32) error {
	vecf32.ScaleInvR(a, b)
	return nil
}

func scaleinvrF32Masked(a []float32, b float32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	var errs errorIndices
	for i, v := range a {
		if !mask[i] {
			if v == float32(0) {
				errs = append(errs, i)
				a[i] = 0
				continue
			}
			a[i] = b / v
		}
	}
	if errs != nil {
		return errs
	}
	return nil
}

func scaleinvrF64(a []float64, b float64) error {
	vecf64.ScaleInvR(a, b)
	return nil
}

func scaleinvrF64Masked(a []float64, b float64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	var errs errorIndices
	for i, v := range a {
		if !mask[i] {
			if v == float64(0) {
				errs = append(errs, i)
				a[i] = 0
				continue
			}
			a[i] = b / v
		}
	}
	if errs != nil {
		return errs
	}
	return nil
}

func scaleinvrC64(a []complex64, b complex64) error {
	for i, v := range a {
		a[i] = b / v
	}
	return nil
}

func scaleinvrC64Masked(a []complex64, b complex64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = b / v
		}
	}
	return nil
}

func scaleinvrC128(a []complex128, b complex128) error {
	for i, v := range a {
		a[i] = b / v
	}
	return nil
}

func scaleinvrC128Masked(a []complex128, b complex128, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = b / v
		}
	}
	return nil
}

/* PowOf */

func powofI(a []int, b int) error {
	for i, v := range a {
		a[i] = int(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func powofIMasked(a []int, b int, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = int(math.Pow(float64(v), float64(b)))
		}
	}
	return nil
}

func powofI8(a []int8, b int8) error {
	for i, v := range a {
		a[i] = int8(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func powofI8Masked(a []int8, b int8, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = int8(math.Pow(float64(v), float64(b)))
		}
	}
	return nil
}

func powofI16(a []int16, b int16) error {
	for i, v := range a {
		a[i] = int16(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func powofI16Masked(a []int16, b int16, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = int16(math.Pow(float64(v), float64(b)))
		}
	}
	return nil
}

func powofI32(a []int32, b int32) error {
	for i, v := range a {
		a[i] = int32(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func powofI32Masked(a []int32, b int32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = int32(math.Pow(float64(v), float64(b)))
		}
	}
	return nil
}

func powofI64(a []int64, b int64) error {
	for i, v := range a {
		a[i] = int64(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func powofI64Masked(a []int64, b int64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = int64(math.Pow(float64(v), float64(b)))
		}
	}
	return nil
}

func powofU(a []uint, b uint) error {
	for i, v := range a {
		a[i] = uint(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func powofUMasked(a []uint, b uint, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = uint(math.Pow(float64(v), float64(b)))
		}
	}
	return nil
}

func powofU8(a []uint8, b uint8) error {
	for i, v := range a {
		a[i] = uint8(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func powofU8Masked(a []uint8, b uint8, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = uint8(math.Pow(float64(v), float64(b)))
		}
	}
	return nil
}

func powofU16(a []uint16, b uint16) error {
	for i, v := range a {
		a[i] = uint16(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func powofU16Masked(a []uint16, b uint16, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = uint16(math.Pow(float64(v), float64(b)))
		}
	}
	return nil
}

func powofU32(a []uint32, b uint32) error {
	for i, v := range a {
		a[i] = uint32(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func powofU32Masked(a []uint32, b uint32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = uint32(math.Pow(float64(v), float64(b)))
		}
	}
	return nil
}

func powofU64(a []uint64, b uint64) error {
	for i, v := range a {
		a[i] = uint64(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func powofU64Masked(a []uint64, b uint64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = uint64(math.Pow(float64(v), float64(b)))
		}
	}
	return nil
}

func powofF32(a []float32, b float32) error {
	vecf32.PowOf(a, b)
	return nil
}

func powofF32Masked(a []float32, b float32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = float32(math.Pow(float64(v), float64(b)))
		}
	}
	return nil
}

func powofF64(a []float64, b float64) error {
	vecf64.PowOf(a, b)
	return nil
}

func powofF64Masked(a []float64, b float64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = float64(math.Pow(float64(v), float64(b)))
		}
	}
	return nil
}

func powofC64(a []complex64, b complex64) error {
	for i, v := range a {
		a[i] = complex64(cmplx.Pow(complex128(v), complex128(b)))
	}
	return nil
}

func powofC64Masked(a []complex64, b complex64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = complex64(cmplx.Pow(complex128(v), complex128(b)))
		}
	}
	return nil
}

func powofC128(a []complex128, b complex128) error {
	for i, v := range a {
		a[i] = complex128(cmplx.Pow(complex128(v), complex128(b)))
	}
	return nil
}

func powofC128Masked(a []complex128, b complex128, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = complex128(cmplx.Pow(complex128(v), complex128(b)))
		}
	}
	return nil
}

/* PowOfR */

func powofrI(a []int, b int) error {
	for i, v := range a {
		a[i] = int(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func powofrIMasked(a []int, b int, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = int(math.Pow(float64(b), float64(v)))
		}
	}
	return nil
}

func powofrI8(a []int8, b int8) error {
	for i, v := range a {
		a[i] = int8(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func powofrI8Masked(a []int8, b int8, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = int8(math.Pow(float64(b), float64(v)))
		}
	}
	return nil
}

func powofrI16(a []int16, b int16) error {
	for i, v := range a {
		a[i] = int16(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func powofrI16Masked(a []int16, b int16, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = int16(math.Pow(float64(b), float64(v)))
		}
	}
	return nil
}

func powofrI32(a []int32, b int32) error {
	for i, v := range a {
		a[i] = int32(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func powofrI32Masked(a []int32, b int32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = int32(math.Pow(float64(b), float64(v)))
		}
	}
	return nil
}

func powofrI64(a []int64, b int64) error {
	for i, v := range a {
		a[i] = int64(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func powofrI64Masked(a []int64, b int64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = int64(math.Pow(float64(b), float64(v)))
		}
	}
	return nil
}

func powofrU(a []uint, b uint) error {
	for i, v := range a {
		a[i] = uint(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func powofrUMasked(a []uint, b uint, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = uint(math.Pow(float64(b), float64(v)))
		}
	}
	return nil
}

func powofrU8(a []uint8, b uint8) error {
	for i, v := range a {
		a[i] = uint8(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func powofrU8Masked(a []uint8, b uint8, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = uint8(math.Pow(float64(b), float64(v)))
		}
	}
	return nil
}

func powofrU16(a []uint16, b uint16) error {
	for i, v := range a {
		a[i] = uint16(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func powofrU16Masked(a []uint16, b uint16, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = uint16(math.Pow(float64(b), float64(v)))
		}
	}
	return nil
}

func powofrU32(a []uint32, b uint32) error {
	for i, v := range a {
		a[i] = uint32(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func powofrU32Masked(a []uint32, b uint32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = uint32(math.Pow(float64(b), float64(v)))
		}
	}
	return nil
}

func powofrU64(a []uint64, b uint64) error {
	for i, v := range a {
		a[i] = uint64(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func powofrU64Masked(a []uint64, b uint64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = uint64(math.Pow(float64(b), float64(v)))
		}
	}
	return nil
}

func powofrF32(a []float32, b float32) error {
	vecf32.PowOfR(a, b)
	return nil
}

func powofrF32Masked(a []float32, b float32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = float32(math.Pow(float64(b), float64(v)))
		}
	}
	return nil
}

func powofrF64(a []float64, b float64) error {
	vecf64.PowOfR(a, b)
	return nil
}

func powofrF64Masked(a []float64, b float64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = float64(math.Pow(float64(b), float64(v)))
		}
	}
	return nil
}

func powofrC64(a []complex64, b complex64) error {
	for i, v := range a {
		a[i] = complex64(cmplx.Pow(complex128(b), complex128(v)))
	}
	return nil
}

func powofrC64Masked(a []complex64, b complex64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = complex64(cmplx.Pow(complex128(b), complex128(v)))
		}
	}
	return nil
}

func powofrC128(a []complex128, b complex128) error {
	for i, v := range a {
		a[i] = complex128(cmplx.Pow(complex128(b), complex128(v)))
	}
	return nil
}

func powofrC128Masked(a []complex128, b complex128, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			a[i] = complex128(cmplx.Pow(complex128(b), complex128(v)))
		}
	}
	return nil
}

/* incr Trans */

func incrTransI(a, incr []int, b int) error {
	for i, v := range a {
		incr[i] += v + b
	}
	return nil
}

func incrTransIMasked(a, incr []int, b int, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v + b
		}
	}
	return nil
}

func incrTransI8(a, incr []int8, b int8) error {
	for i, v := range a {
		incr[i] += v + b
	}
	return nil
}

func incrTransI8Masked(a, incr []int8, b int8, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v + b
		}
	}
	return nil
}

func incrTransI16(a, incr []int16, b int16) error {
	for i, v := range a {
		incr[i] += v + b
	}
	return nil
}

func incrTransI16Masked(a, incr []int16, b int16, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v + b
		}
	}
	return nil
}

func incrTransI32(a, incr []int32, b int32) error {
	for i, v := range a {
		incr[i] += v + b
	}
	return nil
}

func incrTransI32Masked(a, incr []int32, b int32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v + b
		}
	}
	return nil
}

func incrTransI64(a, incr []int64, b int64) error {
	for i, v := range a {
		incr[i] += v + b
	}
	return nil
}

func incrTransI64Masked(a, incr []int64, b int64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v + b
		}
	}
	return nil
}

func incrTransU(a, incr []uint, b uint) error {
	for i, v := range a {
		incr[i] += v + b
	}
	return nil
}

func incrTransUMasked(a, incr []uint, b uint, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v + b
		}
	}
	return nil
}

func incrTransU8(a, incr []uint8, b uint8) error {
	for i, v := range a {
		incr[i] += v + b
	}
	return nil
}

func incrTransU8Masked(a, incr []uint8, b uint8, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v + b
		}
	}
	return nil
}

func incrTransU16(a, incr []uint16, b uint16) error {
	for i, v := range a {
		incr[i] += v + b
	}
	return nil
}

func incrTransU16Masked(a, incr []uint16, b uint16, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v + b
		}
	}
	return nil
}

func incrTransU32(a, incr []uint32, b uint32) error {
	for i, v := range a {
		incr[i] += v + b
	}
	return nil
}

func incrTransU32Masked(a, incr []uint32, b uint32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v + b
		}
	}
	return nil
}

func incrTransU64(a, incr []uint64, b uint64) error {
	for i, v := range a {
		incr[i] += v + b
	}
	return nil
}

func incrTransU64Masked(a, incr []uint64, b uint64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v + b
		}
	}
	return nil
}

func incrTransF32(a, incr []float32, b float32) error {
	vecf32.IncrTrans(a, b, incr)
	return nil
}

func incrTransF32Masked(a, incr []float32, b float32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v + b
		}
	}
	return nil
}

func incrTransF64(a, incr []float64, b float64) error {
	vecf64.IncrTrans(a, b, incr)
	return nil
}

func incrTransF64Masked(a, incr []float64, b float64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v + b
		}
	}
	return nil
}

func incrTransC64(a, incr []complex64, b complex64) error {
	for i, v := range a {
		incr[i] += v + b
	}
	return nil
}

func incrTransC64Masked(a, incr []complex64, b complex64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v + b
		}
	}
	return nil
}

func incrTransC128(a, incr []complex128, b complex128) error {
	for i, v := range a {
		incr[i] += v + b
	}
	return nil
}

func incrTransC128Masked(a, incr []complex128, b complex128, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v + b
		}
	}
	return nil
}

/* incr TransInv */

func incrTransInvI(a, incr []int, b int) error {
	for i, v := range a {
		incr[i] += v - b
	}
	return nil
}

func incrTransInvIMasked(a, incr []int, b int, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v - b
		}
	}
	return nil
}

func incrTransInvI8(a, incr []int8, b int8) error {
	for i, v := range a {
		incr[i] += v - b
	}
	return nil
}

func incrTransInvI8Masked(a, incr []int8, b int8, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v - b
		}
	}
	return nil
}

func incrTransInvI16(a, incr []int16, b int16) error {
	for i, v := range a {
		incr[i] += v - b
	}
	return nil
}

func incrTransInvI16Masked(a, incr []int16, b int16, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v - b
		}
	}
	return nil
}

func incrTransInvI32(a, incr []int32, b int32) error {
	for i, v := range a {
		incr[i] += v - b
	}
	return nil
}

func incrTransInvI32Masked(a, incr []int32, b int32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v - b
		}
	}
	return nil
}

func incrTransInvI64(a, incr []int64, b int64) error {
	for i, v := range a {
		incr[i] += v - b
	}
	return nil
}

func incrTransInvI64Masked(a, incr []int64, b int64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v - b
		}
	}
	return nil
}

func incrTransInvU(a, incr []uint, b uint) error {
	for i, v := range a {
		incr[i] += v - b
	}
	return nil
}

func incrTransInvUMasked(a, incr []uint, b uint, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v - b
		}
	}
	return nil
}

func incrTransInvU8(a, incr []uint8, b uint8) error {
	for i, v := range a {
		incr[i] += v - b
	}
	return nil
}

func incrTransInvU8Masked(a, incr []uint8, b uint8, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v - b
		}
	}
	return nil
}

func incrTransInvU16(a, incr []uint16, b uint16) error {
	for i, v := range a {
		incr[i] += v - b
	}
	return nil
}

func incrTransInvU16Masked(a, incr []uint16, b uint16, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v - b
		}
	}
	return nil
}

func incrTransInvU32(a, incr []uint32, b uint32) error {
	for i, v := range a {
		incr[i] += v - b
	}
	return nil
}

func incrTransInvU32Masked(a, incr []uint32, b uint32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v - b
		}
	}
	return nil
}

func incrTransInvU64(a, incr []uint64, b uint64) error {
	for i, v := range a {
		incr[i] += v - b
	}
	return nil
}

func incrTransInvU64Masked(a, incr []uint64, b uint64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v - b
		}
	}
	return nil
}

func incrTransInvF32(a, incr []float32, b float32) error {
	vecf32.IncrTransInv(a, b, incr)
	return nil
}

func incrTransInvF32Masked(a, incr []float32, b float32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v - b
		}
	}
	return nil
}

func incrTransInvF64(a, incr []float64, b float64) error {
	vecf64.IncrTransInv(a, b, incr)
	return nil
}

func incrTransInvF64Masked(a, incr []float64, b float64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v - b
		}
	}
	return nil
}

func incrTransInvC64(a, incr []complex64, b complex64) error {
	for i, v := range a {
		incr[i] += v - b
	}
	return nil
}

func incrTransInvC64Masked(a, incr []complex64, b complex64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v - b
		}
	}
	return nil
}

func incrTransInvC128(a, incr []complex128, b complex128) error {
	for i, v := range a {
		incr[i] += v - b
	}
	return nil
}

func incrTransInvC128Masked(a, incr []complex128, b complex128, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v - b
		}
	}
	return nil
}

/* incr TransInvR */

func incrTransInvRI(a, incr []int, b int) error {
	for i, v := range a {
		incr[i] += b - v
	}
	return nil
}

func incrTransInvRIMasked(a, incr []int, b int, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += b - v
		}
	}
	return nil
}

func incrTransInvRI8(a, incr []int8, b int8) error {
	for i, v := range a {
		incr[i] += b - v
	}
	return nil
}

func incrTransInvRI8Masked(a, incr []int8, b int8, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += b - v
		}
	}
	return nil
}

func incrTransInvRI16(a, incr []int16, b int16) error {
	for i, v := range a {
		incr[i] += b - v
	}
	return nil
}

func incrTransInvRI16Masked(a, incr []int16, b int16, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += b - v
		}
	}
	return nil
}

func incrTransInvRI32(a, incr []int32, b int32) error {
	for i, v := range a {
		incr[i] += b - v
	}
	return nil
}

func incrTransInvRI32Masked(a, incr []int32, b int32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += b - v
		}
	}
	return nil
}

func incrTransInvRI64(a, incr []int64, b int64) error {
	for i, v := range a {
		incr[i] += b - v
	}
	return nil
}

func incrTransInvRI64Masked(a, incr []int64, b int64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += b - v
		}
	}
	return nil
}

func incrTransInvRU(a, incr []uint, b uint) error {
	for i, v := range a {
		incr[i] += b - v
	}
	return nil
}

func incrTransInvRUMasked(a, incr []uint, b uint, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += b - v
		}
	}
	return nil
}

func incrTransInvRU8(a, incr []uint8, b uint8) error {
	for i, v := range a {
		incr[i] += b - v
	}
	return nil
}

func incrTransInvRU8Masked(a, incr []uint8, b uint8, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += b - v
		}
	}
	return nil
}

func incrTransInvRU16(a, incr []uint16, b uint16) error {
	for i, v := range a {
		incr[i] += b - v
	}
	return nil
}

func incrTransInvRU16Masked(a, incr []uint16, b uint16, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += b - v
		}
	}
	return nil
}

func incrTransInvRU32(a, incr []uint32, b uint32) error {
	for i, v := range a {
		incr[i] += b - v
	}
	return nil
}

func incrTransInvRU32Masked(a, incr []uint32, b uint32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += b - v
		}
	}
	return nil
}

func incrTransInvRU64(a, incr []uint64, b uint64) error {
	for i, v := range a {
		incr[i] += b - v
	}
	return nil
}

func incrTransInvRU64Masked(a, incr []uint64, b uint64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += b - v
		}
	}
	return nil
}

func incrTransInvRF32(a, incr []float32, b float32) error {
	vecf32.IncrTransInvR(a, b, incr)
	return nil
}

func incrTransInvRF32Masked(a, incr []float32, b float32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += b - v
		}
	}
	return nil
}

func incrTransInvRF64(a, incr []float64, b float64) error {
	vecf64.IncrTransInvR(a, b, incr)
	return nil
}

func incrTransInvRF64Masked(a, incr []float64, b float64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += b - v
		}
	}
	return nil
}

func incrTransInvRC64(a, incr []complex64, b complex64) error {
	for i, v := range a {
		incr[i] += b - v
	}
	return nil
}

func incrTransInvRC64Masked(a, incr []complex64, b complex64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += b - v
		}
	}
	return nil
}

func incrTransInvRC128(a, incr []complex128, b complex128) error {
	for i, v := range a {
		incr[i] += b - v
	}
	return nil
}

func incrTransInvRC128Masked(a, incr []complex128, b complex128, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += b - v
		}
	}
	return nil
}

/* incr Scale */

func incrScaleI(a, incr []int, b int) error {
	for i, v := range a {
		incr[i] += v * b
	}
	return nil
}

func incrScaleIMasked(a, incr []int, b int, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v * b
		}
	}
	return nil
}

func incrScaleI8(a, incr []int8, b int8) error {
	for i, v := range a {
		incr[i] += v * b
	}
	return nil
}

func incrScaleI8Masked(a, incr []int8, b int8, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v * b
		}
	}
	return nil
}

func incrScaleI16(a, incr []int16, b int16) error {
	for i, v := range a {
		incr[i] += v * b
	}
	return nil
}

func incrScaleI16Masked(a, incr []int16, b int16, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v * b
		}
	}
	return nil
}

func incrScaleI32(a, incr []int32, b int32) error {
	for i, v := range a {
		incr[i] += v * b
	}
	return nil
}

func incrScaleI32Masked(a, incr []int32, b int32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v * b
		}
	}
	return nil
}

func incrScaleI64(a, incr []int64, b int64) error {
	for i, v := range a {
		incr[i] += v * b
	}
	return nil
}

func incrScaleI64Masked(a, incr []int64, b int64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v * b
		}
	}
	return nil
}

func incrScaleU(a, incr []uint, b uint) error {
	for i, v := range a {
		incr[i] += v * b
	}
	return nil
}

func incrScaleUMasked(a, incr []uint, b uint, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v * b
		}
	}
	return nil
}

func incrScaleU8(a, incr []uint8, b uint8) error {
	for i, v := range a {
		incr[i] += v * b
	}
	return nil
}

func incrScaleU8Masked(a, incr []uint8, b uint8, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v * b
		}
	}
	return nil
}

func incrScaleU16(a, incr []uint16, b uint16) error {
	for i, v := range a {
		incr[i] += v * b
	}
	return nil
}

func incrScaleU16Masked(a, incr []uint16, b uint16, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v * b
		}
	}
	return nil
}

func incrScaleU32(a, incr []uint32, b uint32) error {
	for i, v := range a {
		incr[i] += v * b
	}
	return nil
}

func incrScaleU32Masked(a, incr []uint32, b uint32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v * b
		}
	}
	return nil
}

func incrScaleU64(a, incr []uint64, b uint64) error {
	for i, v := range a {
		incr[i] += v * b
	}
	return nil
}

func incrScaleU64Masked(a, incr []uint64, b uint64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v * b
		}
	}
	return nil
}

func incrScaleF32(a, incr []float32, b float32) error {
	vecf32.IncrScale(a, b, incr)
	return nil
}

func incrScaleF32Masked(a, incr []float32, b float32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v * b
		}
	}
	return nil
}

func incrScaleF64(a, incr []float64, b float64) error {
	vecf64.IncrScale(a, b, incr)
	return nil
}

func incrScaleF64Masked(a, incr []float64, b float64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v * b
		}
	}
	return nil
}

func incrScaleC64(a, incr []complex64, b complex64) error {
	for i, v := range a {
		incr[i] += v * b
	}
	return nil
}

func incrScaleC64Masked(a, incr []complex64, b complex64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v * b
		}
	}
	return nil
}

func incrScaleC128(a, incr []complex128, b complex128) error {
	for i, v := range a {
		incr[i] += v * b
	}
	return nil
}

func incrScaleC128Masked(a, incr []complex128, b complex128, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v * b
		}
	}
	return nil
}

/* incr ScaleInv */

func incrScaleInvI(a, incr []int, b int) error {
	if b == int(0) {
		return errors.New(div0General)
	}
	for i, v := range a {
		incr[i] += v / b
	}
	return nil
}

func incrScaleInvIMasked(a, incr []int, b int, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	if b == int(0) {
		return errors.New(div0General)
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v / b
		}
	}
	return nil
}

func incrScaleInvI8(a, incr []int8, b int8) error {
	if b == int8(0) {
		return errors.New(div0General)
	}
	for i, v := range a {
		incr[i] += v / b
	}
	return nil
}

func incrScaleInvI8Masked(a, incr []int8, b int8, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	if b == int8(0) {
		return errors.New(div0General)
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v / b
		}
	}
	return nil
}

func incrScaleInvI16(a, incr []int16, b int16) error {
	if b == int16(0) {
		return errors.New(div0General)
	}
	for i, v := range a {
		incr[i] += v / b
	}
	return nil
}

func incrScaleInvI16Masked(a, incr []int16, b int16, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	if b == int16(0) {
		return errors.New(div0General)
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v / b
		}
	}
	return nil
}

func incrScaleInvI32(a, incr []int32, b int32) error {
	if b == int32(0) {
		return errors.New(div0General)
	}
	for i, v := range a {
		incr[i] += v / b
	}
	return nil
}

func incrScaleInvI32Masked(a, incr []int32, b int32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	if b == int32(0) {
		return errors.New(div0General)
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v / b
		}
	}
	return nil
}

func incrScaleInvI64(a, incr []int64, b int64) error {
	if b == int64(0) {
		return errors.New(div0General)
	}
	for i, v := range a {
		incr[i] += v / b
	}
	return nil
}

func incrScaleInvI64Masked(a, incr []int64, b int64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	if b == int64(0) {
		return errors.New(div0General)
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v / b
		}
	}
	return nil
}

func incrScaleInvU(a, incr []uint, b uint) error {
	if b == uint(0) {
		return errors.New(div0General)
	}
	for i, v := range a {
		incr[i] += v / b
	}
	return nil
}

func incrScaleInvUMasked(a, incr []uint, b uint, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	if b == uint(0) {
		return errors.New(div0General)
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v / b
		}
	}
	return nil
}

func incrScaleInvU8(a, incr []uint8, b uint8) error {
	if b == uint8(0) {
		return errors.New(div0General)
	}
	for i, v := range a {
		incr[i] += v / b
	}
	return nil
}

func incrScaleInvU8Masked(a, incr []uint8, b uint8, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	if b == uint8(0) {
		return errors.New(div0General)
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v / b
		}
	}
	return nil
}

func incrScaleInvU16(a, incr []uint16, b uint16) error {
	if b == uint16(0) {
		return errors.New(div0General)
	}
	for i, v := range a {
		incr[i] += v / b
	}
	return nil
}

func incrScaleInvU16Masked(a, incr []uint16, b uint16, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	if b == uint16(0) {
		return errors.New(div0General)
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v / b
		}
	}
	return nil
}

func incrScaleInvU32(a, incr []uint32, b uint32) error {
	if b == uint32(0) {
		return errors.New(div0General)
	}
	for i, v := range a {
		incr[i] += v / b
	}
	return nil
}

func incrScaleInvU32Masked(a, incr []uint32, b uint32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	if b == uint32(0) {
		return errors.New(div0General)
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v / b
		}
	}
	return nil
}

func incrScaleInvU64(a, incr []uint64, b uint64) error {
	if b == uint64(0) {
		return errors.New(div0General)
	}
	for i, v := range a {
		incr[i] += v / b
	}
	return nil
}

func incrScaleInvU64Masked(a, incr []uint64, b uint64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	if b == uint64(0) {
		return errors.New(div0General)
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v / b
		}
	}
	return nil
}

func incrScaleInvF32(a, incr []float32, b float32) error {
	vecf32.IncrScaleInv(a, b, incr)
	return nil
}

func incrScaleInvF32Masked(a, incr []float32, b float32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v / b
		}
	}
	return nil
}

func incrScaleInvF64(a, incr []float64, b float64) error {
	vecf64.IncrScaleInv(a, b, incr)
	return nil
}

func incrScaleInvF64Masked(a, incr []float64, b float64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v / b
		}
	}
	return nil
}

func incrScaleInvC64(a, incr []complex64, b complex64) error {
	for i, v := range a {
		incr[i] += v / b
	}
	return nil
}

func incrScaleInvC64Masked(a, incr []complex64, b complex64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v / b
		}
	}
	return nil
}

func incrScaleInvC128(a, incr []complex128, b complex128) error {
	for i, v := range a {
		incr[i] += v / b
	}
	return nil
}

func incrScaleInvC128Masked(a, incr []complex128, b complex128, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += v / b
		}
	}
	return nil
}

/* incr ScaleInvR */

func incrScaleInvRI(a, incr []int, b int) error {
	if b == int(0) {
		return errors.New(div0General)
	}
	var errs errorIndices
	for i, v := range a {
		if v == int(0) {
			errs = append(errs, i)
			incr[i] = 0
			continue
		}
		incr[i] += b / v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrScaleInvRIMasked(a, incr []int, b int, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	var errs errorIndices
	for i, v := range a {
		if !mask[i] {
			if v == int(0) {
				errs = append(errs, i)
				incr[i] = 0
				continue
			}
			incr[i] += b / v
		}
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrScaleInvRI8(a, incr []int8, b int8) error {
	if b == int8(0) {
		return errors.New(div0General)
	}
	var errs errorIndices
	for i, v := range a {
		if v == int8(0) {
			errs = append(errs, i)
			incr[i] = 0
			continue
		}
		incr[i] += b / v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrScaleInvRI8Masked(a, incr []int8, b int8, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	var errs errorIndices
	for i, v := range a {
		if !mask[i] {
			if v == int8(0) {
				errs = append(errs, i)
				incr[i] = 0
				continue
			}
			incr[i] += b / v
		}
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrScaleInvRI16(a, incr []int16, b int16) error {
	if b == int16(0) {
		return errors.New(div0General)
	}
	var errs errorIndices
	for i, v := range a {
		if v == int16(0) {
			errs = append(errs, i)
			incr[i] = 0
			continue
		}
		incr[i] += b / v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrScaleInvRI16Masked(a, incr []int16, b int16, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	var errs errorIndices
	for i, v := range a {
		if !mask[i] {
			if v == int16(0) {
				errs = append(errs, i)
				incr[i] = 0
				continue
			}
			incr[i] += b / v
		}
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrScaleInvRI32(a, incr []int32, b int32) error {
	if b == int32(0) {
		return errors.New(div0General)
	}
	var errs errorIndices
	for i, v := range a {
		if v == int32(0) {
			errs = append(errs, i)
			incr[i] = 0
			continue
		}
		incr[i] += b / v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrScaleInvRI32Masked(a, incr []int32, b int32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	var errs errorIndices
	for i, v := range a {
		if !mask[i] {
			if v == int32(0) {
				errs = append(errs, i)
				incr[i] = 0
				continue
			}
			incr[i] += b / v
		}
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrScaleInvRI64(a, incr []int64, b int64) error {
	if b == int64(0) {
		return errors.New(div0General)
	}
	var errs errorIndices
	for i, v := range a {
		if v == int64(0) {
			errs = append(errs, i)
			incr[i] = 0
			continue
		}
		incr[i] += b / v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrScaleInvRI64Masked(a, incr []int64, b int64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	var errs errorIndices
	for i, v := range a {
		if !mask[i] {
			if v == int64(0) {
				errs = append(errs, i)
				incr[i] = 0
				continue
			}
			incr[i] += b / v
		}
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrScaleInvRU(a, incr []uint, b uint) error {
	if b == uint(0) {
		return errors.New(div0General)
	}
	var errs errorIndices
	for i, v := range a {
		if v == uint(0) {
			errs = append(errs, i)
			incr[i] = 0
			continue
		}
		incr[i] += b / v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrScaleInvRUMasked(a, incr []uint, b uint, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	var errs errorIndices
	for i, v := range a {
		if !mask[i] {
			if v == uint(0) {
				errs = append(errs, i)
				incr[i] = 0
				continue
			}
			incr[i] += b / v
		}
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrScaleInvRU8(a, incr []uint8, b uint8) error {
	if b == uint8(0) {
		return errors.New(div0General)
	}
	var errs errorIndices
	for i, v := range a {
		if v == uint8(0) {
			errs = append(errs, i)
			incr[i] = 0
			continue
		}
		incr[i] += b / v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrScaleInvRU8Masked(a, incr []uint8, b uint8, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	var errs errorIndices
	for i, v := range a {
		if !mask[i] {
			if v == uint8(0) {
				errs = append(errs, i)
				incr[i] = 0
				continue
			}
			incr[i] += b / v
		}
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrScaleInvRU16(a, incr []uint16, b uint16) error {
	if b == uint16(0) {
		return errors.New(div0General)
	}
	var errs errorIndices
	for i, v := range a {
		if v == uint16(0) {
			errs = append(errs, i)
			incr[i] = 0
			continue
		}
		incr[i] += b / v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrScaleInvRU16Masked(a, incr []uint16, b uint16, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	var errs errorIndices
	for i, v := range a {
		if !mask[i] {
			if v == uint16(0) {
				errs = append(errs, i)
				incr[i] = 0
				continue
			}
			incr[i] += b / v
		}
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrScaleInvRU32(a, incr []uint32, b uint32) error {
	if b == uint32(0) {
		return errors.New(div0General)
	}
	var errs errorIndices
	for i, v := range a {
		if v == uint32(0) {
			errs = append(errs, i)
			incr[i] = 0
			continue
		}
		incr[i] += b / v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrScaleInvRU32Masked(a, incr []uint32, b uint32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	var errs errorIndices
	for i, v := range a {
		if !mask[i] {
			if v == uint32(0) {
				errs = append(errs, i)
				incr[i] = 0
				continue
			}
			incr[i] += b / v
		}
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrScaleInvRU64(a, incr []uint64, b uint64) error {
	if b == uint64(0) {
		return errors.New(div0General)
	}
	var errs errorIndices
	for i, v := range a {
		if v == uint64(0) {
			errs = append(errs, i)
			incr[i] = 0
			continue
		}
		incr[i] += b / v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrScaleInvRU64Masked(a, incr []uint64, b uint64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	var errs errorIndices
	for i, v := range a {
		if !mask[i] {
			if v == uint64(0) {
				errs = append(errs, i)
				incr[i] = 0
				continue
			}
			incr[i] += b / v
		}
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrScaleInvRF32(a, incr []float32, b float32) error {
	vecf32.IncrScaleInvR(a, b, incr)
	return nil
}

func incrScaleInvRF32Masked(a, incr []float32, b float32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	var errs errorIndices
	for i, v := range a {
		if !mask[i] {
			if v == float32(0) {
				errs = append(errs, i)
				incr[i] = 0
				continue
			}
			incr[i] += b / v
		}
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrScaleInvRF64(a, incr []float64, b float64) error {
	vecf64.IncrScaleInvR(a, b, incr)
	return nil
}

func incrScaleInvRF64Masked(a, incr []float64, b float64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	var errs errorIndices
	for i, v := range a {
		if !mask[i] {
			if v == float64(0) {
				errs = append(errs, i)
				incr[i] = 0
				continue
			}
			incr[i] += b / v
		}
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrScaleInvRC64(a, incr []complex64, b complex64) error {
	for i, v := range a {
		incr[i] += b / v
	}
	return nil
}

func incrScaleInvRC64Masked(a, incr []complex64, b complex64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += b / v
		}
	}
	return nil
}

func incrScaleInvRC128(a, incr []complex128, b complex128) error {
	for i, v := range a {
		incr[i] += b / v
	}
	return nil
}

func incrScaleInvRC128Masked(a, incr []complex128, b complex128, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += b / v
		}
	}
	return nil
}

/* incr PowOf */

func incrPowOfI(a, incr []int, b int) error {
	for i, v := range a {
		incr[i] += int(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func incrPowOfIMasked(a, incr []int, b int, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += int(math.Pow(float64(v), float64(b)))
		}
	}
	return nil
}

func incrPowOfI8(a, incr []int8, b int8) error {
	for i, v := range a {
		incr[i] += int8(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func incrPowOfI8Masked(a, incr []int8, b int8, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += int8(math.Pow(float64(v), float64(b)))
		}
	}
	return nil
}

func incrPowOfI16(a, incr []int16, b int16) error {
	for i, v := range a {
		incr[i] += int16(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func incrPowOfI16Masked(a, incr []int16, b int16, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += int16(math.Pow(float64(v), float64(b)))
		}
	}
	return nil
}

func incrPowOfI32(a, incr []int32, b int32) error {
	for i, v := range a {
		incr[i] += int32(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func incrPowOfI32Masked(a, incr []int32, b int32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += int32(math.Pow(float64(v), float64(b)))
		}
	}
	return nil
}

func incrPowOfI64(a, incr []int64, b int64) error {
	for i, v := range a {
		incr[i] += int64(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func incrPowOfI64Masked(a, incr []int64, b int64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += int64(math.Pow(float64(v), float64(b)))
		}
	}
	return nil
}

func incrPowOfU(a, incr []uint, b uint) error {
	for i, v := range a {
		incr[i] += uint(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func incrPowOfUMasked(a, incr []uint, b uint, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += uint(math.Pow(float64(v), float64(b)))
		}
	}
	return nil
}

func incrPowOfU8(a, incr []uint8, b uint8) error {
	for i, v := range a {
		incr[i] += uint8(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func incrPowOfU8Masked(a, incr []uint8, b uint8, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += uint8(math.Pow(float64(v), float64(b)))
		}
	}
	return nil
}

func incrPowOfU16(a, incr []uint16, b uint16) error {
	for i, v := range a {
		incr[i] += uint16(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func incrPowOfU16Masked(a, incr []uint16, b uint16, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += uint16(math.Pow(float64(v), float64(b)))
		}
	}
	return nil
}

func incrPowOfU32(a, incr []uint32, b uint32) error {
	for i, v := range a {
		incr[i] += uint32(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func incrPowOfU32Masked(a, incr []uint32, b uint32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += uint32(math.Pow(float64(v), float64(b)))
		}
	}
	return nil
}

func incrPowOfU64(a, incr []uint64, b uint64) error {
	for i, v := range a {
		incr[i] += uint64(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func incrPowOfU64Masked(a, incr []uint64, b uint64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += uint64(math.Pow(float64(v), float64(b)))
		}
	}
	return nil
}

func incrPowOfF32(a, incr []float32, b float32) error {
	vecf32.IncrPowOf(a, b, incr)
	return nil
}

func incrPowOfF32Masked(a, incr []float32, b float32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += float32(math.Pow(float64(v), float64(b)))
		}
	}
	return nil
}

func incrPowOfF64(a, incr []float64, b float64) error {
	vecf64.IncrPowOf(a, b, incr)
	return nil
}

func incrPowOfF64Masked(a, incr []float64, b float64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += float64(math.Pow(float64(v), float64(b)))
		}
	}
	return nil
}

func incrPowOfC64(a, incr []complex64, b complex64) error {
	for i, v := range a {
		incr[i] += complex64(cmplx.Pow(complex128(v), complex128(b)))
	}
	return nil
}

func incrPowOfC64Masked(a, incr []complex64, b complex64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += complex64(cmplx.Pow(complex128(v), complex128(b)))
		}
	}
	return nil
}

func incrPowOfC128(a, incr []complex128, b complex128) error {
	for i, v := range a {
		incr[i] += complex128(cmplx.Pow(complex128(v), complex128(b)))
	}
	return nil
}

func incrPowOfC128Masked(a, incr []complex128, b complex128, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += complex128(cmplx.Pow(complex128(v), complex128(b)))
		}
	}
	return nil
}

/* incr PowOfR */

func incrPowOfRI(a, incr []int, b int) error {
	for i, v := range a {
		incr[i] += int(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func incrPowOfRIMasked(a, incr []int, b int, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += int(math.Pow(float64(b), float64(v)))
		}
	}
	return nil
}

func incrPowOfRI8(a, incr []int8, b int8) error {
	for i, v := range a {
		incr[i] += int8(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func incrPowOfRI8Masked(a, incr []int8, b int8, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += int8(math.Pow(float64(b), float64(v)))
		}
	}
	return nil
}

func incrPowOfRI16(a, incr []int16, b int16) error {
	for i, v := range a {
		incr[i] += int16(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func incrPowOfRI16Masked(a, incr []int16, b int16, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += int16(math.Pow(float64(b), float64(v)))
		}
	}
	return nil
}

func incrPowOfRI32(a, incr []int32, b int32) error {
	for i, v := range a {
		incr[i] += int32(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func incrPowOfRI32Masked(a, incr []int32, b int32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += int32(math.Pow(float64(b), float64(v)))
		}
	}
	return nil
}

func incrPowOfRI64(a, incr []int64, b int64) error {
	for i, v := range a {
		incr[i] += int64(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func incrPowOfRI64Masked(a, incr []int64, b int64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += int64(math.Pow(float64(b), float64(v)))
		}
	}
	return nil
}

func incrPowOfRU(a, incr []uint, b uint) error {
	for i, v := range a {
		incr[i] += uint(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func incrPowOfRUMasked(a, incr []uint, b uint, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += uint(math.Pow(float64(b), float64(v)))
		}
	}
	return nil
}

func incrPowOfRU8(a, incr []uint8, b uint8) error {
	for i, v := range a {
		incr[i] += uint8(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func incrPowOfRU8Masked(a, incr []uint8, b uint8, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += uint8(math.Pow(float64(b), float64(v)))
		}
	}
	return nil
}

func incrPowOfRU16(a, incr []uint16, b uint16) error {
	for i, v := range a {
		incr[i] += uint16(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func incrPowOfRU16Masked(a, incr []uint16, b uint16, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += uint16(math.Pow(float64(b), float64(v)))
		}
	}
	return nil
}

func incrPowOfRU32(a, incr []uint32, b uint32) error {
	for i, v := range a {
		incr[i] += uint32(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func incrPowOfRU32Masked(a, incr []uint32, b uint32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += uint32(math.Pow(float64(b), float64(v)))
		}
	}
	return nil
}

func incrPowOfRU64(a, incr []uint64, b uint64) error {
	for i, v := range a {
		incr[i] += uint64(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func incrPowOfRU64Masked(a, incr []uint64, b uint64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += uint64(math.Pow(float64(b), float64(v)))
		}
	}
	return nil
}

func incrPowOfRF32(a, incr []float32, b float32) error {
	vecf32.IncrPowOfR(a, b, incr)
	return nil
}

func incrPowOfRF32Masked(a, incr []float32, b float32, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += float32(math.Pow(float64(b), float64(v)))
		}
	}
	return nil
}

func incrPowOfRF64(a, incr []float64, b float64) error {
	vecf64.IncrPowOfR(a, b, incr)
	return nil
}

func incrPowOfRF64Masked(a, incr []float64, b float64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += float64(math.Pow(float64(b), float64(v)))
		}
	}
	return nil
}

func incrPowOfRC64(a, incr []complex64, b complex64) error {
	for i, v := range a {
		incr[i] += complex64(cmplx.Pow(complex128(b), complex128(v)))
	}
	return nil
}

func incrPowOfRC64Masked(a, incr []complex64, b complex64, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += complex64(cmplx.Pow(complex128(b), complex128(v)))
		}
	}
	return nil
}

func incrPowOfRC128(a, incr []complex128, b complex128) error {
	for i, v := range a {
		incr[i] += complex128(cmplx.Pow(complex128(b), complex128(v)))
	}
	return nil
}

func incrPowOfRC128Masked(a, incr []complex128, b complex128, mask []bool) error {
	if len(mask) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(mask))
	}
	for i, v := range a {
		if !mask[i] {
			incr[i] += complex128(cmplx.Pow(complex128(b), complex128(v)))
		}
	}
	return nil
}

func addI(a, b int) (c int)                  { return a + b }
func addI8(a, b int8) (c int8)               { return a + b }
func addI16(a, b int16) (c int16)            { return a + b }
func addI32(a, b int32) (c int32)            { return a + b }
func addI64(a, b int64) (c int64)            { return a + b }
func addU(a, b uint) (c uint)                { return a + b }
func addU8(a, b uint8) (c uint8)             { return a + b }
func addU16(a, b uint16) (c uint16)          { return a + b }
func addU32(a, b uint32) (c uint32)          { return a + b }
func addU64(a, b uint64) (c uint64)          { return a + b }
func addF32(a, b float32) (c float32)        { return a + b }
func addF64(a, b float64) (c float64)        { return a + b }
func addC64(a, b complex64) (c complex64)    { return a + b }
func addC128(a, b complex128) (c complex128) { return a + b }
