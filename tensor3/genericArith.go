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

func addI(a, b []int) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] += v
	}

	return nil
}

func addI8(a, b []int8) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] += v
	}

	return nil
}

func addI16(a, b []int16) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] += v
	}

	return nil
}

func addI32(a, b []int32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] += v
	}

	return nil
}

func addI64(a, b []int64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] += v
	}

	return nil
}

func addU(a, b []uint) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] += v
	}

	return nil
}

func addU8(a, b []uint8) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] += v
	}

	return nil
}

func addU16(a, b []uint16) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] += v
	}

	return nil
}

func addU32(a, b []uint32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] += v
	}

	return nil
}

func addU64(a, b []uint64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] += v
	}

	return nil
}

func addF32(a, b []float32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	vecf32.Add(a, b)
	return nil
}

func addF64(a, b []float64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	vecf64.Add(a, b)
	return nil
}

func addC64(a, b []complex64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] += v
	}
	return nil
}

func addC128(a, b []complex128) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] += v
	}
	return nil
}

/* Sub */

func subI(a, b []int) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] -= v
	}

	return nil
}

func subI8(a, b []int8) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] -= v
	}

	return nil
}

func subI16(a, b []int16) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] -= v
	}

	return nil
}

func subI32(a, b []int32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] -= v
	}

	return nil
}

func subI64(a, b []int64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] -= v
	}

	return nil
}

func subU(a, b []uint) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] -= v
	}

	return nil
}

func subU8(a, b []uint8) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] -= v
	}

	return nil
}

func subU16(a, b []uint16) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] -= v
	}

	return nil
}

func subU32(a, b []uint32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] -= v
	}

	return nil
}

func subU64(a, b []uint64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] -= v
	}

	return nil
}

func subF32(a, b []float32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	vecf32.Sub(a, b)
	return nil
}

func subF64(a, b []float64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	vecf64.Sub(a, b)
	return nil
}

func subC64(a, b []complex64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] -= v
	}
	return nil
}

func subC128(a, b []complex128) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] -= v
	}
	return nil
}

/* Mul */

func mulI(a, b []int) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] *= v
	}

	return nil
}

func mulI8(a, b []int8) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] *= v
	}

	return nil
}

func mulI16(a, b []int16) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] *= v
	}

	return nil
}

func mulI32(a, b []int32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] *= v
	}

	return nil
}

func mulI64(a, b []int64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] *= v
	}

	return nil
}

func mulU(a, b []uint) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] *= v
	}

	return nil
}

func mulU8(a, b []uint8) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] *= v
	}

	return nil
}

func mulU16(a, b []uint16) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] *= v
	}

	return nil
}

func mulU32(a, b []uint32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] *= v
	}

	return nil
}

func mulU64(a, b []uint64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] *= v
	}

	return nil
}

func mulF32(a, b []float32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	vecf32.Mul(a, b)
	return nil
}

func mulF64(a, b []float64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	vecf64.Mul(a, b)
	return nil
}

func mulC64(a, b []complex64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] *= v
	}
	return nil
}

func mulC128(a, b []complex128) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] *= v
	}
	return nil
}

/* Div */

func divI(a, b []int) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

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

func divI8(a, b []int8) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

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

func divI16(a, b []int16) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

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

func divI32(a, b []int32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

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

func divI64(a, b []int64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

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

func divU(a, b []uint) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

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

func divU8(a, b []uint8) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

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

func divU16(a, b []uint16) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

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

func divU32(a, b []uint32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

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

func divU64(a, b []uint64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

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

func divF32(a, b []float32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	vecf32.Div(a, b)
	return nil
}

func divF64(a, b []float64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	vecf64.Div(a, b)
	return nil
}

func divC64(a, b []complex64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] /= v
	}
	return nil
}

func divC128(a, b []complex128) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] /= v
	}
	return nil
}

/* Pow */

func powI(a, b []int) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] = int(math.Pow(float64(a[i]), float64(v)))
	}

	return nil
}

func powI8(a, b []int8) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] = int8(math.Pow(float64(a[i]), float64(v)))
	}

	return nil
}

func powI16(a, b []int16) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] = int16(math.Pow(float64(a[i]), float64(v)))
	}

	return nil
}

func powI32(a, b []int32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] = int32(math.Pow(float64(a[i]), float64(v)))
	}

	return nil
}

func powI64(a, b []int64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] = int64(math.Pow(float64(a[i]), float64(v)))
	}

	return nil
}

func powU(a, b []uint) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] = uint(math.Pow(float64(a[i]), float64(v)))
	}

	return nil
}

func powU8(a, b []uint8) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] = uint8(math.Pow(float64(a[i]), float64(v)))
	}

	return nil
}

func powU16(a, b []uint16) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] = uint16(math.Pow(float64(a[i]), float64(v)))
	}

	return nil
}

func powU32(a, b []uint32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] = uint32(math.Pow(float64(a[i]), float64(v)))
	}

	return nil
}

func powU64(a, b []uint64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] = uint64(math.Pow(float64(a[i]), float64(v)))
	}

	return nil
}

func powF32(a, b []float32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	vecf32.Pow(a, b)
	return nil
}

func powF64(a, b []float64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	vecf64.Pow(a, b)
	return nil
}

func powC64(a, b []complex64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] = complex64(cmplx.Pow(complex128(a[i]), complex128(v)))
	}
	return nil
}

func powC128(a, b []complex128) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	for i, v := range b {
		a[i] = complex128(cmplx.Pow(complex128(a[i]), complex128(v)))
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

func transI8(a []int8, b int8) error {
	for i, v := range a {
		a[i] = v + b
	}
	return nil
}

func transI16(a []int16, b int16) error {
	for i, v := range a {
		a[i] = v + b
	}
	return nil
}

func transI32(a []int32, b int32) error {
	for i, v := range a {
		a[i] = v + b
	}
	return nil
}

func transI64(a []int64, b int64) error {
	for i, v := range a {
		a[i] = v + b
	}
	return nil
}

func transU(a []uint, b uint) error {
	for i, v := range a {
		a[i] = v + b
	}
	return nil
}

func transU8(a []uint8, b uint8) error {
	for i, v := range a {
		a[i] = v + b
	}
	return nil
}

func transU16(a []uint16, b uint16) error {
	for i, v := range a {
		a[i] = v + b
	}
	return nil
}

func transU32(a []uint32, b uint32) error {
	for i, v := range a {
		a[i] = v + b
	}
	return nil
}

func transU64(a []uint64, b uint64) error {
	for i, v := range a {
		a[i] = v + b
	}
	return nil
}

func transF32(a []float32, b float32) error {
	vecf32.Trans(a, b)
	return nil
}

func transF64(a []float64, b float64) error {
	vecf64.Trans(a, b)
	return nil
}

func transC64(a []complex64, b complex64) error {
	for i, v := range a {
		a[i] = v + b
	}
	return nil
}

func transC128(a []complex128, b complex128) error {
	for i, v := range a {
		a[i] = v + b
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

func transinvI8(a []int8, b int8) error {
	for i, v := range a {
		a[i] = v - b
	}
	return nil
}

func transinvI16(a []int16, b int16) error {
	for i, v := range a {
		a[i] = v - b
	}
	return nil
}

func transinvI32(a []int32, b int32) error {
	for i, v := range a {
		a[i] = v - b
	}
	return nil
}

func transinvI64(a []int64, b int64) error {
	for i, v := range a {
		a[i] = v - b
	}
	return nil
}

func transinvU(a []uint, b uint) error {
	for i, v := range a {
		a[i] = v - b
	}
	return nil
}

func transinvU8(a []uint8, b uint8) error {
	for i, v := range a {
		a[i] = v - b
	}
	return nil
}

func transinvU16(a []uint16, b uint16) error {
	for i, v := range a {
		a[i] = v - b
	}
	return nil
}

func transinvU32(a []uint32, b uint32) error {
	for i, v := range a {
		a[i] = v - b
	}
	return nil
}

func transinvU64(a []uint64, b uint64) error {
	for i, v := range a {
		a[i] = v - b
	}
	return nil
}

func transinvF32(a []float32, b float32) error {
	vecf32.TransInv(a, b)
	return nil
}

func transinvF64(a []float64, b float64) error {
	vecf64.TransInv(a, b)
	return nil
}

func transinvC64(a []complex64, b complex64) error {
	for i, v := range a {
		a[i] = v - b
	}
	return nil
}

func transinvC128(a []complex128, b complex128) error {
	for i, v := range a {
		a[i] = v - b
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

func transinvrI8(a []int8, b int8) error {
	for i, v := range a {
		a[i] = b - v
	}
	return nil
}

func transinvrI16(a []int16, b int16) error {
	for i, v := range a {
		a[i] = b - v
	}
	return nil
}

func transinvrI32(a []int32, b int32) error {
	for i, v := range a {
		a[i] = b - v
	}
	return nil
}

func transinvrI64(a []int64, b int64) error {
	for i, v := range a {
		a[i] = b - v
	}
	return nil
}

func transinvrU(a []uint, b uint) error {
	for i, v := range a {
		a[i] = b - v
	}
	return nil
}

func transinvrU8(a []uint8, b uint8) error {
	for i, v := range a {
		a[i] = b - v
	}
	return nil
}

func transinvrU16(a []uint16, b uint16) error {
	for i, v := range a {
		a[i] = b - v
	}
	return nil
}

func transinvrU32(a []uint32, b uint32) error {
	for i, v := range a {
		a[i] = b - v
	}
	return nil
}

func transinvrU64(a []uint64, b uint64) error {
	for i, v := range a {
		a[i] = b - v
	}
	return nil
}

func transinvrF32(a []float32, b float32) error {
	vecf32.TransInvR(a, b)
	return nil
}

func transinvrF64(a []float64, b float64) error {
	vecf64.TransInvR(a, b)
	return nil
}

func transinvrC64(a []complex64, b complex64) error {
	for i, v := range a {
		a[i] = b - v
	}
	return nil
}

func transinvrC128(a []complex128, b complex128) error {
	for i, v := range a {
		a[i] = b - v
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

func scaleI8(a []int8, b int8) error {
	for i, v := range a {
		a[i] = v * b
	}
	return nil
}

func scaleI16(a []int16, b int16) error {
	for i, v := range a {
		a[i] = v * b
	}
	return nil
}

func scaleI32(a []int32, b int32) error {
	for i, v := range a {
		a[i] = v * b
	}
	return nil
}

func scaleI64(a []int64, b int64) error {
	for i, v := range a {
		a[i] = v * b
	}
	return nil
}

func scaleU(a []uint, b uint) error {
	for i, v := range a {
		a[i] = v * b
	}
	return nil
}

func scaleU8(a []uint8, b uint8) error {
	for i, v := range a {
		a[i] = v * b
	}
	return nil
}

func scaleU16(a []uint16, b uint16) error {
	for i, v := range a {
		a[i] = v * b
	}
	return nil
}

func scaleU32(a []uint32, b uint32) error {
	for i, v := range a {
		a[i] = v * b
	}
	return nil
}

func scaleU64(a []uint64, b uint64) error {
	for i, v := range a {
		a[i] = v * b
	}
	return nil
}

func scaleF32(a []float32, b float32) error {
	vecf32.Scale(a, b)
	return nil
}

func scaleF64(a []float64, b float64) error {
	vecf64.Scale(a, b)
	return nil
}

func scaleC64(a []complex64, b complex64) error {
	for i, v := range a {
		a[i] = v * b
	}
	return nil
}

func scaleC128(a []complex128, b complex128) error {
	for i, v := range a {
		a[i] = v * b
	}
	return nil
}

/* ScaleInv */

func scaleinvI(a []int, b int) error {
	var errs errorIndices
	for i, v := range a {
		if v == int(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}

		a[i] = v / b
	}
	if errs != nil {
		return errs
	}
	return nil
}

func scaleinvI8(a []int8, b int8) error {
	var errs errorIndices
	for i, v := range a {
		if v == int8(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}

		a[i] = v / b
	}
	if errs != nil {
		return errs
	}
	return nil
}

func scaleinvI16(a []int16, b int16) error {
	var errs errorIndices
	for i, v := range a {
		if v == int16(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}

		a[i] = v / b
	}
	if errs != nil {
		return errs
	}
	return nil
}

func scaleinvI32(a []int32, b int32) error {
	var errs errorIndices
	for i, v := range a {
		if v == int32(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}

		a[i] = v / b
	}
	if errs != nil {
		return errs
	}
	return nil
}

func scaleinvI64(a []int64, b int64) error {
	var errs errorIndices
	for i, v := range a {
		if v == int64(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}

		a[i] = v / b
	}
	if errs != nil {
		return errs
	}
	return nil
}

func scaleinvU(a []uint, b uint) error {
	var errs errorIndices
	for i, v := range a {
		if v == uint(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}

		a[i] = v / b
	}
	if errs != nil {
		return errs
	}
	return nil
}

func scaleinvU8(a []uint8, b uint8) error {
	var errs errorIndices
	for i, v := range a {
		if v == uint8(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}

		a[i] = v / b
	}
	if errs != nil {
		return errs
	}
	return nil
}

func scaleinvU16(a []uint16, b uint16) error {
	var errs errorIndices
	for i, v := range a {
		if v == uint16(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}

		a[i] = v / b
	}
	if errs != nil {
		return errs
	}
	return nil
}

func scaleinvU32(a []uint32, b uint32) error {
	var errs errorIndices
	for i, v := range a {
		if v == uint32(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}

		a[i] = v / b
	}
	if errs != nil {
		return errs
	}
	return nil
}

func scaleinvU64(a []uint64, b uint64) error {
	var errs errorIndices
	for i, v := range a {
		if v == uint64(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}

		a[i] = v / b
	}
	if errs != nil {
		return errs
	}
	return nil
}

func scaleinvF32(a []float32, b float32) error {
	vecf32.ScaleInv(a, b)
	return nil
}

func scaleinvF64(a []float64, b float64) error {
	vecf64.ScaleInv(a, b)
	return nil
}

func scaleinvC64(a []complex64, b complex64) error {
	for i, v := range a {
		a[i] = v / b
	}
	return nil
}

func scaleinvC128(a []complex128, b complex128) error {
	for i, v := range a {
		a[i] = v / b
	}
	return nil
}

/* ScaleInvR */

func scaleinvrI(a []int, b int) error {
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

func scaleinvrI8(a []int8, b int8) error {
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

func scaleinvrI16(a []int16, b int16) error {
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

func scaleinvrI32(a []int32, b int32) error {
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

func scaleinvrI64(a []int64, b int64) error {
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

func scaleinvrU(a []uint, b uint) error {
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

func scaleinvrU8(a []uint8, b uint8) error {
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

func scaleinvrU16(a []uint16, b uint16) error {
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

func scaleinvrU32(a []uint32, b uint32) error {
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

func scaleinvrU64(a []uint64, b uint64) error {
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

func scaleinvrF32(a []float32, b float32) error {
	vecf32.ScaleInvR(a, b)
	return nil
}

func scaleinvrF64(a []float64, b float64) error {
	vecf64.ScaleInvR(a, b)
	return nil
}

func scaleinvrC64(a []complex64, b complex64) error {
	for i, v := range a {
		a[i] = b / v
	}
	return nil
}

func scaleinvrC128(a []complex128, b complex128) error {
	for i, v := range a {
		a[i] = b / v
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

func powofI8(a []int8, b int8) error {
	for i, v := range a {
		a[i] = int8(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func powofI16(a []int16, b int16) error {
	for i, v := range a {
		a[i] = int16(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func powofI32(a []int32, b int32) error {
	for i, v := range a {
		a[i] = int32(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func powofI64(a []int64, b int64) error {
	for i, v := range a {
		a[i] = int64(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func powofU(a []uint, b uint) error {
	for i, v := range a {
		a[i] = uint(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func powofU8(a []uint8, b uint8) error {
	for i, v := range a {
		a[i] = uint8(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func powofU16(a []uint16, b uint16) error {
	for i, v := range a {
		a[i] = uint16(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func powofU32(a []uint32, b uint32) error {
	for i, v := range a {
		a[i] = uint32(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func powofU64(a []uint64, b uint64) error {
	for i, v := range a {
		a[i] = uint64(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func powofF32(a []float32, b float32) error {
	vecf32.PowOf(a, b)
	return nil
}

func powofF64(a []float64, b float64) error {
	vecf64.PowOf(a, b)
	return nil
}

func powofC64(a []complex64, b complex64) error {
	for i, v := range a {
		a[i] = complex64(cmplx.Pow(complex128(v), complex128(b)))
	}
	return nil
}

func powofC128(a []complex128, b complex128) error {
	for i, v := range a {
		a[i] = complex128(cmplx.Pow(complex128(v), complex128(b)))
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

func powofrI8(a []int8, b int8) error {
	for i, v := range a {
		a[i] = int8(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func powofrI16(a []int16, b int16) error {
	for i, v := range a {
		a[i] = int16(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func powofrI32(a []int32, b int32) error {
	for i, v := range a {
		a[i] = int32(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func powofrI64(a []int64, b int64) error {
	for i, v := range a {
		a[i] = int64(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func powofrU(a []uint, b uint) error {
	for i, v := range a {
		a[i] = uint(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func powofrU8(a []uint8, b uint8) error {
	for i, v := range a {
		a[i] = uint8(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func powofrU16(a []uint16, b uint16) error {
	for i, v := range a {
		a[i] = uint16(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func powofrU32(a []uint32, b uint32) error {
	for i, v := range a {
		a[i] = uint32(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func powofrU64(a []uint64, b uint64) error {
	for i, v := range a {
		a[i] = uint64(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func powofrF32(a []float32, b float32) error {
	vecf32.PowOfR(a, b)
	return nil
}

func powofrF64(a []float64, b float64) error {
	vecf64.PowOfR(a, b)
	return nil
}

func powofrC64(a []complex64, b complex64) error {
	for i, v := range a {
		a[i] = complex64(cmplx.Pow(complex128(b), complex128(v)))
	}
	return nil
}

func powofrC128(a []complex128, b complex128) error {
	for i, v := range a {
		a[i] = complex128(cmplx.Pow(complex128(b), complex128(v)))
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

func incrTransI8(a, incr []int8, b int8) error {
	for i, v := range a {
		incr[i] += v + b
	}
	return nil
}

func incrTransI16(a, incr []int16, b int16) error {
	for i, v := range a {
		incr[i] += v + b
	}
	return nil
}

func incrTransI32(a, incr []int32, b int32) error {
	for i, v := range a {
		incr[i] += v + b
	}
	return nil
}

func incrTransI64(a, incr []int64, b int64) error {
	for i, v := range a {
		incr[i] += v + b
	}
	return nil
}

func incrTransU(a, incr []uint, b uint) error {
	for i, v := range a {
		incr[i] += v + b
	}
	return nil
}

func incrTransU8(a, incr []uint8, b uint8) error {
	for i, v := range a {
		incr[i] += v + b
	}
	return nil
}

func incrTransU16(a, incr []uint16, b uint16) error {
	for i, v := range a {
		incr[i] += v + b
	}
	return nil
}

func incrTransU32(a, incr []uint32, b uint32) error {
	for i, v := range a {
		incr[i] += v + b
	}
	return nil
}

func incrTransU64(a, incr []uint64, b uint64) error {
	for i, v := range a {
		incr[i] += v + b
	}
	return nil
}

func incrTransF32(a, incr []float32, b float32) error {
	vecf32.IncrTrans(a, b, incr)
	return nil
}

func incrTransF64(a, incr []float64, b float64) error {
	vecf64.IncrTrans(a, b, incr)
	return nil
}

func incrTransC64(a, incr []complex64, b complex64) error {
	for i, v := range a {
		incr[i] += v + b
	}
	return nil
}

func incrTransC128(a, incr []complex128, b complex128) error {
	for i, v := range a {
		incr[i] += v + b
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

func incrTransInvI8(a, incr []int8, b int8) error {
	for i, v := range a {
		incr[i] += v - b
	}
	return nil
}

func incrTransInvI16(a, incr []int16, b int16) error {
	for i, v := range a {
		incr[i] += v - b
	}
	return nil
}

func incrTransInvI32(a, incr []int32, b int32) error {
	for i, v := range a {
		incr[i] += v - b
	}
	return nil
}

func incrTransInvI64(a, incr []int64, b int64) error {
	for i, v := range a {
		incr[i] += v - b
	}
	return nil
}

func incrTransInvU(a, incr []uint, b uint) error {
	for i, v := range a {
		incr[i] += v - b
	}
	return nil
}

func incrTransInvU8(a, incr []uint8, b uint8) error {
	for i, v := range a {
		incr[i] += v - b
	}
	return nil
}

func incrTransInvU16(a, incr []uint16, b uint16) error {
	for i, v := range a {
		incr[i] += v - b
	}
	return nil
}

func incrTransInvU32(a, incr []uint32, b uint32) error {
	for i, v := range a {
		incr[i] += v - b
	}
	return nil
}

func incrTransInvU64(a, incr []uint64, b uint64) error {
	for i, v := range a {
		incr[i] += v - b
	}
	return nil
}

func incrTransInvF32(a, incr []float32, b float32) error {
	vecf32.IncrTransInv(a, b, incr)
	return nil
}

func incrTransInvF64(a, incr []float64, b float64) error {
	vecf64.IncrTransInv(a, b, incr)
	return nil
}

func incrTransInvC64(a, incr []complex64, b complex64) error {
	for i, v := range a {
		incr[i] += v - b
	}
	return nil
}

func incrTransInvC128(a, incr []complex128, b complex128) error {
	for i, v := range a {
		incr[i] += v - b
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

func incrTransInvRI8(a, incr []int8, b int8) error {
	for i, v := range a {
		incr[i] += b - v
	}
	return nil
}

func incrTransInvRI16(a, incr []int16, b int16) error {
	for i, v := range a {
		incr[i] += b - v
	}
	return nil
}

func incrTransInvRI32(a, incr []int32, b int32) error {
	for i, v := range a {
		incr[i] += b - v
	}
	return nil
}

func incrTransInvRI64(a, incr []int64, b int64) error {
	for i, v := range a {
		incr[i] += b - v
	}
	return nil
}

func incrTransInvRU(a, incr []uint, b uint) error {
	for i, v := range a {
		incr[i] += b - v
	}
	return nil
}

func incrTransInvRU8(a, incr []uint8, b uint8) error {
	for i, v := range a {
		incr[i] += b - v
	}
	return nil
}

func incrTransInvRU16(a, incr []uint16, b uint16) error {
	for i, v := range a {
		incr[i] += b - v
	}
	return nil
}

func incrTransInvRU32(a, incr []uint32, b uint32) error {
	for i, v := range a {
		incr[i] += b - v
	}
	return nil
}

func incrTransInvRU64(a, incr []uint64, b uint64) error {
	for i, v := range a {
		incr[i] += b - v
	}
	return nil
}

func incrTransInvRF32(a, incr []float32, b float32) error {
	vecf32.IncrTransInvR(a, b, incr)
	return nil
}

func incrTransInvRF64(a, incr []float64, b float64) error {
	vecf64.IncrTransInvR(a, b, incr)
	return nil
}

func incrTransInvRC64(a, incr []complex64, b complex64) error {
	for i, v := range a {
		incr[i] += b - v
	}
	return nil
}

func incrTransInvRC128(a, incr []complex128, b complex128) error {
	for i, v := range a {
		incr[i] += b - v
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

func incrScaleI8(a, incr []int8, b int8) error {
	for i, v := range a {
		incr[i] += v * b
	}
	return nil
}

func incrScaleI16(a, incr []int16, b int16) error {
	for i, v := range a {
		incr[i] += v * b
	}
	return nil
}

func incrScaleI32(a, incr []int32, b int32) error {
	for i, v := range a {
		incr[i] += v * b
	}
	return nil
}

func incrScaleI64(a, incr []int64, b int64) error {
	for i, v := range a {
		incr[i] += v * b
	}
	return nil
}

func incrScaleU(a, incr []uint, b uint) error {
	for i, v := range a {
		incr[i] += v * b
	}
	return nil
}

func incrScaleU8(a, incr []uint8, b uint8) error {
	for i, v := range a {
		incr[i] += v * b
	}
	return nil
}

func incrScaleU16(a, incr []uint16, b uint16) error {
	for i, v := range a {
		incr[i] += v * b
	}
	return nil
}

func incrScaleU32(a, incr []uint32, b uint32) error {
	for i, v := range a {
		incr[i] += v * b
	}
	return nil
}

func incrScaleU64(a, incr []uint64, b uint64) error {
	for i, v := range a {
		incr[i] += v * b
	}
	return nil
}

func incrScaleF32(a, incr []float32, b float32) error {
	vecf32.IncrScale(a, b, incr)
	return nil
}

func incrScaleF64(a, incr []float64, b float64) error {
	vecf64.IncrScale(a, b, incr)
	return nil
}

func incrScaleC64(a, incr []complex64, b complex64) error {
	for i, v := range a {
		incr[i] += v * b
	}
	return nil
}

func incrScaleC128(a, incr []complex128, b complex128) error {
	for i, v := range a {
		incr[i] += v * b
	}
	return nil
}

/* incr ScaleInv */

func incrScaleInvI(a, incr []int, b int) error {
	var errs errorIndices
	for i, v := range a {
		if v == int(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}

		incr[i] += v / b
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrScaleInvI8(a, incr []int8, b int8) error {
	var errs errorIndices
	for i, v := range a {
		if v == int8(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}

		incr[i] += v / b
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrScaleInvI16(a, incr []int16, b int16) error {
	var errs errorIndices
	for i, v := range a {
		if v == int16(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}

		incr[i] += v / b
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrScaleInvI32(a, incr []int32, b int32) error {
	var errs errorIndices
	for i, v := range a {
		if v == int32(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}

		incr[i] += v / b
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrScaleInvI64(a, incr []int64, b int64) error {
	var errs errorIndices
	for i, v := range a {
		if v == int64(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}

		incr[i] += v / b
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrScaleInvU(a, incr []uint, b uint) error {
	var errs errorIndices
	for i, v := range a {
		if v == uint(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}

		incr[i] += v / b
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrScaleInvU8(a, incr []uint8, b uint8) error {
	var errs errorIndices
	for i, v := range a {
		if v == uint8(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}

		incr[i] += v / b
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrScaleInvU16(a, incr []uint16, b uint16) error {
	var errs errorIndices
	for i, v := range a {
		if v == uint16(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}

		incr[i] += v / b
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrScaleInvU32(a, incr []uint32, b uint32) error {
	var errs errorIndices
	for i, v := range a {
		if v == uint32(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}

		incr[i] += v / b
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrScaleInvU64(a, incr []uint64, b uint64) error {
	var errs errorIndices
	for i, v := range a {
		if v == uint64(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}

		incr[i] += v / b
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrScaleInvF32(a, incr []float32, b float32) error {
	vecf32.IncrScaleInv(a, b, incr)
	return nil
}

func incrScaleInvF64(a, incr []float64, b float64) error {
	vecf64.IncrScaleInv(a, b, incr)
	return nil
}

func incrScaleInvC64(a, incr []complex64, b complex64) error {
	for i, v := range a {
		incr[i] += v / b
	}
	return nil
}

func incrScaleInvC128(a, incr []complex128, b complex128) error {
	for i, v := range a {
		incr[i] += v / b
	}
	return nil
}

/* incr ScaleInvR */

func incrScaleInvRI(a, incr []int, b int) error {
	var errs errorIndices
	for i, v := range a {
		if v == int(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}

		incr[i] += b / v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrScaleInvRI8(a, incr []int8, b int8) error {
	var errs errorIndices
	for i, v := range a {
		if v == int8(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}

		incr[i] += b / v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrScaleInvRI16(a, incr []int16, b int16) error {
	var errs errorIndices
	for i, v := range a {
		if v == int16(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}

		incr[i] += b / v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrScaleInvRI32(a, incr []int32, b int32) error {
	var errs errorIndices
	for i, v := range a {
		if v == int32(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}

		incr[i] += b / v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrScaleInvRI64(a, incr []int64, b int64) error {
	var errs errorIndices
	for i, v := range a {
		if v == int64(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}

		incr[i] += b / v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrScaleInvRU(a, incr []uint, b uint) error {
	var errs errorIndices
	for i, v := range a {
		if v == uint(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}

		incr[i] += b / v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrScaleInvRU8(a, incr []uint8, b uint8) error {
	var errs errorIndices
	for i, v := range a {
		if v == uint8(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}

		incr[i] += b / v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrScaleInvRU16(a, incr []uint16, b uint16) error {
	var errs errorIndices
	for i, v := range a {
		if v == uint16(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}

		incr[i] += b / v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrScaleInvRU32(a, incr []uint32, b uint32) error {
	var errs errorIndices
	for i, v := range a {
		if v == uint32(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}

		incr[i] += b / v
	}
	if errs != nil {
		return errs
	}
	return nil
}

func incrScaleInvRU64(a, incr []uint64, b uint64) error {
	var errs errorIndices
	for i, v := range a {
		if v == uint64(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}

		incr[i] += b / v
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

func incrScaleInvRF64(a, incr []float64, b float64) error {
	vecf64.IncrScaleInvR(a, b, incr)
	return nil
}

func incrScaleInvRC64(a, incr []complex64, b complex64) error {
	for i, v := range a {
		incr[i] += b / v
	}
	return nil
}

func incrScaleInvRC128(a, incr []complex128, b complex128) error {
	for i, v := range a {
		incr[i] += b / v
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

func incrPowOfI8(a, incr []int8, b int8) error {
	for i, v := range a {
		incr[i] += int8(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func incrPowOfI16(a, incr []int16, b int16) error {
	for i, v := range a {
		incr[i] += int16(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func incrPowOfI32(a, incr []int32, b int32) error {
	for i, v := range a {
		incr[i] += int32(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func incrPowOfI64(a, incr []int64, b int64) error {
	for i, v := range a {
		incr[i] += int64(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func incrPowOfU(a, incr []uint, b uint) error {
	for i, v := range a {
		incr[i] += uint(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func incrPowOfU8(a, incr []uint8, b uint8) error {
	for i, v := range a {
		incr[i] += uint8(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func incrPowOfU16(a, incr []uint16, b uint16) error {
	for i, v := range a {
		incr[i] += uint16(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func incrPowOfU32(a, incr []uint32, b uint32) error {
	for i, v := range a {
		incr[i] += uint32(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func incrPowOfU64(a, incr []uint64, b uint64) error {
	for i, v := range a {
		incr[i] += uint64(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func incrPowOfF32(a, incr []float32, b float32) error {
	vecf32.IncrPowOf(a, b, incr)
	return nil
}

func incrPowOfF64(a, incr []float64, b float64) error {
	vecf64.IncrPowOf(a, b, incr)
	return nil
}

func incrPowOfC64(a, incr []complex64, b complex64) error {
	for i, v := range a {
		incr[i] += complex64(cmplx.Pow(complex128(v), complex128(b)))
	}
	return nil
}

func incrPowOfC128(a, incr []complex128, b complex128) error {
	for i, v := range a {
		incr[i] += complex128(cmplx.Pow(complex128(v), complex128(b)))
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

func incrPowOfRI8(a, incr []int8, b int8) error {
	for i, v := range a {
		incr[i] += int8(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func incrPowOfRI16(a, incr []int16, b int16) error {
	for i, v := range a {
		incr[i] += int16(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func incrPowOfRI32(a, incr []int32, b int32) error {
	for i, v := range a {
		incr[i] += int32(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func incrPowOfRI64(a, incr []int64, b int64) error {
	for i, v := range a {
		incr[i] += int64(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func incrPowOfRU(a, incr []uint, b uint) error {
	for i, v := range a {
		incr[i] += uint(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func incrPowOfRU8(a, incr []uint8, b uint8) error {
	for i, v := range a {
		incr[i] += uint8(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func incrPowOfRU16(a, incr []uint16, b uint16) error {
	for i, v := range a {
		incr[i] += uint16(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func incrPowOfRU32(a, incr []uint32, b uint32) error {
	for i, v := range a {
		incr[i] += uint32(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func incrPowOfRU64(a, incr []uint64, b uint64) error {
	for i, v := range a {
		incr[i] += uint64(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func incrPowOfRF32(a, incr []float32, b float32) error {
	vecf32.IncrPowOfR(a, b, incr)
	return nil
}

func incrPowOfRF64(a, incr []float64, b float64) error {
	vecf64.IncrPowOfR(a, b, incr)
	return nil
}

func incrPowOfRC64(a, incr []complex64, b complex64) error {
	for i, v := range a {
		incr[i] += complex64(cmplx.Pow(complex128(b), complex128(v)))
	}
	return nil
}

func incrPowOfRC128(a, incr []complex128, b complex128) error {
	for i, v := range a {
		incr[i] += complex128(cmplx.Pow(complex128(b), complex128(v)))
	}
	return nil
}
