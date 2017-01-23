package tensor

import (
	"math"

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

	return nil
}

func addC128(a, b []complex128) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
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

	return nil
}

func subC128(a, b []complex128) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
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

	return nil
}

func mulC128(a, b []complex128) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
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

	return nil
}

func divC128(a, b []complex128) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
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

	return nil
}

func powC128(a, b []complex128) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
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
	vecf32.Trans([]float32(a), b)
	return nil
}

func transF64(a []float64, b float64) error {
	vecf64.Trans([]float64(a), b)
	return nil
}

func transC64(a []complex64, b complex64) error {
	return nil
}

func transC128(a []complex128, b complex128) error {
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
	vecf32.TransInv([]float32(a), b)
	return nil
}

func transinvF64(a []float64, b float64) error {
	vecf64.TransInv([]float64(a), b)
	return nil
}

func transinvC64(a []complex64, b complex64) error {
	return nil
}

func transinvC128(a []complex128, b complex128) error {
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
	vecf32.TransInvR([]float32(a), b)
	return nil
}

func transinvrF64(a []float64, b float64) error {
	vecf64.TransInvR([]float64(a), b)
	return nil
}

func transinvrC64(a []complex64, b complex64) error {
	return nil
}

func transinvrC128(a []complex128, b complex128) error {
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
	vecf32.Scale([]float32(a), b)
	return nil
}

func scaleF64(a []float64, b float64) error {
	vecf64.Scale([]float64(a), b)
	return nil
}

func scaleC64(a []complex64, b complex64) error {
	return nil
}

func scaleC128(a []complex128, b complex128) error {
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
	vecf32.ScaleInv([]float32(a), b)
	return nil
}

func scaleinvF64(a []float64, b float64) error {
	vecf64.ScaleInv([]float64(a), b)
	return nil
}

func scaleinvC64(a []complex64, b complex64) error {
	return nil
}

func scaleinvC128(a []complex128, b complex128) error {
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
	vecf32.ScaleInvR([]float32(a), b)
	return nil
}

func scaleinvrF64(a []float64, b float64) error {
	vecf64.ScaleInvR([]float64(a), b)
	return nil
}

func scaleinvrC64(a []complex64, b complex64) error {
	return nil
}

func scaleinvrC128(a []complex128, b complex128) error {
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
	vecf32.PowOf([]float32(a), b)
	return nil
}

func powofF64(a []float64, b float64) error {
	vecf64.PowOf([]float64(a), b)
	return nil
}

func powofC64(a []complex64, b complex64) error {
	return nil
}

func powofC128(a []complex128, b complex128) error {
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
	vecf32.PowOfR([]float32(a), b)
	return nil
}

func powofrF64(a []float64, b float64) error {
	vecf64.PowOfR([]float64(a), b)
	return nil
}

func powofrC64(a []complex64, b complex64) error {
	return nil
}

func powofrC128(a []complex128, b complex128) error {
	return nil
}
