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

func (a f64s) Add(other Number) error {
	b, err := getFloat64s(other)
	if err != nil {
		return errors.Wrapf(err, opFail, "Add")
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Add", len(a), len(b))
	}

	vecf64.Add([]float64(a), b)
	return nil
}

func (a f32s) Add(other Number) error {
	b, err := getFloat32s(other)
	if err != nil {
		return errors.Wrapf(err, opFail, "Add")
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Add", len(a), len(b))
	}

	vecf32.Add([]float32(a), b)
	return nil
}

func (a ints) Add(other Number) error {
	b, err := getInts(other)
	if err != nil {
		return errors.Wrapf(err, opFail, "Add")
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Add", len(a), len(b))
	}

	for i, v := range b {
		a[i] += v
	}

	return nil
}

func (a i64s) Add(other Number) error {
	b, err := getInt64s(other)
	if err != nil {
		return errors.Wrapf(err, opFail, "Add")
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Add", len(a), len(b))
	}

	for i, v := range b {
		a[i] += v
	}

	return nil
}

func (a i32s) Add(other Number) error {
	b, err := getInt32s(other)
	if err != nil {
		return errors.Wrapf(err, opFail, "Add")
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Add", len(a), len(b))
	}

	for i, v := range b {
		a[i] += v
	}

	return nil
}

func (a u8s) Add(other Number) error {
	b, err := getBytes(other)
	if err != nil {
		return errors.Wrapf(err, opFail, "Add")
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Add", len(a), len(b))
	}

	for i, v := range b {
		a[i] += v
	}

	return nil
}

/* Sub */

func (a f64s) Sub(other Number) error {
	b, err := getFloat64s(other)
	if err != nil {
		return errors.Wrapf(err, opFail, "Sub")
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Sub", len(a), len(b))
	}

	vecf64.Sub([]float64(a), b)
	return nil
}

func (a f32s) Sub(other Number) error {
	b, err := getFloat32s(other)
	if err != nil {
		return errors.Wrapf(err, opFail, "Sub")
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Sub", len(a), len(b))
	}

	vecf32.Sub([]float32(a), b)
	return nil
}

func (a ints) Sub(other Number) error {
	b, err := getInts(other)
	if err != nil {
		return errors.Wrapf(err, opFail, "Sub")
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Sub", len(a), len(b))
	}

	for i, v := range b {
		a[i] -= v
	}

	return nil
}

func (a i64s) Sub(other Number) error {
	b, err := getInt64s(other)
	if err != nil {
		return errors.Wrapf(err, opFail, "Sub")
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Sub", len(a), len(b))
	}

	for i, v := range b {
		a[i] -= v
	}

	return nil
}

func (a i32s) Sub(other Number) error {
	b, err := getInt32s(other)
	if err != nil {
		return errors.Wrapf(err, opFail, "Sub")
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Sub", len(a), len(b))
	}

	for i, v := range b {
		a[i] -= v
	}

	return nil
}

func (a u8s) Sub(other Number) error {
	b, err := getBytes(other)
	if err != nil {
		return errors.Wrapf(err, opFail, "Sub")
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Sub", len(a), len(b))
	}

	for i, v := range b {
		a[i] -= v
	}

	return nil
}

/* Mul */

func (a f64s) Mul(other Number) error {
	b, err := getFloat64s(other)
	if err != nil {
		return errors.Wrapf(err, opFail, "Mul")
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Mul", len(a), len(b))
	}

	vecf64.Mul([]float64(a), b)
	return nil
}

func (a f32s) Mul(other Number) error {
	b, err := getFloat32s(other)
	if err != nil {
		return errors.Wrapf(err, opFail, "Mul")
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Mul", len(a), len(b))
	}

	vecf32.Mul([]float32(a), b)
	return nil
}

func (a ints) Mul(other Number) error {
	b, err := getInts(other)
	if err != nil {
		return errors.Wrapf(err, opFail, "Mul")
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Mul", len(a), len(b))
	}

	for i, v := range b {
		a[i] *= v
	}

	return nil
}

func (a i64s) Mul(other Number) error {
	b, err := getInt64s(other)
	if err != nil {
		return errors.Wrapf(err, opFail, "Mul")
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Mul", len(a), len(b))
	}

	for i, v := range b {
		a[i] *= v
	}

	return nil
}

func (a i32s) Mul(other Number) error {
	b, err := getInt32s(other)
	if err != nil {
		return errors.Wrapf(err, opFail, "Mul")
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Mul", len(a), len(b))
	}

	for i, v := range b {
		a[i] *= v
	}

	return nil
}

func (a u8s) Mul(other Number) error {
	b, err := getBytes(other)
	if err != nil {
		return errors.Wrapf(err, opFail, "Mul")
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Mul", len(a), len(b))
	}

	for i, v := range b {
		a[i] *= v
	}

	return nil
}

/* Div */

func (a f64s) Div(other Number) error {
	b, err := getFloat64s(other)
	if err != nil {
		return errors.Wrapf(err, opFail, "Div")
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Div", len(a), len(b))
	}

	vecf64.Div([]float64(a), b)
	return nil
}

func (a f32s) Div(other Number) error {
	b, err := getFloat32s(other)
	if err != nil {
		return errors.Wrapf(err, opFail, "Div")
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Div", len(a), len(b))
	}

	vecf32.Div([]float32(a), b)
	return nil
}

func (a ints) Div(other Number) error {
	b, err := getInts(other)
	if err != nil {
		return errors.Wrapf(err, opFail, "Div")
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Div", len(a), len(b))
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

func (a i64s) Div(other Number) error {
	b, err := getInt64s(other)
	if err != nil {
		return errors.Wrapf(err, opFail, "Div")
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Div", len(a), len(b))
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

func (a i32s) Div(other Number) error {
	b, err := getInt32s(other)
	if err != nil {
		return errors.Wrapf(err, opFail, "Div")
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Div", len(a), len(b))
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

func (a u8s) Div(other Number) error {
	b, err := getBytes(other)
	if err != nil {
		return errors.Wrapf(err, opFail, "Div")
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Div", len(a), len(b))
	}

	var errs errorIndices
	for i, v := range b {
		if v == byte(0) {
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

/* Pow */

func (a f64s) Pow(other Number) error {
	b, err := getFloat64s(other)
	if err != nil {
		return errors.Wrapf(err, opFail, "Pow")
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Pow", len(a), len(b))
	}

	vecf64.Pow([]float64(a), b)
	return nil
}

func (a f32s) Pow(other Number) error {
	b, err := getFloat32s(other)
	if err != nil {
		return errors.Wrapf(err, opFail, "Pow")
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Pow", len(a), len(b))
	}

	vecf32.Pow([]float32(a), b)
	return nil
}

func (a ints) Pow(other Number) error {
	b, err := getInts(other)
	if err != nil {
		return errors.Wrapf(err, opFail, "Pow")
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Pow", len(a), len(b))
	}

	for i, v := range b {
		a[i] = int(math.Pow(float64(a[i]), float64(v)))
	}

	return nil
}

func (a i64s) Pow(other Number) error {
	b, err := getInt64s(other)
	if err != nil {
		return errors.Wrapf(err, opFail, "Pow")
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Pow", len(a), len(b))
	}

	for i, v := range b {
		a[i] = int64(math.Pow(float64(a[i]), float64(v)))
	}

	return nil
}

func (a i32s) Pow(other Number) error {
	b, err := getInt32s(other)
	if err != nil {
		return errors.Wrapf(err, opFail, "Pow")
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Pow", len(a), len(b))
	}

	for i, v := range b {
		a[i] = int32(math.Pow(float64(a[i]), float64(v)))
	}

	return nil
}

func (a u8s) Pow(other Number) error {
	b, err := getBytes(other)
	if err != nil {
		return errors.Wrapf(err, opFail, "Pow")
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Pow", len(a), len(b))
	}

	for i, v := range b {
		a[i] = byte(math.Pow(float64(a[i]), float64(v)))
	}

	return nil
}

/* Trans */

func (a f64s) Trans(other interface{}) (err error) {
	var b float64
	if b, err = getFloat64(other); err != nil {
		return errors.Wrapf(err, opFail, "Trans")
	}

	vecf64.Trans([]float64(a), b)
	return nil
}

func (a f32s) Trans(other interface{}) (err error) {
	var b float32
	if b, err = getFloat32(other); err != nil {
		return errors.Wrapf(err, opFail, "Trans")
	}

	vecf32.Trans([]float32(a), b)
	return nil
}

func (a ints) Trans(other interface{}) (err error) {
	var b int
	if b, err = getInt(other); err != nil {
		return errors.Wrapf(err, opFail, "Trans")
	}

	for i, v := range a {
		a[i] = v + b
	}
	return nil
}

func (a i64s) Trans(other interface{}) (err error) {
	var b int64
	if b, err = getInt64(other); err != nil {
		return errors.Wrapf(err, opFail, "Trans")
	}

	for i, v := range a {
		a[i] = v + b
	}
	return nil
}

func (a i32s) Trans(other interface{}) (err error) {
	var b int32
	if b, err = getInt32(other); err != nil {
		return errors.Wrapf(err, opFail, "Trans")
	}

	for i, v := range a {
		a[i] = v + b
	}
	return nil
}

func (a u8s) Trans(other interface{}) (err error) {
	var b byte
	if b, err = getByte(other); err != nil {
		return errors.Wrapf(err, opFail, "Trans")
	}

	for i, v := range a {
		a[i] = v + b
	}
	return nil
}

/* TransInv */

func (a f64s) TransInv(other interface{}) (err error) {
	var b float64
	if b, err = getFloat64(other); err != nil {
		return errors.Wrapf(err, opFail, "TransInv")
	}

	vecf64.TransInv([]float64(a), b)
	return nil
}

func (a f32s) TransInv(other interface{}) (err error) {
	var b float32
	if b, err = getFloat32(other); err != nil {
		return errors.Wrapf(err, opFail, "TransInv")
	}

	vecf32.TransInv([]float32(a), b)
	return nil
}

func (a ints) TransInv(other interface{}) (err error) {
	var b int
	if b, err = getInt(other); err != nil {
		return errors.Wrapf(err, opFail, "TransInv")
	}

	for i, v := range a {
		a[i] = v - b
	}
	return nil
}

func (a i64s) TransInv(other interface{}) (err error) {
	var b int64
	if b, err = getInt64(other); err != nil {
		return errors.Wrapf(err, opFail, "TransInv")
	}

	for i, v := range a {
		a[i] = v - b
	}
	return nil
}

func (a i32s) TransInv(other interface{}) (err error) {
	var b int32
	if b, err = getInt32(other); err != nil {
		return errors.Wrapf(err, opFail, "TransInv")
	}

	for i, v := range a {
		a[i] = v - b
	}
	return nil
}

func (a u8s) TransInv(other interface{}) (err error) {
	var b byte
	if b, err = getByte(other); err != nil {
		return errors.Wrapf(err, opFail, "TransInv")
	}

	for i, v := range a {
		a[i] = v - b
	}
	return nil
}

/* TransInvR */

func (a f64s) TransInvR(other interface{}) (err error) {
	var b float64
	if b, err = getFloat64(other); err != nil {
		return errors.Wrapf(err, opFail, "TransInvR")
	}

	vecf64.TransInvR([]float64(a), b)
	return nil
}

func (a f32s) TransInvR(other interface{}) (err error) {
	var b float32
	if b, err = getFloat32(other); err != nil {
		return errors.Wrapf(err, opFail, "TransInvR")
	}

	vecf32.TransInvR([]float32(a), b)
	return nil
}

func (a ints) TransInvR(other interface{}) (err error) {
	var b int
	if b, err = getInt(other); err != nil {
		return errors.Wrapf(err, opFail, "TransInvR")
	}

	for i, v := range a {
		a[i] = b - v
	}
	return nil
}

func (a i64s) TransInvR(other interface{}) (err error) {
	var b int64
	if b, err = getInt64(other); err != nil {
		return errors.Wrapf(err, opFail, "TransInvR")
	}

	for i, v := range a {
		a[i] = b - v
	}
	return nil
}

func (a i32s) TransInvR(other interface{}) (err error) {
	var b int32
	if b, err = getInt32(other); err != nil {
		return errors.Wrapf(err, opFail, "TransInvR")
	}

	for i, v := range a {
		a[i] = b - v
	}
	return nil
}

func (a u8s) TransInvR(other interface{}) (err error) {
	var b byte
	if b, err = getByte(other); err != nil {
		return errors.Wrapf(err, opFail, "TransInvR")
	}

	for i, v := range a {
		a[i] = b - v
	}
	return nil
}

/* Scale */

func (a f64s) Scale(other interface{}) (err error) {
	var b float64
	if b, err = getFloat64(other); err != nil {
		return errors.Wrapf(err, opFail, "Scale")
	}

	vecf64.Scale([]float64(a), b)
	return nil
}

func (a f32s) Scale(other interface{}) (err error) {
	var b float32
	if b, err = getFloat32(other); err != nil {
		return errors.Wrapf(err, opFail, "Scale")
	}

	vecf32.Scale([]float32(a), b)
	return nil
}

func (a ints) Scale(other interface{}) (err error) {
	var b int
	if b, err = getInt(other); err != nil {
		return errors.Wrapf(err, opFail, "Scale")
	}

	for i, v := range a {
		a[i] = v * b
	}
	return nil
}

func (a i64s) Scale(other interface{}) (err error) {
	var b int64
	if b, err = getInt64(other); err != nil {
		return errors.Wrapf(err, opFail, "Scale")
	}

	for i, v := range a {
		a[i] = v * b
	}
	return nil
}

func (a i32s) Scale(other interface{}) (err error) {
	var b int32
	if b, err = getInt32(other); err != nil {
		return errors.Wrapf(err, opFail, "Scale")
	}

	for i, v := range a {
		a[i] = v * b
	}
	return nil
}

func (a u8s) Scale(other interface{}) (err error) {
	var b byte
	if b, err = getByte(other); err != nil {
		return errors.Wrapf(err, opFail, "Scale")
	}

	for i, v := range a {
		a[i] = v * b
	}
	return nil
}

/* ScaleInv */

func (a f64s) ScaleInv(other interface{}) (err error) {
	var b float64
	if b, err = getFloat64(other); err != nil {
		return errors.Wrapf(err, opFail, "ScaleInv")
	}

	vecf64.ScaleInv([]float64(a), b)
	return nil
}

func (a f32s) ScaleInv(other interface{}) (err error) {
	var b float32
	if b, err = getFloat32(other); err != nil {
		return errors.Wrapf(err, opFail, "ScaleInv")
	}

	vecf32.ScaleInv([]float32(a), b)
	return nil
}

func (a ints) ScaleInv(other interface{}) (err error) {
	var b int
	if b, err = getInt(other); err != nil {
		return errors.Wrapf(err, opFail, "ScaleInv")
	}

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

func (a i64s) ScaleInv(other interface{}) (err error) {
	var b int64
	if b, err = getInt64(other); err != nil {
		return errors.Wrapf(err, opFail, "ScaleInv")
	}

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

func (a i32s) ScaleInv(other interface{}) (err error) {
	var b int32
	if b, err = getInt32(other); err != nil {
		return errors.Wrapf(err, opFail, "ScaleInv")
	}

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

func (a u8s) ScaleInv(other interface{}) (err error) {
	var b byte
	if b, err = getByte(other); err != nil {
		return errors.Wrapf(err, opFail, "ScaleInv")
	}

	var errs errorIndices
	for i, v := range a {
		if v == byte(0) {
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

/* ScaleInvR */

func (a f64s) ScaleInvR(other interface{}) (err error) {
	var b float64
	if b, err = getFloat64(other); err != nil {
		return errors.Wrapf(err, opFail, "ScaleInvR")
	}

	vecf64.ScaleInvR([]float64(a), b)
	return nil
}

func (a f32s) ScaleInvR(other interface{}) (err error) {
	var b float32
	if b, err = getFloat32(other); err != nil {
		return errors.Wrapf(err, opFail, "ScaleInvR")
	}

	vecf32.ScaleInvR([]float32(a), b)
	return nil
}

func (a ints) ScaleInvR(other interface{}) (err error) {
	var b int
	if b, err = getInt(other); err != nil {
		return errors.Wrapf(err, opFail, "ScaleInvR")
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

func (a i64s) ScaleInvR(other interface{}) (err error) {
	var b int64
	if b, err = getInt64(other); err != nil {
		return errors.Wrapf(err, opFail, "ScaleInvR")
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

func (a i32s) ScaleInvR(other interface{}) (err error) {
	var b int32
	if b, err = getInt32(other); err != nil {
		return errors.Wrapf(err, opFail, "ScaleInvR")
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

func (a u8s) ScaleInvR(other interface{}) (err error) {
	var b byte
	if b, err = getByte(other); err != nil {
		return errors.Wrapf(err, opFail, "ScaleInvR")
	}

	var errs errorIndices
	for i, v := range a {
		if v == byte(0) {
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

/* PowOf */

func (a f64s) PowOf(other interface{}) (err error) {
	var b float64
	if b, err = getFloat64(other); err != nil {
		return errors.Wrapf(err, opFail, "PowOf")
	}

	vecf64.PowOf([]float64(a), b)
	return nil
}

func (a f32s) PowOf(other interface{}) (err error) {
	var b float32
	if b, err = getFloat32(other); err != nil {
		return errors.Wrapf(err, opFail, "PowOf")
	}

	vecf32.PowOf([]float32(a), b)
	return nil
}

func (a ints) PowOf(other interface{}) (err error) {
	var b int
	if b, err = getInt(other); err != nil {
		return errors.Wrapf(err, opFail, "PowOf")
	}

	for i, v := range a {
		a[i] = int(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func (a i64s) PowOf(other interface{}) (err error) {
	var b int64
	if b, err = getInt64(other); err != nil {
		return errors.Wrapf(err, opFail, "PowOf")
	}

	for i, v := range a {
		a[i] = int64(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func (a i32s) PowOf(other interface{}) (err error) {
	var b int32
	if b, err = getInt32(other); err != nil {
		return errors.Wrapf(err, opFail, "PowOf")
	}

	for i, v := range a {
		a[i] = int32(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func (a u8s) PowOf(other interface{}) (err error) {
	var b byte
	if b, err = getByte(other); err != nil {
		return errors.Wrapf(err, opFail, "PowOf")
	}

	for i, v := range a {
		a[i] = byte(math.Pow(float64(v), float64(b)))
	}
	return nil
}

/* PowOfR */

func (a f64s) PowOfR(other interface{}) (err error) {
	var b float64
	if b, err = getFloat64(other); err != nil {
		return errors.Wrapf(err, opFail, "PowOfR")
	}

	vecf64.PowOfR([]float64(a), b)
	return nil
}

func (a f32s) PowOfR(other interface{}) (err error) {
	var b float32
	if b, err = getFloat32(other); err != nil {
		return errors.Wrapf(err, opFail, "PowOfR")
	}

	vecf32.PowOfR([]float32(a), b)
	return nil
}

func (a ints) PowOfR(other interface{}) (err error) {
	var b int
	if b, err = getInt(other); err != nil {
		return errors.Wrapf(err, opFail, "PowOfR")
	}

	for i, v := range a {
		a[i] = int(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func (a i64s) PowOfR(other interface{}) (err error) {
	var b int64
	if b, err = getInt64(other); err != nil {
		return errors.Wrapf(err, opFail, "PowOfR")
	}

	for i, v := range a {
		a[i] = int64(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func (a i32s) PowOfR(other interface{}) (err error) {
	var b int32
	if b, err = getInt32(other); err != nil {
		return errors.Wrapf(err, opFail, "PowOfR")
	}

	for i, v := range a {
		a[i] = int32(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func (a u8s) PowOfR(other interface{}) (err error) {
	var b byte
	if b, err = getByte(other); err != nil {
		return errors.Wrapf(err, opFail, "PowOfR")
	}

	for i, v := range a {
		a[i] = byte(math.Pow(float64(b), float64(v)))
	}
	return nil
}
