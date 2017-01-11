package tensor

import (
	"math"

	"github.com/chewxy/vecf32"
	"github.com/chewxy/vecf64"
	"github.com/pkg/errors"
)

/* Add */

func (a f64s) IncrAdd(other, incrArr Number) (err error) {
	var b, incr []float64
	if b, err = getFloat64s(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrAdd")
	}

	if incr, err = getFloat64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrAdd")
	}

	if len(b) != len(a) {
		return errors.Errorf("lenMismatch", "IncrAdd", len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrAdd", len(a), len(incr))
	}

	vecf64.IncrAdd([]float64(a), b, incr)
	return nil
}

func (a f32s) IncrAdd(other, incrArr Number) (err error) {
	var b, incr []float32
	if b, err = getFloat32s(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrAdd")
	}

	if incr, err = getFloat32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrAdd")
	}

	if len(b) != len(a) {
		return errors.Errorf("lenMismatch", "IncrAdd", len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrAdd", len(a), len(incr))
	}

	vecf32.IncrAdd([]float32(a), b, incr)
	return nil
}

func (a ints) IncrAdd(other, incrArr Number) (err error) {
	var b, incr []int
	if b, err = getInts(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrAdd")
	}

	if incr, err = getInts(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrAdd")
	}

	if len(b) != len(a) {
		return errors.Errorf("lenMismatch", "IncrAdd", len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrAdd", len(a), len(incr))
	}

	for i, v := range b {
		incr[i] += a[i] + v
	}

	return nil
}

func (a i64s) IncrAdd(other, incrArr Number) (err error) {
	var b, incr []int64
	if b, err = getInt64s(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrAdd")
	}

	if incr, err = getInt64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrAdd")
	}

	if len(b) != len(a) {
		return errors.Errorf("lenMismatch", "IncrAdd", len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrAdd", len(a), len(incr))
	}

	for i, v := range b {
		incr[i] += a[i] + v
	}

	return nil
}

func (a i32s) IncrAdd(other, incrArr Number) (err error) {
	var b, incr []int32
	if b, err = getInt32s(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrAdd")
	}

	if incr, err = getInt32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrAdd")
	}

	if len(b) != len(a) {
		return errors.Errorf("lenMismatch", "IncrAdd", len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrAdd", len(a), len(incr))
	}

	for i, v := range b {
		incr[i] += a[i] + v
	}

	return nil
}

func (a u8s) IncrAdd(other, incrArr Number) (err error) {
	var b, incr []byte
	if b, err = getBytes(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrAdd")
	}

	if incr, err = getBytes(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrAdd")
	}

	if len(b) != len(a) {
		return errors.Errorf("lenMismatch", "IncrAdd", len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrAdd", len(a), len(incr))
	}

	for i, v := range b {
		incr[i] += a[i] + v
	}

	return nil
}

/* Sub */

func (a f64s) IncrSub(other, incrArr Number) (err error) {
	var b, incr []float64
	if b, err = getFloat64s(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrSub")
	}

	if incr, err = getFloat64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrSub")
	}

	if len(b) != len(a) {
		return errors.Errorf("lenMismatch", "IncrSub", len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrSub", len(a), len(incr))
	}

	vecf64.IncrSub([]float64(a), b, incr)
	return nil
}

func (a f32s) IncrSub(other, incrArr Number) (err error) {
	var b, incr []float32
	if b, err = getFloat32s(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrSub")
	}

	if incr, err = getFloat32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrSub")
	}

	if len(b) != len(a) {
		return errors.Errorf("lenMismatch", "IncrSub", len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrSub", len(a), len(incr))
	}

	vecf32.IncrSub([]float32(a), b, incr)
	return nil
}

func (a ints) IncrSub(other, incrArr Number) (err error) {
	var b, incr []int
	if b, err = getInts(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrSub")
	}

	if incr, err = getInts(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrSub")
	}

	if len(b) != len(a) {
		return errors.Errorf("lenMismatch", "IncrSub", len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrSub", len(a), len(incr))
	}

	for i, v := range b {
		incr[i] += a[i] - v
	}

	return nil
}

func (a i64s) IncrSub(other, incrArr Number) (err error) {
	var b, incr []int64
	if b, err = getInt64s(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrSub")
	}

	if incr, err = getInt64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrSub")
	}

	if len(b) != len(a) {
		return errors.Errorf("lenMismatch", "IncrSub", len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrSub", len(a), len(incr))
	}

	for i, v := range b {
		incr[i] += a[i] - v
	}

	return nil
}

func (a i32s) IncrSub(other, incrArr Number) (err error) {
	var b, incr []int32
	if b, err = getInt32s(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrSub")
	}

	if incr, err = getInt32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrSub")
	}

	if len(b) != len(a) {
		return errors.Errorf("lenMismatch", "IncrSub", len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrSub", len(a), len(incr))
	}

	for i, v := range b {
		incr[i] += a[i] - v
	}

	return nil
}

func (a u8s) IncrSub(other, incrArr Number) (err error) {
	var b, incr []byte
	if b, err = getBytes(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrSub")
	}

	if incr, err = getBytes(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrSub")
	}

	if len(b) != len(a) {
		return errors.Errorf("lenMismatch", "IncrSub", len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrSub", len(a), len(incr))
	}

	for i, v := range b {
		incr[i] += a[i] - v
	}

	return nil
}

/* Mul */

func (a f64s) IncrMul(other, incrArr Number) (err error) {
	var b, incr []float64
	if b, err = getFloat64s(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrMul")
	}

	if incr, err = getFloat64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrMul")
	}

	if len(b) != len(a) {
		return errors.Errorf("lenMismatch", "IncrMul", len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrMul", len(a), len(incr))
	}

	vecf64.IncrMul([]float64(a), b, incr)
	return nil
}

func (a f32s) IncrMul(other, incrArr Number) (err error) {
	var b, incr []float32
	if b, err = getFloat32s(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrMul")
	}

	if incr, err = getFloat32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrMul")
	}

	if len(b) != len(a) {
		return errors.Errorf("lenMismatch", "IncrMul", len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrMul", len(a), len(incr))
	}

	vecf32.IncrMul([]float32(a), b, incr)
	return nil
}

func (a ints) IncrMul(other, incrArr Number) (err error) {
	var b, incr []int
	if b, err = getInts(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrMul")
	}

	if incr, err = getInts(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrMul")
	}

	if len(b) != len(a) {
		return errors.Errorf("lenMismatch", "IncrMul", len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrMul", len(a), len(incr))
	}

	for i, v := range b {
		incr[i] += a[i] * v
	}

	return nil
}

func (a i64s) IncrMul(other, incrArr Number) (err error) {
	var b, incr []int64
	if b, err = getInt64s(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrMul")
	}

	if incr, err = getInt64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrMul")
	}

	if len(b) != len(a) {
		return errors.Errorf("lenMismatch", "IncrMul", len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrMul", len(a), len(incr))
	}

	for i, v := range b {
		incr[i] += a[i] * v
	}

	return nil
}

func (a i32s) IncrMul(other, incrArr Number) (err error) {
	var b, incr []int32
	if b, err = getInt32s(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrMul")
	}

	if incr, err = getInt32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrMul")
	}

	if len(b) != len(a) {
		return errors.Errorf("lenMismatch", "IncrMul", len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrMul", len(a), len(incr))
	}

	for i, v := range b {
		incr[i] += a[i] * v
	}

	return nil
}

func (a u8s) IncrMul(other, incrArr Number) (err error) {
	var b, incr []byte
	if b, err = getBytes(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrMul")
	}

	if incr, err = getBytes(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrMul")
	}

	if len(b) != len(a) {
		return errors.Errorf("lenMismatch", "IncrMul", len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrMul", len(a), len(incr))
	}

	for i, v := range b {
		incr[i] += a[i] * v
	}

	return nil
}

/* Div */

func (a f64s) IncrDiv(other, incrArr Number) (err error) {
	var b, incr []float64
	if b, err = getFloat64s(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrDiv")
	}

	if incr, err = getFloat64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrDiv")
	}

	if len(b) != len(a) {
		return errors.Errorf("lenMismatch", "IncrDiv", len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrDiv", len(a), len(incr))
	}

	vecf64.IncrDiv([]float64(a), b, incr)
	return nil
}

func (a f32s) IncrDiv(other, incrArr Number) (err error) {
	var b, incr []float32
	if b, err = getFloat32s(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrDiv")
	}

	if incr, err = getFloat32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrDiv")
	}

	if len(b) != len(a) {
		return errors.Errorf("lenMismatch", "IncrDiv", len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrDiv", len(a), len(incr))
	}

	vecf32.IncrDiv([]float32(a), b, incr)
	return nil
}

func (a ints) IncrDiv(other, incrArr Number) (err error) {
	var b, incr []int
	if b, err = getInts(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrDiv")
	}

	if incr, err = getInts(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrDiv")
	}

	if len(b) != len(a) {
		return errors.Errorf("lenMismatch", "IncrDiv", len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrDiv", len(a), len(incr))
	}

	var errs errorIndices
	for i, v := range b {
		if v == int(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}

		incr[i] += a[i] / v
	}

	if errs != nil {
		return errs
	}
	return nil
}

func (a i64s) IncrDiv(other, incrArr Number) (err error) {
	var b, incr []int64
	if b, err = getInt64s(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrDiv")
	}

	if incr, err = getInt64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrDiv")
	}

	if len(b) != len(a) {
		return errors.Errorf("lenMismatch", "IncrDiv", len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrDiv", len(a), len(incr))
	}

	var errs errorIndices
	for i, v := range b {
		if v == int64(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}

		incr[i] += a[i] / v
	}

	if errs != nil {
		return errs
	}
	return nil
}

func (a i32s) IncrDiv(other, incrArr Number) (err error) {
	var b, incr []int32
	if b, err = getInt32s(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrDiv")
	}

	if incr, err = getInt32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrDiv")
	}

	if len(b) != len(a) {
		return errors.Errorf("lenMismatch", "IncrDiv", len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrDiv", len(a), len(incr))
	}

	var errs errorIndices
	for i, v := range b {
		if v == int32(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}

		incr[i] += a[i] / v
	}

	if errs != nil {
		return errs
	}
	return nil
}

func (a u8s) IncrDiv(other, incrArr Number) (err error) {
	var b, incr []byte
	if b, err = getBytes(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrDiv")
	}

	if incr, err = getBytes(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrDiv")
	}

	if len(b) != len(a) {
		return errors.Errorf("lenMismatch", "IncrDiv", len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrDiv", len(a), len(incr))
	}

	var errs errorIndices
	for i, v := range b {
		if v == byte(0) {
			errs = append(errs, i)
			a[i] = 0
			continue
		}

		incr[i] += a[i] / v
	}

	if errs != nil {
		return errs
	}
	return nil
}

/* Pow */

func (a f64s) IncrPow(other, incrArr Number) (err error) {
	var b, incr []float64
	if b, err = getFloat64s(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrPow")
	}

	if incr, err = getFloat64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrPow")
	}

	if len(b) != len(a) {
		return errors.Errorf("lenMismatch", "IncrPow", len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrPow", len(a), len(incr))
	}

	vecf64.IncrPow([]float64(a), b, incr)
	return nil
}

func (a f32s) IncrPow(other, incrArr Number) (err error) {
	var b, incr []float32
	if b, err = getFloat32s(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrPow")
	}

	if incr, err = getFloat32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrPow")
	}

	if len(b) != len(a) {
		return errors.Errorf("lenMismatch", "IncrPow", len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrPow", len(a), len(incr))
	}

	vecf32.IncrPow([]float32(a), b, incr)
	return nil
}

func (a ints) IncrPow(other, incrArr Number) (err error) {
	var b, incr []int
	if b, err = getInts(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrPow")
	}

	if incr, err = getInts(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrPow")
	}

	if len(b) != len(a) {
		return errors.Errorf("lenMismatch", "IncrPow", len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrPow", len(a), len(incr))
	}

	for i, v := range b {
		incr[i] += int(math.Pow(float64(a[i]), float64(v)))
	}

	return nil
}

func (a i64s) IncrPow(other, incrArr Number) (err error) {
	var b, incr []int64
	if b, err = getInt64s(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrPow")
	}

	if incr, err = getInt64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrPow")
	}

	if len(b) != len(a) {
		return errors.Errorf("lenMismatch", "IncrPow", len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrPow", len(a), len(incr))
	}

	for i, v := range b {
		incr[i] += int64(math.Pow(float64(a[i]), float64(v)))
	}

	return nil
}

func (a i32s) IncrPow(other, incrArr Number) (err error) {
	var b, incr []int32
	if b, err = getInt32s(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrPow")
	}

	if incr, err = getInt32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrPow")
	}

	if len(b) != len(a) {
		return errors.Errorf("lenMismatch", "IncrPow", len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrPow", len(a), len(incr))
	}

	for i, v := range b {
		incr[i] += int32(math.Pow(float64(a[i]), float64(v)))
	}

	return nil
}

func (a u8s) IncrPow(other, incrArr Number) (err error) {
	var b, incr []byte
	if b, err = getBytes(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrPow")
	}

	if incr, err = getBytes(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrPow")
	}

	if len(b) != len(a) {
		return errors.Errorf("lenMismatch", "IncrPow", len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrPow", len(a), len(incr))
	}

	for i, v := range b {
		incr[i] += byte(math.Pow(float64(a[i]), float64(v)))
	}

	return nil
}

/* Trans */

func (a f64s) IncrTrans(other interface{}, incrArr Number) (err error) {
	var b float64
	if b, err = getFloat64(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrTrans")
	}

	var incr []float64
	if incr, err = getFloat64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrTrans")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrTrans", len(a), len(incr))
	}

	vecf64.IncrTrans([]float64(a), b, incr)
	return nil
}

func (a f32s) IncrTrans(other interface{}, incrArr Number) (err error) {
	var b float32
	if b, err = getFloat32(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrTrans")
	}

	var incr []float32
	if incr, err = getFloat32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrTrans")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrTrans", len(a), len(incr))
	}

	vecf32.IncrTrans([]float32(a), b, incr)
	return nil
}

func (a ints) IncrTrans(other interface{}, incrArr Number) (err error) {
	var b int
	if b, err = getInt(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrTrans")
	}

	var incr []int
	if incr, err = getInts(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrTrans")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrTrans", len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += v + b
	}
	return nil
}

func (a i64s) IncrTrans(other interface{}, incrArr Number) (err error) {
	var b int64
	if b, err = getInt64(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrTrans")
	}

	var incr []int64
	if incr, err = getInt64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrTrans")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrTrans", len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += v + b
	}
	return nil
}

func (a i32s) IncrTrans(other interface{}, incrArr Number) (err error) {
	var b int32
	if b, err = getInt32(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrTrans")
	}

	var incr []int32
	if incr, err = getInt32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrTrans")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrTrans", len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += v + b
	}
	return nil
}

func (a u8s) IncrTrans(other interface{}, incrArr Number) (err error) {
	var b byte
	if b, err = getByte(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrTrans")
	}

	var incr []byte
	if incr, err = getBytes(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrTrans")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrTrans", len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += v + b
	}
	return nil
}

/* TransInv */

func (a f64s) IncrTransInv(other interface{}, incrArr Number) (err error) {
	var b float64
	if b, err = getFloat64(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInv")
	}

	var incr []float64
	if incr, err = getFloat64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInv")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrTransInv", len(a), len(incr))
	}

	vecf64.IncrTransInv([]float64(a), b, incr)
	return nil
}

func (a f32s) IncrTransInv(other interface{}, incrArr Number) (err error) {
	var b float32
	if b, err = getFloat32(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInv")
	}

	var incr []float32
	if incr, err = getFloat32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInv")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrTransInv", len(a), len(incr))
	}

	vecf32.IncrTransInv([]float32(a), b, incr)
	return nil
}

func (a ints) IncrTransInv(other interface{}, incrArr Number) (err error) {
	var b int
	if b, err = getInt(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInv")
	}

	var incr []int
	if incr, err = getInts(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInv")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrTransInv", len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += v - b
	}
	return nil
}

func (a i64s) IncrTransInv(other interface{}, incrArr Number) (err error) {
	var b int64
	if b, err = getInt64(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInv")
	}

	var incr []int64
	if incr, err = getInt64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInv")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrTransInv", len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += v - b
	}
	return nil
}

func (a i32s) IncrTransInv(other interface{}, incrArr Number) (err error) {
	var b int32
	if b, err = getInt32(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInv")
	}

	var incr []int32
	if incr, err = getInt32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInv")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrTransInv", len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += v - b
	}
	return nil
}

func (a u8s) IncrTransInv(other interface{}, incrArr Number) (err error) {
	var b byte
	if b, err = getByte(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInv")
	}

	var incr []byte
	if incr, err = getBytes(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInv")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrTransInv", len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += v - b
	}
	return nil
}

/* TransInvR */

func (a f64s) IncrTransInvR(other interface{}, incrArr Number) (err error) {
	var b float64
	if b, err = getFloat64(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInvR")
	}

	var incr []float64
	if incr, err = getFloat64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInvR")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrTransInvR", len(a), len(incr))
	}

	vecf64.IncrTransInvR([]float64(a), b, incr)
	return nil
}

func (a f32s) IncrTransInvR(other interface{}, incrArr Number) (err error) {
	var b float32
	if b, err = getFloat32(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInvR")
	}

	var incr []float32
	if incr, err = getFloat32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInvR")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrTransInvR", len(a), len(incr))
	}

	vecf32.IncrTransInvR([]float32(a), b, incr)
	return nil
}

func (a ints) IncrTransInvR(other interface{}, incrArr Number) (err error) {
	var b int
	if b, err = getInt(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInvR")
	}

	var incr []int
	if incr, err = getInts(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInvR")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrTransInvR", len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += b - v
	}
	return nil
}

func (a i64s) IncrTransInvR(other interface{}, incrArr Number) (err error) {
	var b int64
	if b, err = getInt64(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInvR")
	}

	var incr []int64
	if incr, err = getInt64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInvR")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrTransInvR", len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += b - v
	}
	return nil
}

func (a i32s) IncrTransInvR(other interface{}, incrArr Number) (err error) {
	var b int32
	if b, err = getInt32(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInvR")
	}

	var incr []int32
	if incr, err = getInt32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInvR")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrTransInvR", len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += b - v
	}
	return nil
}

func (a u8s) IncrTransInvR(other interface{}, incrArr Number) (err error) {
	var b byte
	if b, err = getByte(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInvR")
	}

	var incr []byte
	if incr, err = getBytes(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInvR")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrTransInvR", len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += b - v
	}
	return nil
}

/* Scale */

func (a f64s) IncrScale(other interface{}, incrArr Number) (err error) {
	var b float64
	if b, err = getFloat64(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrScale")
	}

	var incr []float64
	if incr, err = getFloat64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrScale")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrScale", len(a), len(incr))
	}

	vecf64.IncrScale([]float64(a), b, incr)
	return nil
}

func (a f32s) IncrScale(other interface{}, incrArr Number) (err error) {
	var b float32
	if b, err = getFloat32(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrScale")
	}

	var incr []float32
	if incr, err = getFloat32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrScale")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrScale", len(a), len(incr))
	}

	vecf32.IncrScale([]float32(a), b, incr)
	return nil
}

func (a ints) IncrScale(other interface{}, incrArr Number) (err error) {
	var b int
	if b, err = getInt(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrScale")
	}

	var incr []int
	if incr, err = getInts(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrScale")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrScale", len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += v * b
	}
	return nil
}

func (a i64s) IncrScale(other interface{}, incrArr Number) (err error) {
	var b int64
	if b, err = getInt64(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrScale")
	}

	var incr []int64
	if incr, err = getInt64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrScale")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrScale", len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += v * b
	}
	return nil
}

func (a i32s) IncrScale(other interface{}, incrArr Number) (err error) {
	var b int32
	if b, err = getInt32(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrScale")
	}

	var incr []int32
	if incr, err = getInt32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrScale")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrScale", len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += v * b
	}
	return nil
}

func (a u8s) IncrScale(other interface{}, incrArr Number) (err error) {
	var b byte
	if b, err = getByte(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrScale")
	}

	var incr []byte
	if incr, err = getBytes(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrScale")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrScale", len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += v * b
	}
	return nil
}

/* ScaleInv */

func (a f64s) IncrScaleInv(other interface{}, incrArr Number) (err error) {
	var b float64
	if b, err = getFloat64(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInv")
	}

	var incr []float64
	if incr, err = getFloat64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInv")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrScaleInv", len(a), len(incr))
	}

	vecf64.IncrScaleInv([]float64(a), b, incr)
	return nil
}

func (a f32s) IncrScaleInv(other interface{}, incrArr Number) (err error) {
	var b float32
	if b, err = getFloat32(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInv")
	}

	var incr []float32
	if incr, err = getFloat32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInv")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrScaleInv", len(a), len(incr))
	}

	vecf32.IncrScaleInv([]float32(a), b, incr)
	return nil
}

func (a ints) IncrScaleInv(other interface{}, incrArr Number) (err error) {
	var b int
	if b, err = getInt(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInv")
	}

	var incr []int
	if incr, err = getInts(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInv")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrScaleInv", len(a), len(incr))
	}

	var errs errorIndices
	for i, v := range a {
		if v == int(0) {
			errs = append(errs, i)
			incr[i] = 0
			continue
		}
		incr[i] += v / b
	}
	if errs != nil {
		return errs
	}
	return nil
}

func (a i64s) IncrScaleInv(other interface{}, incrArr Number) (err error) {
	var b int64
	if b, err = getInt64(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInv")
	}

	var incr []int64
	if incr, err = getInt64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInv")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrScaleInv", len(a), len(incr))
	}

	var errs errorIndices
	for i, v := range a {
		if v == int64(0) {
			errs = append(errs, i)
			incr[i] = 0
			continue
		}
		incr[i] += v / b
	}
	if errs != nil {
		return errs
	}
	return nil
}

func (a i32s) IncrScaleInv(other interface{}, incrArr Number) (err error) {
	var b int32
	if b, err = getInt32(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInv")
	}

	var incr []int32
	if incr, err = getInt32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInv")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrScaleInv", len(a), len(incr))
	}

	var errs errorIndices
	for i, v := range a {
		if v == int32(0) {
			errs = append(errs, i)
			incr[i] = 0
			continue
		}
		incr[i] += v / b
	}
	if errs != nil {
		return errs
	}
	return nil
}

func (a u8s) IncrScaleInv(other interface{}, incrArr Number) (err error) {
	var b byte
	if b, err = getByte(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInv")
	}

	var incr []byte
	if incr, err = getBytes(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInv")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrScaleInv", len(a), len(incr))
	}

	var errs errorIndices
	for i, v := range a {
		if v == byte(0) {
			errs = append(errs, i)
			incr[i] = 0
			continue
		}
		incr[i] += v / b
	}
	if errs != nil {
		return errs
	}
	return nil
}

/* ScaleInvR */

func (a f64s) IncrScaleInvR(other interface{}, incrArr Number) (err error) {
	var b float64
	if b, err = getFloat64(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInvR")
	}

	var incr []float64
	if incr, err = getFloat64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInvR")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrScaleInvR", len(a), len(incr))
	}

	vecf64.IncrScaleInvR([]float64(a), b, incr)
	return nil
}

func (a f32s) IncrScaleInvR(other interface{}, incrArr Number) (err error) {
	var b float32
	if b, err = getFloat32(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInvR")
	}

	var incr []float32
	if incr, err = getFloat32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInvR")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrScaleInvR", len(a), len(incr))
	}

	vecf32.IncrScaleInvR([]float32(a), b, incr)
	return nil
}

func (a ints) IncrScaleInvR(other interface{}, incrArr Number) (err error) {
	var b int
	if b, err = getInt(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInvR")
	}

	var incr []int
	if incr, err = getInts(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInvR")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrScaleInvR", len(a), len(incr))
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

func (a i64s) IncrScaleInvR(other interface{}, incrArr Number) (err error) {
	var b int64
	if b, err = getInt64(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInvR")
	}

	var incr []int64
	if incr, err = getInt64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInvR")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrScaleInvR", len(a), len(incr))
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

func (a i32s) IncrScaleInvR(other interface{}, incrArr Number) (err error) {
	var b int32
	if b, err = getInt32(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInvR")
	}

	var incr []int32
	if incr, err = getInt32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInvR")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrScaleInvR", len(a), len(incr))
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

func (a u8s) IncrScaleInvR(other interface{}, incrArr Number) (err error) {
	var b byte
	if b, err = getByte(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInvR")
	}

	var incr []byte
	if incr, err = getBytes(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInvR")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrScaleInvR", len(a), len(incr))
	}

	var errs errorIndices
	for i, v := range a {
		if v == byte(0) {
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

/* PowOf */

func (a f64s) IncrPowOf(other interface{}, incrArr Number) (err error) {
	var b float64
	if b, err = getFloat64(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOf")
	}

	var incr []float64
	if incr, err = getFloat64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOf")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrPowOf", len(a), len(incr))
	}

	vecf64.IncrPowOf([]float64(a), b, incr)
	return nil
}

func (a f32s) IncrPowOf(other interface{}, incrArr Number) (err error) {
	var b float32
	if b, err = getFloat32(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOf")
	}

	var incr []float32
	if incr, err = getFloat32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOf")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrPowOf", len(a), len(incr))
	}

	vecf32.IncrPowOf([]float32(a), b, incr)
	return nil
}

func (a ints) IncrPowOf(other interface{}, incrArr Number) (err error) {
	var b int
	if b, err = getInt(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOf")
	}

	var incr []int
	if incr, err = getInts(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOf")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrPowOf", len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += int(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func (a i64s) IncrPowOf(other interface{}, incrArr Number) (err error) {
	var b int64
	if b, err = getInt64(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOf")
	}

	var incr []int64
	if incr, err = getInt64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOf")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrPowOf", len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += int64(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func (a i32s) IncrPowOf(other interface{}, incrArr Number) (err error) {
	var b int32
	if b, err = getInt32(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOf")
	}

	var incr []int32
	if incr, err = getInt32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOf")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrPowOf", len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += int32(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func (a u8s) IncrPowOf(other interface{}, incrArr Number) (err error) {
	var b byte
	if b, err = getByte(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOf")
	}

	var incr []byte
	if incr, err = getBytes(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOf")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrPowOf", len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += byte(math.Pow(float64(v), float64(b)))
	}
	return nil
}

/* PowOfR */

func (a f64s) IncrPowOfR(other interface{}, incrArr Number) (err error) {
	var b float64
	if b, err = getFloat64(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOfR")
	}

	var incr []float64
	if incr, err = getFloat64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOfR")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrPowOfR", len(a), len(incr))
	}

	vecf64.IncrPowOfR([]float64(a), b, incr)
	return nil
}

func (a f32s) IncrPowOfR(other interface{}, incrArr Number) (err error) {
	var b float32
	if b, err = getFloat32(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOfR")
	}

	var incr []float32
	if incr, err = getFloat32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOfR")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrPowOfR", len(a), len(incr))
	}

	vecf32.IncrPowOfR([]float32(a), b, incr)
	return nil
}

func (a ints) IncrPowOfR(other interface{}, incrArr Number) (err error) {
	var b int
	if b, err = getInt(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOfR")
	}

	var incr []int
	if incr, err = getInts(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOfR")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrPowOfR", len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += int(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func (a i64s) IncrPowOfR(other interface{}, incrArr Number) (err error) {
	var b int64
	if b, err = getInt64(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOfR")
	}

	var incr []int64
	if incr, err = getInt64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOfR")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrPowOfR", len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += int64(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func (a i32s) IncrPowOfR(other interface{}, incrArr Number) (err error) {
	var b int32
	if b, err = getInt32(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOfR")
	}

	var incr []int32
	if incr, err = getInt32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOfR")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrPowOfR", len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += int32(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func (a u8s) IncrPowOfR(other interface{}, incrArr Number) (err error) {
	var b byte
	if b, err = getByte(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOfR")
	}

	var incr []byte
	if incr, err = getBytes(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOfR")
	}

	if len(incr) != len(a) {
		return errors.Errorf("lenMismatch", "IncrPowOfR", len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += byte(math.Pow(float64(b), float64(v)))
	}
	return nil
}
