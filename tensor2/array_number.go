package tensor

import (
	"github.com/chewxy/vecf32"
	"github.com/chewxy/vecf64"
	"github.com/pkg/errors"
)

/* Add */

func (a f64s) Add(other Number) error {
	var b []float64

	switch ot := other.(type) {
	case f64s:
		b = []float64(ot)
	case Float64ser:
		b = ot.Float64s()
	default:
		return errors.Errorf(typeMismatch, "Add", a, other)
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Add", len(a), len(b))
	}

	vecf64.Add([]float64(a), b)
	return nil

}

func (a f32s) Add(other Number) error {
	var b []float32

	switch ot := other.(type) {
	case f32s:
		b = []float32(ot)
	case Float32ser:
		b = ot.Float32s()
	default:
		return errors.Errorf(typeMismatch, "Add", a, other)
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Add", len(a), len(b))
	}

	vecf32.Add([]float32(a), b)
	return nil

}

func (a ints) Add(other Number) error {
	var b []int

	switch ot := other.(type) {
	case ints:
		b = []int(ot)
	case Intser:
		b = ot.Ints()
	default:
		return errors.Errorf(typeMismatch, "Add", a, other)
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
	var b []int64

	switch ot := other.(type) {
	case i64s:
		b = []int64(ot)
	case Int64ser:
		b = ot.Int64s()
	default:
		return errors.Errorf(typeMismatch, "Add", a, other)
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
	var b []int32

	switch ot := other.(type) {
	case i32s:
		b = []int32(ot)
	case Int32ser:
		b = ot.Int32s()
	default:
		return errors.Errorf(typeMismatch, "Add", a, other)
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
	var b []byte

	switch ot := other.(type) {
	case u8s:
		b = []byte(ot)
	case Byteser:
		b = ot.Bytes()
	default:
		return errors.Errorf(typeMismatch, "Add", a, other)
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
	var b []float64

	switch ot := other.(type) {
	case f64s:
		b = []float64(ot)
	case Float64ser:
		b = ot.Float64s()
	default:
		return errors.Errorf(typeMismatch, "Sub", a, other)
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Sub", len(a), len(b))
	}

	vecf64.Sub([]float64(a), b)
	return nil

}

func (a f32s) Sub(other Number) error {
	var b []float32

	switch ot := other.(type) {
	case f32s:
		b = []float32(ot)
	case Float32ser:
		b = ot.Float32s()
	default:
		return errors.Errorf(typeMismatch, "Sub", a, other)
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Sub", len(a), len(b))
	}

	vecf32.Sub([]float32(a), b)
	return nil

}

func (a ints) Sub(other Number) error {
	var b []int

	switch ot := other.(type) {
	case ints:
		b = []int(ot)
	case Intser:
		b = ot.Ints()
	default:
		return errors.Errorf(typeMismatch, "Sub", a, other)
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
	var b []int64

	switch ot := other.(type) {
	case i64s:
		b = []int64(ot)
	case Int64ser:
		b = ot.Int64s()
	default:
		return errors.Errorf(typeMismatch, "Sub", a, other)
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
	var b []int32

	switch ot := other.(type) {
	case i32s:
		b = []int32(ot)
	case Int32ser:
		b = ot.Int32s()
	default:
		return errors.Errorf(typeMismatch, "Sub", a, other)
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
	var b []byte

	switch ot := other.(type) {
	case u8s:
		b = []byte(ot)
	case Byteser:
		b = ot.Bytes()
	default:
		return errors.Errorf(typeMismatch, "Sub", a, other)
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
	var b []float64

	switch ot := other.(type) {
	case f64s:
		b = []float64(ot)
	case Float64ser:
		b = ot.Float64s()
	default:
		return errors.Errorf(typeMismatch, "Mul", a, other)
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Mul", len(a), len(b))
	}

	vecf64.Mul([]float64(a), b)
	return nil

}

func (a f32s) Mul(other Number) error {
	var b []float32

	switch ot := other.(type) {
	case f32s:
		b = []float32(ot)
	case Float32ser:
		b = ot.Float32s()
	default:
		return errors.Errorf(typeMismatch, "Mul", a, other)
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Mul", len(a), len(b))
	}

	vecf32.Mul([]float32(a), b)
	return nil

}

func (a ints) Mul(other Number) error {
	var b []int

	switch ot := other.(type) {
	case ints:
		b = []int(ot)
	case Intser:
		b = ot.Ints()
	default:
		return errors.Errorf(typeMismatch, "Mul", a, other)
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
	var b []int64

	switch ot := other.(type) {
	case i64s:
		b = []int64(ot)
	case Int64ser:
		b = ot.Int64s()
	default:
		return errors.Errorf(typeMismatch, "Mul", a, other)
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
	var b []int32

	switch ot := other.(type) {
	case i32s:
		b = []int32(ot)
	case Int32ser:
		b = ot.Int32s()
	default:
		return errors.Errorf(typeMismatch, "Mul", a, other)
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
	var b []byte

	switch ot := other.(type) {
	case u8s:
		b = []byte(ot)
	case Byteser:
		b = ot.Bytes()
	default:
		return errors.Errorf(typeMismatch, "Mul", a, other)
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
	var b []float64

	switch ot := other.(type) {
	case f64s:
		b = []float64(ot)
	case Float64ser:
		b = ot.Float64s()
	default:
		return errors.Errorf(typeMismatch, "Div", a, other)
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Div", len(a), len(b))
	}

	vecf64.Div([]float64(a), b)
	return nil

}

func (a f32s) Div(other Number) error {
	var b []float32

	switch ot := other.(type) {
	case f32s:
		b = []float32(ot)
	case Float32ser:
		b = ot.Float32s()
	default:
		return errors.Errorf(typeMismatch, "Div", a, other)
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Div", len(a), len(b))
	}

	vecf32.Div([]float32(a), b)
	return nil

}

func (a ints) Div(other Number) error {
	var b []int

	switch ot := other.(type) {
	case ints:
		b = []int(ot)
	case Intser:
		b = ot.Ints()
	default:
		return errors.Errorf(typeMismatch, "Div", a, other)
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Div", len(a), len(b))
	}

	for i, v := range b {
		a[i] /= v
	}
	return nil
}

func (a i64s) Div(other Number) error {
	var b []int64

	switch ot := other.(type) {
	case i64s:
		b = []int64(ot)
	case Int64ser:
		b = ot.Int64s()
	default:
		return errors.Errorf(typeMismatch, "Div", a, other)
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Div", len(a), len(b))
	}

	for i, v := range b {
		a[i] /= v
	}
	return nil
}

func (a i32s) Div(other Number) error {
	var b []int32

	switch ot := other.(type) {
	case i32s:
		b = []int32(ot)
	case Int32ser:
		b = ot.Int32s()
	default:
		return errors.Errorf(typeMismatch, "Div", a, other)
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Div", len(a), len(b))
	}

	for i, v := range b {
		a[i] /= v
	}
	return nil
}

func (a u8s) Div(other Number) error {
	var b []byte

	switch ot := other.(type) {
	case u8s:
		b = []byte(ot)
	case Byteser:
		b = ot.Bytes()
	default:
		return errors.Errorf(typeMismatch, "Div", a, other)
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Div", len(a), len(b))
	}

	for i, v := range b {
		a[i] /= v
	}
	return nil
}
