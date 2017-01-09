package tensor

import (
	"math"

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

/* Pow */

func (a f64s) Pow(other Number) error {
	var b []float64

	switch ot := other.(type) {
	case f64s:
		b = []float64(ot)
	case Float64ser:
		b = ot.Float64s()
	default:
		return errors.Errorf(typeMismatch, "Pow", a, other)
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Pow", len(a), len(b))
	}

	vecf64.Pow([]float64(a), b)
	return nil

}

func (a f32s) Pow(other Number) error {
	var b []float32

	switch ot := other.(type) {
	case f32s:
		b = []float32(ot)
	case Float32ser:
		b = ot.Float32s()
	default:
		return errors.Errorf(typeMismatch, "Pow", a, other)
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Pow", len(a), len(b))
	}

	vecf32.Pow([]float32(a), b)
	return nil

}

func (a ints) Pow(other Number) error {
	var b []int

	switch ot := other.(type) {
	case ints:
		b = []int(ot)
	case Intser:
		b = ot.Ints()
	default:
		return errors.Errorf(typeMismatch, "Pow", a, other)
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
	var b []int64

	switch ot := other.(type) {
	case i64s:
		b = []int64(ot)
	case Int64ser:
		b = ot.Int64s()
	default:
		return errors.Errorf(typeMismatch, "Pow", a, other)
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
	var b []int32

	switch ot := other.(type) {
	case i32s:
		b = []int32(ot)
	case Int32ser:
		b = ot.Int32s()
	default:
		return errors.Errorf(typeMismatch, "Pow", a, other)
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
	var b []byte

	switch ot := other.(type) {
	case u8s:
		b = []byte(ot)
	case Byteser:
		b = ot.Bytes()
	default:
		return errors.Errorf(typeMismatch, "Pow", a, other)
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

func (a f64s) Trans(other interface{}) error {
	var b float64
	var ok bool

	if b, ok = other.(float64); !ok {
		return errors.Errorf("Expected float64. Got %T instead", other)
	}

	vecf64.Trans(b, []float64(a))
	return nil

}

func (a f32s) Trans(other interface{}) error {
	var b float32
	var ok bool

	if b, ok = other.(float32); !ok {
		return errors.Errorf("Expected float32. Got %T instead", other)
	}

	vecf32.Trans(b, []float32(a))
	return nil

}

func (a ints) Trans(other interface{}) error {
	var b int
	var ok bool

	if b, ok = other.(int); !ok {
		return errors.Errorf("Expected int. Got %T instead", other)
	}

	for i, v := range a {
		a[i] = b + v

	}
	return nil
}

func (a i64s) Trans(other interface{}) error {
	var b int64
	var ok bool

	if b, ok = other.(int64); !ok {
		return errors.Errorf("Expected int64. Got %T instead", other)
	}

	for i, v := range a {
		a[i] = b + v

	}
	return nil
}

func (a i32s) Trans(other interface{}) error {
	var b int32
	var ok bool

	if b, ok = other.(int32); !ok {
		return errors.Errorf("Expected int32. Got %T instead", other)
	}

	for i, v := range a {
		a[i] = b + v

	}
	return nil
}

func (a u8s) Trans(other interface{}) error {
	var b byte
	var ok bool

	if b, ok = other.(byte); !ok {
		return errors.Errorf("Expected byte. Got %T instead", other)
	}

	for i, v := range a {
		a[i] = b + v

	}
	return nil
}

/* TransR */

func (a f64s) TransR(other interface{}) error {
	var b float64
	var ok bool

	if b, ok = other.(float64); !ok {
		return errors.Errorf("Expected float64. Got %T instead", other)
	}

	vecf64.TransR(b, []float64(a))
	return nil

}

func (a f32s) TransR(other interface{}) error {
	var b float32
	var ok bool

	if b, ok = other.(float32); !ok {
		return errors.Errorf("Expected float32. Got %T instead", other)
	}

	vecf32.TransR(b, []float32(a))
	return nil

}

func (a ints) TransR(other interface{}) error {
	var b int
	var ok bool

	if b, ok = other.(int); !ok {
		return errors.Errorf("Expected int. Got %T instead", other)
	}

	for i, v := range a {
		a[i] = b - v

	}
	return nil
}

func (a i64s) TransR(other interface{}) error {
	var b int64
	var ok bool

	if b, ok = other.(int64); !ok {
		return errors.Errorf("Expected int64. Got %T instead", other)
	}

	for i, v := range a {
		a[i] = b - v

	}
	return nil
}

func (a i32s) TransR(other interface{}) error {
	var b int32
	var ok bool

	if b, ok = other.(int32); !ok {
		return errors.Errorf("Expected int32. Got %T instead", other)
	}

	for i, v := range a {
		a[i] = b - v

	}
	return nil
}

func (a u8s) TransR(other interface{}) error {
	var b byte
	var ok bool

	if b, ok = other.(byte); !ok {
		return errors.Errorf("Expected byte. Got %T instead", other)
	}

	for i, v := range a {
		a[i] = b - v

	}
	return nil
}

/* Scale */

func (a f64s) Scale(other interface{}) error {
	var b float64
	var ok bool

	if b, ok = other.(float64); !ok {
		return errors.Errorf("Expected float64. Got %T instead", other)
	}

	vecf64.Scale(b, []float64(a))
	return nil

}

func (a f32s) Scale(other interface{}) error {
	var b float32
	var ok bool

	if b, ok = other.(float32); !ok {
		return errors.Errorf("Expected float32. Got %T instead", other)
	}

	vecf32.Scale(b, []float32(a))
	return nil

}

func (a ints) Scale(other interface{}) error {
	var b int
	var ok bool

	if b, ok = other.(int); !ok {
		return errors.Errorf("Expected int. Got %T instead", other)
	}

	for i, v := range a {
		a[i] = b * v

	}
	return nil
}

func (a i64s) Scale(other interface{}) error {
	var b int64
	var ok bool

	if b, ok = other.(int64); !ok {
		return errors.Errorf("Expected int64. Got %T instead", other)
	}

	for i, v := range a {
		a[i] = b * v

	}
	return nil
}

func (a i32s) Scale(other interface{}) error {
	var b int32
	var ok bool

	if b, ok = other.(int32); !ok {
		return errors.Errorf("Expected int32. Got %T instead", other)
	}

	for i, v := range a {
		a[i] = b * v

	}
	return nil
}

func (a u8s) Scale(other interface{}) error {
	var b byte
	var ok bool

	if b, ok = other.(byte); !ok {
		return errors.Errorf("Expected byte. Got %T instead", other)
	}

	for i, v := range a {
		a[i] = b * v

	}
	return nil
}

/* DivR */

func (a f64s) DivR(other interface{}) error {
	var b float64
	var ok bool

	if b, ok = other.(float64); !ok {
		return errors.Errorf("Expected float64. Got %T instead", other)
	}

	vecf64.DivR(b, []float64(a))
	return nil

}

func (a f32s) DivR(other interface{}) error {
	var b float32
	var ok bool

	if b, ok = other.(float32); !ok {
		return errors.Errorf("Expected float32. Got %T instead", other)
	}

	vecf32.DivR(b, []float32(a))
	return nil

}

func (a ints) DivR(other interface{}) error {
	var b int
	var ok bool

	if b, ok = other.(int); !ok {
		return errors.Errorf("Expected int. Got %T instead", other)
	}

	for i, v := range a {
		a[i] = b / v

	}
	return nil
}

func (a i64s) DivR(other interface{}) error {
	var b int64
	var ok bool

	if b, ok = other.(int64); !ok {
		return errors.Errorf("Expected int64. Got %T instead", other)
	}

	for i, v := range a {
		a[i] = b / v

	}
	return nil
}

func (a i32s) DivR(other interface{}) error {
	var b int32
	var ok bool

	if b, ok = other.(int32); !ok {
		return errors.Errorf("Expected int32. Got %T instead", other)
	}

	for i, v := range a {
		a[i] = b / v

	}
	return nil
}

func (a u8s) DivR(other interface{}) error {
	var b byte
	var ok bool

	if b, ok = other.(byte); !ok {
		return errors.Errorf("Expected byte. Got %T instead", other)
	}

	for i, v := range a {
		a[i] = b / v

	}
	return nil
}

/* PowOf */

func (a f64s) PowOf(other interface{}) error {
	var b float64
	var ok bool

	if b, ok = other.(float64); !ok {
		return errors.Errorf("Expected float64. Got %T instead", other)
	}

	vecf64.PowOf(b, []float64(a))
	return nil

}

func (a f32s) PowOf(other interface{}) error {
	var b float32
	var ok bool

	if b, ok = other.(float32); !ok {
		return errors.Errorf("Expected float32. Got %T instead", other)
	}

	vecf32.PowOf(b, []float32(a))
	return nil

}

func (a ints) PowOf(other interface{}) error {
	var b int
	var ok bool

	if b, ok = other.(int); !ok {
		return errors.Errorf("Expected int. Got %T instead", other)
	}

	for i, v := range a {
		a[i] = int(math.Pow(float64(v), float64(b)))

	}
	return nil
}

func (a i64s) PowOf(other interface{}) error {
	var b int64
	var ok bool

	if b, ok = other.(int64); !ok {
		return errors.Errorf("Expected int64. Got %T instead", other)
	}

	for i, v := range a {
		a[i] = int64(math.Pow(float64(v), float64(b)))

	}
	return nil
}

func (a i32s) PowOf(other interface{}) error {
	var b int32
	var ok bool

	if b, ok = other.(int32); !ok {
		return errors.Errorf("Expected int32. Got %T instead", other)
	}

	for i, v := range a {
		a[i] = int32(math.Pow(float64(v), float64(b)))

	}
	return nil
}

func (a u8s) PowOf(other interface{}) error {
	var b byte
	var ok bool

	if b, ok = other.(byte); !ok {
		return errors.Errorf("Expected byte. Got %T instead", other)
	}

	for i, v := range a {
		a[i] = byte(math.Pow(float64(v), float64(b)))

	}
	return nil
}

/* PowOfR */

func (a f64s) PowOfR(other interface{}) error {
	var b float64
	var ok bool

	if b, ok = other.(float64); !ok {
		return errors.Errorf("Expected float64. Got %T instead", other)
	}

	vecf64.PowOfR(b, []float64(a))
	return nil

}

func (a f32s) PowOfR(other interface{}) error {
	var b float32
	var ok bool

	if b, ok = other.(float32); !ok {
		return errors.Errorf("Expected float32. Got %T instead", other)
	}

	vecf32.PowOfR(b, []float32(a))
	return nil

}

func (a ints) PowOfR(other interface{}) error {
	var b int
	var ok bool

	if b, ok = other.(int); !ok {
		return errors.Errorf("Expected int. Got %T instead", other)
	}

	for i, v := range a {
		a[i] = int(math.Pow(float64(v), float64(b)))

	}
	return nil
}

func (a i64s) PowOfR(other interface{}) error {
	var b int64
	var ok bool

	if b, ok = other.(int64); !ok {
		return errors.Errorf("Expected int64. Got %T instead", other)
	}

	for i, v := range a {
		a[i] = int64(math.Pow(float64(v), float64(b)))

	}
	return nil
}

func (a i32s) PowOfR(other interface{}) error {
	var b int32
	var ok bool

	if b, ok = other.(int32); !ok {
		return errors.Errorf("Expected int32. Got %T instead", other)
	}

	for i, v := range a {
		a[i] = int32(math.Pow(float64(v), float64(b)))

	}
	return nil
}

func (a u8s) PowOfR(other interface{}) error {
	var b byte
	var ok bool

	if b, ok = other.(byte); !ok {
		return errors.Errorf("Expected byte. Got %T instead", other)
	}

	for i, v := range a {
		a[i] = byte(math.Pow(float64(v), float64(b)))

	}
	return nil
}
