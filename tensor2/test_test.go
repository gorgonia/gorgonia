package tensor

import (
	"math"

	"github.com/chewxy/vecf32"
	"github.com/chewxy/vecf64"
	"github.com/pkg/errors"
)

type f64sDummy []float64
type f32sDummy []float32
type intsDummy []int
type i64sDummy []int64
type i32sDummy []int32
type u8sDummy []byte
type bsDummy []bool

/* Len */

func (a f64sDummy) Len() int { return len(a) }
func (a f32sDummy) Len() int { return len(a) }
func (a intsDummy) Len() int { return len(a) }
func (a i64sDummy) Len() int { return len(a) }
func (a i32sDummy) Len() int { return len(a) }
func (a u8sDummy) Len() int  { return len(a) }
func (a bsDummy) Len() int   { return len(a) }

/* Cap */

func (a f64sDummy) Cap() int { return cap(a) }
func (a f32sDummy) Cap() int { return cap(a) }
func (a intsDummy) Cap() int { return cap(a) }
func (a i64sDummy) Cap() int { return cap(a) }
func (a i32sDummy) Cap() int { return cap(a) }
func (a u8sDummy) Cap() int  { return cap(a) }
func (a bsDummy) Cap() int   { return cap(a) }

/* Data */

func (a f64sDummy) Data() interface{} { return []float64(a) }
func (a f32sDummy) Data() interface{} { return []float32(a) }
func (a intsDummy) Data() interface{} { return []int(a) }
func (a i64sDummy) Data() interface{} { return []int64(a) }
func (a i32sDummy) Data() interface{} { return []int32(a) }
func (a u8sDummy) Data() interface{}  { return []byte(a) }
func (a bsDummy) Data() interface{}   { return []bool(a) }

/* Get */

func (a f64sDummy) Get(i int) interface{} { return a[i] }
func (a f32sDummy) Get(i int) interface{} { return a[i] }
func (a intsDummy) Get(i int) interface{} { return a[i] }
func (a i64sDummy) Get(i int) interface{} { return a[i] }
func (a i32sDummy) Get(i int) interface{} { return a[i] }
func (a u8sDummy) Get(i int) interface{}  { return a[i] }
func (a bsDummy) Get(i int) interface{}   { return a[i] }

/* Set */

func (a f64sDummy) Set(i int, v interface{}) error {
	if f, ok := v.(float64); ok {
		a[i] = f
		return nil
	}
	return errors.Errorf("Cannot set %v of %T to []float64", v, v)
}

func (a f32sDummy) Set(i int, v interface{}) error {
	if f, ok := v.(float32); ok {
		a[i] = f
		return nil
	}
	return errors.Errorf("Cannot set %v of %T to []float32", v, v)
}

func (a intsDummy) Set(i int, v interface{}) error {
	if f, ok := v.(int); ok {
		a[i] = f
		return nil
	}
	return errors.Errorf("Cannot set %v of %T to []int", v, v)
}

func (a i64sDummy) Set(i int, v interface{}) error {
	if f, ok := v.(int64); ok {
		a[i] = f
		return nil
	}
	return errors.Errorf("Cannot set %v of %T to []int64", v, v)
}

func (a i32sDummy) Set(i int, v interface{}) error {
	if f, ok := v.(int32); ok {
		a[i] = f
		return nil
	}
	return errors.Errorf("Cannot set %v of %T to []int32", v, v)
}

func (a u8sDummy) Set(i int, v interface{}) error {
	if f, ok := v.(byte); ok {
		a[i] = f
		return nil
	}
	return errors.Errorf("Cannot set %v of %T to []byte", v, v)
}

func (a bsDummy) Set(i int, v interface{}) error {
	if f, ok := v.(bool); ok {
		a[i] = f
		return nil
	}
	return errors.Errorf("Cannot set %v of %T to []bool", v, v)
}

/* Eq */

func (a f64sDummy) Eq(other interface{}) bool {
	if b, ok := other.(f64sDummy); ok {
		if len(a) != len(b) {
			return false
		}

		for i, v := range a {
			if v != b[i] {
				return false
			}
		}
		return true
	}

	if b, ok := other.([]float64); ok {
		if len(a) != len(b) {
			return false
		}
		for i, v := range a {
			if v != b[i] {
				return false
			}
		}
		return true
	}
	return false
}

func (a f32sDummy) Eq(other interface{}) bool {
	if b, ok := other.(f32sDummy); ok {
		if len(a) != len(b) {
			return false
		}

		for i, v := range a {
			if v != b[i] {
				return false
			}
		}
		return true
	}

	if b, ok := other.([]float32); ok {
		if len(a) != len(b) {
			return false
		}
		for i, v := range a {
			if v != b[i] {
				return false
			}
		}
		return true
	}
	return false
}

func (a intsDummy) Eq(other interface{}) bool {
	if b, ok := other.(intsDummy); ok {
		if len(a) != len(b) {
			return false
		}

		for i, v := range a {
			if v != b[i] {
				return false
			}
		}
		return true
	}

	if b, ok := other.([]int); ok {
		if len(a) != len(b) {
			return false
		}
		for i, v := range a {
			if v != b[i] {
				return false
			}
		}
		return true
	}
	return false
}

func (a i64sDummy) Eq(other interface{}) bool {
	if b, ok := other.(i64sDummy); ok {
		if len(a) != len(b) {
			return false
		}

		for i, v := range a {
			if v != b[i] {
				return false
			}
		}
		return true
	}

	if b, ok := other.([]int64); ok {
		if len(a) != len(b) {
			return false
		}
		for i, v := range a {
			if v != b[i] {
				return false
			}
		}
		return true
	}
	return false
}

func (a i32sDummy) Eq(other interface{}) bool {
	if b, ok := other.(i32sDummy); ok {
		if len(a) != len(b) {
			return false
		}

		for i, v := range a {
			if v != b[i] {
				return false
			}
		}
		return true
	}

	if b, ok := other.([]int32); ok {
		if len(a) != len(b) {
			return false
		}
		for i, v := range a {
			if v != b[i] {
				return false
			}
		}
		return true
	}
	return false
}

func (a u8sDummy) Eq(other interface{}) bool {
	if b, ok := other.(u8sDummy); ok {
		if len(a) != len(b) {
			return false
		}

		for i, v := range a {
			if v != b[i] {
				return false
			}
		}
		return true
	}

	if b, ok := other.([]byte); ok {
		if len(a) != len(b) {
			return false
		}
		for i, v := range a {
			if v != b[i] {
				return false
			}
		}
		return true
	}
	return false
}

func (a bsDummy) Eq(other interface{}) bool {
	if b, ok := other.(bsDummy); ok {
		if len(a) != len(b) {
			return false
		}

		for i, v := range a {
			if v != b[i] {
				return false
			}
		}
		return true
	}

	if b, ok := other.([]bool); ok {
		if len(a) != len(b) {
			return false
		}
		for i, v := range a {
			if v != b[i] {
				return false
			}
		}
		return true
	}
	return false
}

/* Zeror */

func (a f64sDummy) Zero() {
	for i := range a {
		a[i] = float64(0)
	}
}

func (a f32sDummy) Zero() {
	for i := range a {
		a[i] = float32(0)
	}
}

func (a intsDummy) Zero() {
	for i := range a {
		a[i] = int(0)
	}
}

func (a i64sDummy) Zero() {
	for i := range a {
		a[i] = int64(0)
	}
}

func (a i32sDummy) Zero() {
	for i := range a {
		a[i] = int32(0)
	}
}

func (a u8sDummy) Zero() {
	for i := range a {
		a[i] = byte(0)
	}
}

func (a bsDummy) Zero() {
	for i := range a {
		a[i] = false
	}
}

/* Oner */

func (a f64sDummy) One() {
	for i := range a {
		a[i] = float64(1)
	}
}

func (a f32sDummy) One() {
	for i := range a {
		a[i] = float32(1)
	}
}

func (a intsDummy) One() {
	for i := range a {
		a[i] = int(1)
	}
}

func (a i64sDummy) One() {
	for i := range a {
		a[i] = int64(1)
	}
}

func (a i32sDummy) One() {
	for i := range a {
		a[i] = int32(1)
	}
}

func (a u8sDummy) One() {
	for i := range a {
		a[i] = byte(1)
	}
}

func (a bsDummy) One() {
	for i := range a {
		a[i] = true
	}
}

/* CopierFrom */

func (a f64sDummy) CopyFrom(other interface{}) (int, error) {
	switch b := other.(type) {
	case f64sDummy:
		return copy(a, b), nil
	case []float64:
		return copy(a, b), nil
	case Float64ser:
		return copy(a, b.Float64s()), nil
	}

	return 0, errors.Errorf("Cannot copy from %T", other)
}

func (a f32sDummy) CopyFrom(other interface{}) (int, error) {
	switch b := other.(type) {
	case f32sDummy:
		return copy(a, b), nil
	case []float32:
		return copy(a, b), nil
	case Float32ser:
		return copy(a, b.Float32s()), nil
	}

	return 0, errors.Errorf("Cannot copy from %T", other)
}

func (a intsDummy) CopyFrom(other interface{}) (int, error) {
	switch b := other.(type) {
	case intsDummy:
		return copy(a, b), nil
	case []int:
		return copy(a, b), nil
	case Intser:
		return copy(a, b.Ints()), nil
	}

	return 0, errors.Errorf("Cannot copy from %T", other)
}

func (a i64sDummy) CopyFrom(other interface{}) (int, error) {
	switch b := other.(type) {
	case i64sDummy:
		return copy(a, b), nil
	case []int64:
		return copy(a, b), nil
	case Int64ser:
		return copy(a, b.Int64s()), nil
	}

	return 0, errors.Errorf("Cannot copy from %T", other)
}

func (a i32sDummy) CopyFrom(other interface{}) (int, error) {
	switch b := other.(type) {
	case i32sDummy:
		return copy(a, b), nil
	case []int32:
		return copy(a, b), nil
	case Int32ser:
		return copy(a, b.Int32s()), nil
	}

	return 0, errors.Errorf("Cannot copy from %T", other)
}

func (a u8sDummy) CopyFrom(other interface{}) (int, error) {
	switch b := other.(type) {
	case u8sDummy:
		return copy(a, b), nil
	case []byte:
		return copy(a, b), nil
	case Byteser:
		return copy(a, b.Bytes()), nil
	}

	return 0, errors.Errorf("Cannot copy from %T", other)
}

func (a bsDummy) CopyFrom(other interface{}) (int, error) {
	switch b := other.(type) {
	case bsDummy:
		return copy(a, b), nil
	case []bool:
		return copy(a, b), nil
	case Boolser:
		return copy(a, b.Bools()), nil
	}

	return 0, errors.Errorf("Cannot copy from %T", other)
}

/* COMPAT */

func (a f64sDummy) Float64s() []float64 { return []float64(a) }
func (a f32sDummy) Float32s() []float32 { return []float32(a) }
func (a intsDummy) Ints() []int         { return []int(a) }
func (a i64sDummy) Int64s() []int64     { return []int64(a) }
func (a i32sDummy) Int32s() []int32     { return []int32(a) }
func (a u8sDummy) Bytes() []byte        { return []byte(a) }
func (a bsDummy) Bools() []bool         { return []bool(a) }

/* Add */

func (a f64sDummy) Add(other Number) error {
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

func (a f32sDummy) Add(other Number) error {
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

func (a intsDummy) Add(other Number) error {
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

func (a i64sDummy) Add(other Number) error {
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

func (a i32sDummy) Add(other Number) error {
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

func (a u8sDummy) Add(other Number) error {
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

func (a f64sDummy) Sub(other Number) error {
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

func (a f32sDummy) Sub(other Number) error {
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

func (a intsDummy) Sub(other Number) error {
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

func (a i64sDummy) Sub(other Number) error {
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

func (a i32sDummy) Sub(other Number) error {
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

func (a u8sDummy) Sub(other Number) error {
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

func (a f64sDummy) Mul(other Number) error {
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

func (a f32sDummy) Mul(other Number) error {
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

func (a intsDummy) Mul(other Number) error {
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

func (a i64sDummy) Mul(other Number) error {
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

func (a i32sDummy) Mul(other Number) error {
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

func (a u8sDummy) Mul(other Number) error {
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

func (a f64sDummy) Div(other Number) error {
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

func (a f32sDummy) Div(other Number) error {
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

func (a intsDummy) Div(other Number) error {
	b, err := getInts(other)
	if err != nil {
		return errors.Wrapf(err, opFail, "Div")
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Div", len(a), len(b))
	}

	for i, v := range b {
		a[i] /= v
	}

	return nil
}

func (a i64sDummy) Div(other Number) error {
	b, err := getInt64s(other)
	if err != nil {
		return errors.Wrapf(err, opFail, "Div")
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Div", len(a), len(b))
	}

	for i, v := range b {
		a[i] /= v
	}

	return nil
}

func (a i32sDummy) Div(other Number) error {
	b, err := getInt32s(other)
	if err != nil {
		return errors.Wrapf(err, opFail, "Div")
	}

	if len(a) != len(b) {
		return errors.Errorf("lenMismatch", "Div", len(a), len(b))
	}

	for i, v := range b {
		a[i] /= v
	}

	return nil
}

func (a u8sDummy) Div(other Number) error {
	b, err := getBytes(other)
	if err != nil {
		return errors.Wrapf(err, opFail, "Div")
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

func (a f64sDummy) Pow(other Number) error {
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

func (a f32sDummy) Pow(other Number) error {
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

func (a intsDummy) Pow(other Number) error {
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

func (a i64sDummy) Pow(other Number) error {
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

func (a i32sDummy) Pow(other Number) error {
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

func (a u8sDummy) Pow(other Number) error {
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

func (a f64sDummy) Trans(other interface{}) (err error) {
	var b float64
	if b, err = getFloat64(other); err != nil {
		return errors.Wrapf(err, opFail, "Trans")
	}

	vecf64.Trans([]float64(a), b)
	return nil
}

func (a f32sDummy) Trans(other interface{}) (err error) {
	var b float32
	if b, err = getFloat32(other); err != nil {
		return errors.Wrapf(err, opFail, "Trans")
	}

	vecf32.Trans([]float32(a), b)
	return nil
}

func (a intsDummy) Trans(other interface{}) (err error) {
	var b int
	if b, err = getInt(other); err != nil {
		return errors.Wrapf(err, opFail, "Trans")
	}

	for i, v := range a {
		a[i] = v + b
	}
	return nil
}

func (a i64sDummy) Trans(other interface{}) (err error) {
	var b int64
	if b, err = getInt64(other); err != nil {
		return errors.Wrapf(err, opFail, "Trans")
	}

	for i, v := range a {
		a[i] = v + b
	}
	return nil
}

func (a i32sDummy) Trans(other interface{}) (err error) {
	var b int32
	if b, err = getInt32(other); err != nil {
		return errors.Wrapf(err, opFail, "Trans")
	}

	for i, v := range a {
		a[i] = v + b
	}
	return nil
}

func (a u8sDummy) Trans(other interface{}) (err error) {
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

func (a f64sDummy) TransInv(other interface{}) (err error) {
	var b float64
	if b, err = getFloat64(other); err != nil {
		return errors.Wrapf(err, opFail, "TransInv")
	}

	vecf64.TransInv([]float64(a), b)
	return nil
}

func (a f32sDummy) TransInv(other interface{}) (err error) {
	var b float32
	if b, err = getFloat32(other); err != nil {
		return errors.Wrapf(err, opFail, "TransInv")
	}

	vecf32.TransInv([]float32(a), b)
	return nil
}

func (a intsDummy) TransInv(other interface{}) (err error) {
	var b int
	if b, err = getInt(other); err != nil {
		return errors.Wrapf(err, opFail, "TransInv")
	}

	for i, v := range a {
		a[i] = v - b
	}
	return nil
}

func (a i64sDummy) TransInv(other interface{}) (err error) {
	var b int64
	if b, err = getInt64(other); err != nil {
		return errors.Wrapf(err, opFail, "TransInv")
	}

	for i, v := range a {
		a[i] = v - b
	}
	return nil
}

func (a i32sDummy) TransInv(other interface{}) (err error) {
	var b int32
	if b, err = getInt32(other); err != nil {
		return errors.Wrapf(err, opFail, "TransInv")
	}

	for i, v := range a {
		a[i] = v - b
	}
	return nil
}

func (a u8sDummy) TransInv(other interface{}) (err error) {
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

func (a f64sDummy) TransInvR(other interface{}) (err error) {
	var b float64
	if b, err = getFloat64(other); err != nil {
		return errors.Wrapf(err, opFail, "TransInvR")
	}

	vecf64.TransInvR([]float64(a), b)
	return nil
}

func (a f32sDummy) TransInvR(other interface{}) (err error) {
	var b float32
	if b, err = getFloat32(other); err != nil {
		return errors.Wrapf(err, opFail, "TransInvR")
	}

	vecf32.TransInvR([]float32(a), b)
	return nil
}

func (a intsDummy) TransInvR(other interface{}) (err error) {
	var b int
	if b, err = getInt(other); err != nil {
		return errors.Wrapf(err, opFail, "TransInvR")
	}

	for i, v := range a {
		a[i] = b - v
	}
	return nil
}

func (a i64sDummy) TransInvR(other interface{}) (err error) {
	var b int64
	if b, err = getInt64(other); err != nil {
		return errors.Wrapf(err, opFail, "TransInvR")
	}

	for i, v := range a {
		a[i] = b - v
	}
	return nil
}

func (a i32sDummy) TransInvR(other interface{}) (err error) {
	var b int32
	if b, err = getInt32(other); err != nil {
		return errors.Wrapf(err, opFail, "TransInvR")
	}

	for i, v := range a {
		a[i] = b - v
	}
	return nil
}

func (a u8sDummy) TransInvR(other interface{}) (err error) {
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

func (a f64sDummy) Scale(other interface{}) (err error) {
	var b float64
	if b, err = getFloat64(other); err != nil {
		return errors.Wrapf(err, opFail, "Scale")
	}

	vecf64.Scale([]float64(a), b)
	return nil
}

func (a f32sDummy) Scale(other interface{}) (err error) {
	var b float32
	if b, err = getFloat32(other); err != nil {
		return errors.Wrapf(err, opFail, "Scale")
	}

	vecf32.Scale([]float32(a), b)
	return nil
}

func (a intsDummy) Scale(other interface{}) (err error) {
	var b int
	if b, err = getInt(other); err != nil {
		return errors.Wrapf(err, opFail, "Scale")
	}

	for i, v := range a {
		a[i] = v * b
	}
	return nil
}

func (a i64sDummy) Scale(other interface{}) (err error) {
	var b int64
	if b, err = getInt64(other); err != nil {
		return errors.Wrapf(err, opFail, "Scale")
	}

	for i, v := range a {
		a[i] = v * b
	}
	return nil
}

func (a i32sDummy) Scale(other interface{}) (err error) {
	var b int32
	if b, err = getInt32(other); err != nil {
		return errors.Wrapf(err, opFail, "Scale")
	}

	for i, v := range a {
		a[i] = v * b
	}
	return nil
}

func (a u8sDummy) Scale(other interface{}) (err error) {
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

func (a f64sDummy) ScaleInv(other interface{}) (err error) {
	var b float64
	if b, err = getFloat64(other); err != nil {
		return errors.Wrapf(err, opFail, "ScaleInv")
	}

	vecf64.ScaleInv([]float64(a), b)
	return nil
}

func (a f32sDummy) ScaleInv(other interface{}) (err error) {
	var b float32
	if b, err = getFloat32(other); err != nil {
		return errors.Wrapf(err, opFail, "ScaleInv")
	}

	vecf32.ScaleInv([]float32(a), b)
	return nil
}

func (a intsDummy) ScaleInv(other interface{}) (err error) {
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

func (a i64sDummy) ScaleInv(other interface{}) (err error) {
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

func (a i32sDummy) ScaleInv(other interface{}) (err error) {
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

func (a u8sDummy) ScaleInv(other interface{}) (err error) {
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

func (a f64sDummy) ScaleInvR(other interface{}) (err error) {
	var b float64
	if b, err = getFloat64(other); err != nil {
		return errors.Wrapf(err, opFail, "ScaleInvR")
	}

	vecf64.ScaleInvR([]float64(a), b)
	return nil
}

func (a f32sDummy) ScaleInvR(other interface{}) (err error) {
	var b float32
	if b, err = getFloat32(other); err != nil {
		return errors.Wrapf(err, opFail, "ScaleInvR")
	}

	vecf32.ScaleInvR([]float32(a), b)
	return nil
}

func (a intsDummy) ScaleInvR(other interface{}) (err error) {
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

func (a i64sDummy) ScaleInvR(other interface{}) (err error) {
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

func (a i32sDummy) ScaleInvR(other interface{}) (err error) {
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

func (a u8sDummy) ScaleInvR(other interface{}) (err error) {
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

func (a f64sDummy) PowOf(other interface{}) (err error) {
	var b float64
	if b, err = getFloat64(other); err != nil {
		return errors.Wrapf(err, opFail, "PowOf")
	}

	vecf64.PowOf([]float64(a), b)
	return nil
}

func (a f32sDummy) PowOf(other interface{}) (err error) {
	var b float32
	if b, err = getFloat32(other); err != nil {
		return errors.Wrapf(err, opFail, "PowOf")
	}

	vecf32.PowOf([]float32(a), b)
	return nil
}

func (a intsDummy) PowOf(other interface{}) (err error) {
	var b int
	if b, err = getInt(other); err != nil {
		return errors.Wrapf(err, opFail, "PowOf")
	}

	for i, v := range a {
		a[i] = int(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func (a i64sDummy) PowOf(other interface{}) (err error) {
	var b int64
	if b, err = getInt64(other); err != nil {
		return errors.Wrapf(err, opFail, "PowOf")
	}

	for i, v := range a {
		a[i] = int64(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func (a i32sDummy) PowOf(other interface{}) (err error) {
	var b int32
	if b, err = getInt32(other); err != nil {
		return errors.Wrapf(err, opFail, "PowOf")
	}

	for i, v := range a {
		a[i] = int32(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func (a u8sDummy) PowOf(other interface{}) (err error) {
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

func (a f64sDummy) PowOfR(other interface{}) (err error) {
	var b float64
	if b, err = getFloat64(other); err != nil {
		return errors.Wrapf(err, opFail, "PowOfR")
	}

	vecf64.PowOfR([]float64(a), b)
	return nil
}

func (a f32sDummy) PowOfR(other interface{}) (err error) {
	var b float32
	if b, err = getFloat32(other); err != nil {
		return errors.Wrapf(err, opFail, "PowOfR")
	}

	vecf32.PowOfR([]float32(a), b)
	return nil
}

func (a intsDummy) PowOfR(other interface{}) (err error) {
	var b int
	if b, err = getInt(other); err != nil {
		return errors.Wrapf(err, opFail, "PowOfR")
	}

	for i, v := range a {
		a[i] = int(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func (a i64sDummy) PowOfR(other interface{}) (err error) {
	var b int64
	if b, err = getInt64(other); err != nil {
		return errors.Wrapf(err, opFail, "PowOfR")
	}

	for i, v := range a {
		a[i] = int64(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func (a i32sDummy) PowOfR(other interface{}) (err error) {
	var b int32
	if b, err = getInt32(other); err != nil {
		return errors.Wrapf(err, opFail, "PowOfR")
	}

	for i, v := range a {
		a[i] = int32(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func (a u8sDummy) PowOfR(other interface{}) (err error) {
	var b byte
	if b, err = getByte(other); err != nil {
		return errors.Wrapf(err, opFail, "PowOfR")
	}

	for i, v := range a {
		a[i] = byte(math.Pow(float64(b), float64(v)))
	}
	return nil
}

/* ElEq */

func (a f64sDummy) ElEq(other ElEq, same bool) (Array, error) {
	var b f64sDummy

	switch ot := other.(type) {
	case f64sDummy:
		b = ot
	case Float64ser:
		b = f64sDummy(ot.Float64s())
	default:
		return nil, errors.Errorf(typeMismatch, "ElEq", a, other)
	}

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(f64sDummy, len(a))
		for i, v := range a {
			if v == b[i] {
				retVal[i] = float64(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v == b[i]
	}
	return retVal, nil
}

func (a f32sDummy) ElEq(other ElEq, same bool) (Array, error) {
	var b f32sDummy

	switch ot := other.(type) {
	case f32sDummy:
		b = ot
	case Float32ser:
		b = f32sDummy(ot.Float32s())
	default:
		return nil, errors.Errorf(typeMismatch, "ElEq", a, other)
	}

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(f32sDummy, len(a))
		for i, v := range a {
			if v == b[i] {
				retVal[i] = float32(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v == b[i]
	}
	return retVal, nil
}

func (a intsDummy) ElEq(other ElEq, same bool) (Array, error) {
	var b intsDummy

	switch ot := other.(type) {
	case intsDummy:
		b = ot
	case Intser:
		b = intsDummy(ot.Ints())
	default:
		return nil, errors.Errorf(typeMismatch, "ElEq", a, other)
	}

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(intsDummy, len(a))
		for i, v := range a {
			if v == b[i] {
				retVal[i] = int(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v == b[i]
	}
	return retVal, nil
}

func (a i64sDummy) ElEq(other ElEq, same bool) (Array, error) {
	var b i64sDummy

	switch ot := other.(type) {
	case i64sDummy:
		b = ot
	case Int64ser:
		b = i64sDummy(ot.Int64s())
	default:
		return nil, errors.Errorf(typeMismatch, "ElEq", a, other)
	}

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(i64sDummy, len(a))
		for i, v := range a {
			if v == b[i] {
				retVal[i] = int64(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v == b[i]
	}
	return retVal, nil
}

func (a i32sDummy) ElEq(other ElEq, same bool) (Array, error) {
	var b i32sDummy

	switch ot := other.(type) {
	case i32sDummy:
		b = ot
	case Int32ser:
		b = i32sDummy(ot.Int32s())
	default:
		return nil, errors.Errorf(typeMismatch, "ElEq", a, other)
	}

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(i32sDummy, len(a))
		for i, v := range a {
			if v == b[i] {
				retVal[i] = int32(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v == b[i]
	}
	return retVal, nil
}

func (a u8sDummy) ElEq(other ElEq, same bool) (Array, error) {
	var b u8sDummy

	switch ot := other.(type) {
	case u8sDummy:
		b = ot
	case Byteser:
		b = u8sDummy(ot.Bytes())
	default:
		return nil, errors.Errorf(typeMismatch, "ElEq", a, other)
	}

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(u8sDummy, len(a))
		for i, v := range a {
			if v == b[i] {
				retVal[i] = byte(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v == b[i]
	}
	return retVal, nil
}

func (a bsDummy) ElEq(other ElEq, same bool) (Array, error) {
	var b bsDummy

	switch ot := other.(type) {
	case bsDummy:
		b = ot
	case Boolser:
		b = bsDummy(ot.Bools())
	default:
		return nil, errors.Errorf(typeMismatch, "ElEq", a, other)
	}

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(bsDummy, len(a))
		for i, v := range a {
			if v == b[i] {
				retVal[i] = true
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v == b[i]
	}
	return retVal, nil
}

/* Gt */

func (a f64sDummy) Gt(other ElOrd, same bool) (Array, error) {
	var b f64sDummy

	switch ot := other.(type) {
	case f64sDummy:
		b = ot
	case Float64ser:
		b = f64sDummy(ot.Float64s())
	default:
		return nil, errors.Errorf(typeMismatch, "Gt", a, other)
	}

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(f64sDummy, len(a))
		for i, v := range a {
			if v > b[i] {
				retVal[i] = float64(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v > b[i]
	}
	return retVal, nil
}

func (a f32sDummy) Gt(other ElOrd, same bool) (Array, error) {
	var b f32sDummy

	switch ot := other.(type) {
	case f32sDummy:
		b = ot
	case Float32ser:
		b = f32sDummy(ot.Float32s())
	default:
		return nil, errors.Errorf(typeMismatch, "Gt", a, other)
	}

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(f32sDummy, len(a))
		for i, v := range a {
			if v > b[i] {
				retVal[i] = float32(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v > b[i]
	}
	return retVal, nil
}

func (a intsDummy) Gt(other ElOrd, same bool) (Array, error) {
	var b intsDummy

	switch ot := other.(type) {
	case intsDummy:
		b = ot
	case Intser:
		b = intsDummy(ot.Ints())
	default:
		return nil, errors.Errorf(typeMismatch, "Gt", a, other)
	}

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(intsDummy, len(a))
		for i, v := range a {
			if v > b[i] {
				retVal[i] = int(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v > b[i]
	}
	return retVal, nil
}

func (a i64sDummy) Gt(other ElOrd, same bool) (Array, error) {
	var b i64sDummy

	switch ot := other.(type) {
	case i64sDummy:
		b = ot
	case Int64ser:
		b = i64sDummy(ot.Int64s())
	default:
		return nil, errors.Errorf(typeMismatch, "Gt", a, other)
	}

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(i64sDummy, len(a))
		for i, v := range a {
			if v > b[i] {
				retVal[i] = int64(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v > b[i]
	}
	return retVal, nil
}

func (a i32sDummy) Gt(other ElOrd, same bool) (Array, error) {
	var b i32sDummy

	switch ot := other.(type) {
	case i32sDummy:
		b = ot
	case Int32ser:
		b = i32sDummy(ot.Int32s())
	default:
		return nil, errors.Errorf(typeMismatch, "Gt", a, other)
	}

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(i32sDummy, len(a))
		for i, v := range a {
			if v > b[i] {
				retVal[i] = int32(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v > b[i]
	}
	return retVal, nil
}

func (a u8sDummy) Gt(other ElOrd, same bool) (Array, error) {
	var b u8sDummy

	switch ot := other.(type) {
	case u8sDummy:
		b = ot
	case Byteser:
		b = u8sDummy(ot.Bytes())
	default:
		return nil, errors.Errorf(typeMismatch, "Gt", a, other)
	}

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(u8sDummy, len(a))
		for i, v := range a {
			if v > b[i] {
				retVal[i] = byte(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v > b[i]
	}
	return retVal, nil
}

/* Gte */

func (a f64sDummy) Gte(other ElOrd, same bool) (Array, error) {
	var b f64sDummy

	switch ot := other.(type) {
	case f64sDummy:
		b = ot
	case Float64ser:
		b = f64sDummy(ot.Float64s())
	default:
		return nil, errors.Errorf(typeMismatch, "Gte", a, other)
	}

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(f64sDummy, len(a))
		for i, v := range a {
			if v >= b[i] {
				retVal[i] = float64(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v >= b[i]
	}
	return retVal, nil
}

func (a f32sDummy) Gte(other ElOrd, same bool) (Array, error) {
	var b f32sDummy

	switch ot := other.(type) {
	case f32sDummy:
		b = ot
	case Float32ser:
		b = f32sDummy(ot.Float32s())
	default:
		return nil, errors.Errorf(typeMismatch, "Gte", a, other)
	}

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(f32sDummy, len(a))
		for i, v := range a {
			if v >= b[i] {
				retVal[i] = float32(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v >= b[i]
	}
	return retVal, nil
}

func (a intsDummy) Gte(other ElOrd, same bool) (Array, error) {
	var b intsDummy

	switch ot := other.(type) {
	case intsDummy:
		b = ot
	case Intser:
		b = intsDummy(ot.Ints())
	default:
		return nil, errors.Errorf(typeMismatch, "Gte", a, other)
	}

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(intsDummy, len(a))
		for i, v := range a {
			if v >= b[i] {
				retVal[i] = int(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v >= b[i]
	}
	return retVal, nil
}

func (a i64sDummy) Gte(other ElOrd, same bool) (Array, error) {
	var b i64sDummy

	switch ot := other.(type) {
	case i64sDummy:
		b = ot
	case Int64ser:
		b = i64sDummy(ot.Int64s())
	default:
		return nil, errors.Errorf(typeMismatch, "Gte", a, other)
	}

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(i64sDummy, len(a))
		for i, v := range a {
			if v >= b[i] {
				retVal[i] = int64(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v >= b[i]
	}
	return retVal, nil
}

func (a i32sDummy) Gte(other ElOrd, same bool) (Array, error) {
	var b i32sDummy

	switch ot := other.(type) {
	case i32sDummy:
		b = ot
	case Int32ser:
		b = i32sDummy(ot.Int32s())
	default:
		return nil, errors.Errorf(typeMismatch, "Gte", a, other)
	}

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(i32sDummy, len(a))
		for i, v := range a {
			if v >= b[i] {
				retVal[i] = int32(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v >= b[i]
	}
	return retVal, nil
}

func (a u8sDummy) Gte(other ElOrd, same bool) (Array, error) {
	var b u8sDummy

	switch ot := other.(type) {
	case u8sDummy:
		b = ot
	case Byteser:
		b = u8sDummy(ot.Bytes())
	default:
		return nil, errors.Errorf(typeMismatch, "Gte", a, other)
	}

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(u8sDummy, len(a))
		for i, v := range a {
			if v >= b[i] {
				retVal[i] = byte(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v >= b[i]
	}
	return retVal, nil
}

/* Lt */

func (a f64sDummy) Lt(other ElOrd, same bool) (Array, error) {
	var b f64sDummy

	switch ot := other.(type) {
	case f64sDummy:
		b = ot
	case Float64ser:
		b = f64sDummy(ot.Float64s())
	default:
		return nil, errors.Errorf(typeMismatch, "Lt", a, other)
	}

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(f64sDummy, len(a))
		for i, v := range a {
			if v < b[i] {
				retVal[i] = float64(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v < b[i]
	}
	return retVal, nil
}

func (a f32sDummy) Lt(other ElOrd, same bool) (Array, error) {
	var b f32sDummy

	switch ot := other.(type) {
	case f32sDummy:
		b = ot
	case Float32ser:
		b = f32sDummy(ot.Float32s())
	default:
		return nil, errors.Errorf(typeMismatch, "Lt", a, other)
	}

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(f32sDummy, len(a))
		for i, v := range a {
			if v < b[i] {
				retVal[i] = float32(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v < b[i]
	}
	return retVal, nil
}

func (a intsDummy) Lt(other ElOrd, same bool) (Array, error) {
	var b intsDummy

	switch ot := other.(type) {
	case intsDummy:
		b = ot
	case Intser:
		b = intsDummy(ot.Ints())
	default:
		return nil, errors.Errorf(typeMismatch, "Lt", a, other)
	}

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(intsDummy, len(a))
		for i, v := range a {
			if v < b[i] {
				retVal[i] = int(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v < b[i]
	}
	return retVal, nil
}

func (a i64sDummy) Lt(other ElOrd, same bool) (Array, error) {
	var b i64sDummy

	switch ot := other.(type) {
	case i64sDummy:
		b = ot
	case Int64ser:
		b = i64sDummy(ot.Int64s())
	default:
		return nil, errors.Errorf(typeMismatch, "Lt", a, other)
	}

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(i64sDummy, len(a))
		for i, v := range a {
			if v < b[i] {
				retVal[i] = int64(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v < b[i]
	}
	return retVal, nil
}

func (a i32sDummy) Lt(other ElOrd, same bool) (Array, error) {
	var b i32sDummy

	switch ot := other.(type) {
	case i32sDummy:
		b = ot
	case Int32ser:
		b = i32sDummy(ot.Int32s())
	default:
		return nil, errors.Errorf(typeMismatch, "Lt", a, other)
	}

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(i32sDummy, len(a))
		for i, v := range a {
			if v < b[i] {
				retVal[i] = int32(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v < b[i]
	}
	return retVal, nil
}

func (a u8sDummy) Lt(other ElOrd, same bool) (Array, error) {
	var b u8sDummy

	switch ot := other.(type) {
	case u8sDummy:
		b = ot
	case Byteser:
		b = u8sDummy(ot.Bytes())
	default:
		return nil, errors.Errorf(typeMismatch, "Lt", a, other)
	}

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(u8sDummy, len(a))
		for i, v := range a {
			if v < b[i] {
				retVal[i] = byte(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v < b[i]
	}
	return retVal, nil
}

/* Lte */

func (a f64sDummy) Lte(other ElOrd, same bool) (Array, error) {
	var b f64sDummy

	switch ot := other.(type) {
	case f64sDummy:
		b = ot
	case Float64ser:
		b = f64sDummy(ot.Float64s())
	default:
		return nil, errors.Errorf(typeMismatch, "Lte", a, other)
	}

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(f64sDummy, len(a))
		for i, v := range a {
			if v <= b[i] {
				retVal[i] = float64(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v <= b[i]
	}
	return retVal, nil
}

func (a f32sDummy) Lte(other ElOrd, same bool) (Array, error) {
	var b f32sDummy

	switch ot := other.(type) {
	case f32sDummy:
		b = ot
	case Float32ser:
		b = f32sDummy(ot.Float32s())
	default:
		return nil, errors.Errorf(typeMismatch, "Lte", a, other)
	}

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(f32sDummy, len(a))
		for i, v := range a {
			if v <= b[i] {
				retVal[i] = float32(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v <= b[i]
	}
	return retVal, nil
}

func (a intsDummy) Lte(other ElOrd, same bool) (Array, error) {
	var b intsDummy

	switch ot := other.(type) {
	case intsDummy:
		b = ot
	case Intser:
		b = intsDummy(ot.Ints())
	default:
		return nil, errors.Errorf(typeMismatch, "Lte", a, other)
	}

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(intsDummy, len(a))
		for i, v := range a {
			if v <= b[i] {
				retVal[i] = int(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v <= b[i]
	}
	return retVal, nil
}

func (a i64sDummy) Lte(other ElOrd, same bool) (Array, error) {
	var b i64sDummy

	switch ot := other.(type) {
	case i64sDummy:
		b = ot
	case Int64ser:
		b = i64sDummy(ot.Int64s())
	default:
		return nil, errors.Errorf(typeMismatch, "Lte", a, other)
	}

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(i64sDummy, len(a))
		for i, v := range a {
			if v <= b[i] {
				retVal[i] = int64(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v <= b[i]
	}
	return retVal, nil
}

func (a i32sDummy) Lte(other ElOrd, same bool) (Array, error) {
	var b i32sDummy

	switch ot := other.(type) {
	case i32sDummy:
		b = ot
	case Int32ser:
		b = i32sDummy(ot.Int32s())
	default:
		return nil, errors.Errorf(typeMismatch, "Lte", a, other)
	}

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(i32sDummy, len(a))
		for i, v := range a {
			if v <= b[i] {
				retVal[i] = int32(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v <= b[i]
	}
	return retVal, nil
}

func (a u8sDummy) Lte(other ElOrd, same bool) (Array, error) {
	var b u8sDummy

	switch ot := other.(type) {
	case u8sDummy:
		b = ot
	case Byteser:
		b = u8sDummy(ot.Bytes())
	default:
		return nil, errors.Errorf(typeMismatch, "Lte", a, other)
	}

	if len(a) != len(b) {
		return nil, errors.Errorf(lenMismatch, len(a), len(b))
	}

	if same {
		retVal := make(u8sDummy, len(a))
		for i, v := range a {
			if v <= b[i] {
				retVal[i] = byte(1)
			}
		}

		return retVal, nil
	}

	retVal := make(bs, len(a))
	for i, v := range a {
		retVal[i] = v <= b[i]
	}
	return retVal, nil
}

func (a f64sDummy) Slice(s Slice) (Array, error) {
	start, end, _, err := SliceDetails(s, len(a))
	if err != nil {
		return nil, err
	}
	return a[start:end], nil
}

func (a f64sDummy) Dtype() Dtype { return Float64 }
func (a f32sDummy) Slice(s Slice) (Array, error) {
	start, end, _, err := SliceDetails(s, len(a))
	if err != nil {
		return nil, err
	}
	return a[start:end], nil
}

func (a f32sDummy) Dtype() Dtype { return Float32 }
func (a intsDummy) Slice(s Slice) (Array, error) {
	start, end, _, err := SliceDetails(s, len(a))
	if err != nil {
		return nil, err
	}
	return a[start:end], nil
}

func (a intsDummy) Dtype() Dtype { return Int }
func (a i64sDummy) Slice(s Slice) (Array, error) {
	start, end, _, err := SliceDetails(s, len(a))
	if err != nil {
		return nil, err
	}
	return a[start:end], nil
}

func (a i64sDummy) Dtype() Dtype { return Int64 }
func (a i32sDummy) Slice(s Slice) (Array, error) {
	start, end, _, err := SliceDetails(s, len(a))
	if err != nil {
		return nil, err
	}
	return a[start:end], nil
}

func (a i32sDummy) Dtype() Dtype { return Int32 }
func (a u8sDummy) Slice(s Slice) (Array, error) {
	start, end, _, err := SliceDetails(s, len(a))
	if err != nil {
		return nil, err
	}
	return a[start:end], nil
}

func (a u8sDummy) Dtype() Dtype { return Byte }
func (a bsDummy) Slice(s Slice) (Array, error) {
	start, end, _, err := SliceDetails(s, len(a))
	if err != nil {
		return nil, err
	}
	return a[start:end], nil
}

func (a bsDummy) Dtype() Dtype { return Bool }
