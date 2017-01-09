package tensor

import "github.com/pkg/errors"

/* Len */

func (a f64s) Len() int { return len(a) }
func (a f32s) Len() int { return len(a) }
func (a ints) Len() int { return len(a) }
func (a i64s) Len() int { return len(a) }
func (a i32s) Len() int { return len(a) }
func (a u8s) Len() int  { return len(a) }
func (a bs) Len() int   { return len(a) }

/* Cap */

func (a f64s) Cap() int { return cap(a) }
func (a f32s) Cap() int { return cap(a) }
func (a ints) Cap() int { return cap(a) }
func (a i64s) Cap() int { return cap(a) }
func (a i32s) Cap() int { return cap(a) }
func (a u8s) Cap() int  { return cap(a) }
func (a bs) Cap() int   { return cap(a) }

/* Data */

func (a f64s) Data() interface{} { return []float64(a) }
func (a f32s) Data() interface{} { return []float32(a) }
func (a ints) Data() interface{} { return []int(a) }
func (a i64s) Data() interface{} { return []int64(a) }
func (a i32s) Data() interface{} { return []int32(a) }
func (a u8s) Data() interface{}  { return []byte(a) }
func (a bs) Data() interface{}   { return []bool(a) }

/* Get */

func (a f64s) Get(i int) interface{} { return a[i] }
func (a f32s) Get(i int) interface{} { return a[i] }
func (a ints) Get(i int) interface{} { return a[i] }
func (a i64s) Get(i int) interface{} { return a[i] }
func (a i32s) Get(i int) interface{} { return a[i] }
func (a u8s) Get(i int) interface{}  { return a[i] }
func (a bs) Get(i int) interface{}   { return a[i] }

/* Set */

func (a f64s) Set(i int, v interface{}) error {
	if f, ok := v.(float64); ok {
		a[i] = f
		return nil
	}
	return errors.Errorf("Cannot set %v of %T to []float64", v, v)
}

func (a f32s) Set(i int, v interface{}) error {
	if f, ok := v.(float32); ok {
		a[i] = f
		return nil
	}
	return errors.Errorf("Cannot set %v of %T to []float32", v, v)
}

func (a ints) Set(i int, v interface{}) error {
	if f, ok := v.(int); ok {
		a[i] = f
		return nil
	}
	return errors.Errorf("Cannot set %v of %T to []int", v, v)
}

func (a i64s) Set(i int, v interface{}) error {
	if f, ok := v.(int64); ok {
		a[i] = f
		return nil
	}
	return errors.Errorf("Cannot set %v of %T to []int64", v, v)
}

func (a i32s) Set(i int, v interface{}) error {
	if f, ok := v.(int32); ok {
		a[i] = f
		return nil
	}
	return errors.Errorf("Cannot set %v of %T to []int32", v, v)
}

func (a u8s) Set(i int, v interface{}) error {
	if f, ok := v.(byte); ok {
		a[i] = f
		return nil
	}
	return errors.Errorf("Cannot set %v of %T to []byte", v, v)
}

func (a bs) Set(i int, v interface{}) error {
	if f, ok := v.(bool); ok {
		a[i] = f
		return nil
	}
	return errors.Errorf("Cannot set %v of %T to []bool", v, v)
}

/* Eq */

func (a f64s) Eq(other interface{}) bool {
	if b, ok := other.(f64s); ok {
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

func (a f32s) Eq(other interface{}) bool {
	if b, ok := other.(f32s); ok {
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

func (a ints) Eq(other interface{}) bool {
	if b, ok := other.(ints); ok {
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

func (a i64s) Eq(other interface{}) bool {
	if b, ok := other.(i64s); ok {
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

func (a i32s) Eq(other interface{}) bool {
	if b, ok := other.(i32s); ok {
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

func (a u8s) Eq(other interface{}) bool {
	if b, ok := other.(u8s); ok {
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

func (a bs) Eq(other interface{}) bool {
	if b, ok := other.(bs); ok {
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

func (a f64s) Zero() {
	for i := range a {
		a[i] = float64(0)
	}
}

func (a f32s) Zero() {
	for i := range a {
		a[i] = float32(0)
	}
}

func (a ints) Zero() {
	for i := range a {
		a[i] = int(0)
	}
}

func (a i64s) Zero() {
	for i := range a {
		a[i] = int64(0)
	}
}

func (a i32s) Zero() {
	for i := range a {
		a[i] = int32(0)
	}
}

func (a u8s) Zero() {
	for i := range a {
		a[i] = byte(0)
	}
}

func (a bs) Zero() {
	for i := range a {
		a[i] = false
	}
}

/* Oner */

func (a f64s) One() {
	for i := range a {
		a[i] = float64(1)
	}
}

func (a f32s) One() {
	for i := range a {
		a[i] = float32(1)
	}
}

func (a ints) One() {
	for i := range a {
		a[i] = int(1)
	}
}

func (a i64s) One() {
	for i := range a {
		a[i] = int64(1)
	}
}

func (a i32s) One() {
	for i := range a {
		a[i] = int32(1)
	}
}

func (a u8s) One() {
	for i := range a {
		a[i] = byte(1)
	}
}

func (a bs) One() {
	for i := range a {
		a[i] = true
	}
}

/* CopierFrom */

func (a f64s) CopyFrom(other interface{}) (int, error) {
	switch b := other.(type) {
	case f64s:
		return copy(a, b), nil
	case []float64:
		return copy(a, b), nil
	case Float64ser:
		return copy(a, b.Float64s()), nil
	}

	return 0, errors.Errorf("Cannot copy from %T", other)
}

func (a f32s) CopyFrom(other interface{}) (int, error) {
	switch b := other.(type) {
	case f32s:
		return copy(a, b), nil
	case []float32:
		return copy(a, b), nil
	case Float32ser:
		return copy(a, b.Float32s()), nil
	}

	return 0, errors.Errorf("Cannot copy from %T", other)
}

func (a ints) CopyFrom(other interface{}) (int, error) {
	switch b := other.(type) {
	case ints:
		return copy(a, b), nil
	case []int:
		return copy(a, b), nil
	case Intser:
		return copy(a, b.Ints()), nil
	}

	return 0, errors.Errorf("Cannot copy from %T", other)
}

func (a i64s) CopyFrom(other interface{}) (int, error) {
	switch b := other.(type) {
	case i64s:
		return copy(a, b), nil
	case []int64:
		return copy(a, b), nil
	case Int64ser:
		return copy(a, b.Int64s()), nil
	}

	return 0, errors.Errorf("Cannot copy from %T", other)
}

func (a i32s) CopyFrom(other interface{}) (int, error) {
	switch b := other.(type) {
	case i32s:
		return copy(a, b), nil
	case []int32:
		return copy(a, b), nil
	case Int32ser:
		return copy(a, b.Int32s()), nil
	}

	return 0, errors.Errorf("Cannot copy from %T", other)
}

func (a u8s) CopyFrom(other interface{}) (int, error) {
	switch b := other.(type) {
	case u8s:
		return copy(a, b), nil
	case []byte:
		return copy(a, b), nil
	case Byteser:
		return copy(a, b.Bytes()), nil
	}

	return 0, errors.Errorf("Cannot copy from %T", other)
}

func (a bs) CopyFrom(other interface{}) (int, error) {
	switch b := other.(type) {
	case bs:
		return copy(a, b), nil
	case []bool:
		return copy(a, b), nil
	case Boolser:
		return copy(a, b.Bools()), nil
	}

	return 0, errors.Errorf("Cannot copy from %T", other)
}

/* Transpose Specialization */

func (a f64s) Transpose(oldShape, oldStrides, axes, newStrides []int) {
	size := len(a)
	track := NewBitMap(size)
	track.Set(0)
	track.Set(size - 1) // first and last don't change

	var saved, tmp float64
	var i int

	for i = 1; ; {
		dest := TransposeIndex(i, oldShape, axes, oldStrides, newStrides)

		if track.IsSet(i) && track.IsSet(dest) {
			a[i] = saved
			saved = float64(0)

			for i < size && track.IsSet(i) {
				i++
			}

			if i >= size {
				break
			}
			continue
		}

		track.Set(i)
		tmp = a[i]
		a[i] = saved
		saved = tmp

		i = dest
	}
}

func (a f32s) Transpose(oldShape, oldStrides, axes, newStrides []int) {
	size := len(a)
	track := NewBitMap(size)
	track.Set(0)
	track.Set(size - 1) // first and last don't change

	var saved, tmp float32
	var i int

	for i = 1; ; {
		dest := TransposeIndex(i, oldShape, axes, oldStrides, newStrides)

		if track.IsSet(i) && track.IsSet(dest) {
			a[i] = saved
			saved = float32(0)

			for i < size && track.IsSet(i) {
				i++
			}

			if i >= size {
				break
			}
			continue
		}

		track.Set(i)
		tmp = a[i]
		a[i] = saved
		saved = tmp

		i = dest
	}
}

func (a ints) Transpose(oldShape, oldStrides, axes, newStrides []int) {
	size := len(a)
	track := NewBitMap(size)
	track.Set(0)
	track.Set(size - 1) // first and last don't change

	var saved, tmp int
	var i int

	for i = 1; ; {
		dest := TransposeIndex(i, oldShape, axes, oldStrides, newStrides)

		if track.IsSet(i) && track.IsSet(dest) {
			a[i] = saved
			saved = int(0)

			for i < size && track.IsSet(i) {
				i++
			}

			if i >= size {
				break
			}
			continue
		}

		track.Set(i)
		tmp = a[i]
		a[i] = saved
		saved = tmp

		i = dest
	}
}

func (a i64s) Transpose(oldShape, oldStrides, axes, newStrides []int) {
	size := len(a)
	track := NewBitMap(size)
	track.Set(0)
	track.Set(size - 1) // first and last don't change

	var saved, tmp int64
	var i int

	for i = 1; ; {
		dest := TransposeIndex(i, oldShape, axes, oldStrides, newStrides)

		if track.IsSet(i) && track.IsSet(dest) {
			a[i] = saved
			saved = int64(0)

			for i < size && track.IsSet(i) {
				i++
			}

			if i >= size {
				break
			}
			continue
		}

		track.Set(i)
		tmp = a[i]
		a[i] = saved
		saved = tmp

		i = dest
	}
}

func (a i32s) Transpose(oldShape, oldStrides, axes, newStrides []int) {
	size := len(a)
	track := NewBitMap(size)
	track.Set(0)
	track.Set(size - 1) // first and last don't change

	var saved, tmp int32
	var i int

	for i = 1; ; {
		dest := TransposeIndex(i, oldShape, axes, oldStrides, newStrides)

		if track.IsSet(i) && track.IsSet(dest) {
			a[i] = saved
			saved = int32(0)

			for i < size && track.IsSet(i) {
				i++
			}

			if i >= size {
				break
			}
			continue
		}

		track.Set(i)
		tmp = a[i]
		a[i] = saved
		saved = tmp

		i = dest
	}
}

func (a u8s) Transpose(oldShape, oldStrides, axes, newStrides []int) {
	size := len(a)
	track := NewBitMap(size)
	track.Set(0)
	track.Set(size - 1) // first and last don't change

	var saved, tmp byte
	var i int

	for i = 1; ; {
		dest := TransposeIndex(i, oldShape, axes, oldStrides, newStrides)

		if track.IsSet(i) && track.IsSet(dest) {
			a[i] = saved
			saved = byte(0)

			for i < size && track.IsSet(i) {
				i++
			}

			if i >= size {
				break
			}
			continue
		}

		track.Set(i)
		tmp = a[i]
		a[i] = saved
		saved = tmp

		i = dest
	}
}

func (a bs) Transpose(oldShape, oldStrides, axes, newStrides []int) {
	size := len(a)
	track := NewBitMap(size)
	track.Set(0)
	track.Set(size - 1) // first and last don't change

	var saved, tmp bool
	var i int

	for i = 1; ; {
		dest := TransposeIndex(i, oldShape, axes, oldStrides, newStrides)

		if track.IsSet(i) && track.IsSet(dest) {
			a[i] = saved
			saved = false

			for i < size && track.IsSet(i) {
				i++
			}

			if i >= size {
				break
			}
			continue
		}

		track.Set(i)
		tmp = a[i]
		a[i] = saved
		saved = tmp

		i = dest
	}
}
