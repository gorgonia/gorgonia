package tensor

import "github.com/pkg/errors"

/*
GENERATED FILE. DO NOT EDIT
*/

/* extraction functions */
func getFloat64s(a Array) ([]float64, error) {
	switch at := a.(type) {
	case f64s:
		return []float64(at), nil
	case Float64ser:
		return at.Float64s(), nil
	}
	return nil, errors.Errorf(extractionFail, "[]float64", a)
}

func getFloat64(a interface{}) (retVal float64, err error) {
	if b, ok := a.(float64); ok {
		return b, nil
	}
	err = errors.Errorf(extractionFail, "float64", a)
	return
}

func getFloat32s(a Array) ([]float32, error) {
	switch at := a.(type) {
	case f32s:
		return []float32(at), nil
	case Float32ser:
		return at.Float32s(), nil
	}
	return nil, errors.Errorf(extractionFail, "[]float32", a)
}

func getFloat32(a interface{}) (retVal float32, err error) {
	if b, ok := a.(float32); ok {
		return b, nil
	}
	err = errors.Errorf(extractionFail, "float32", a)
	return
}

func getInts(a Array) ([]int, error) {
	switch at := a.(type) {
	case ints:
		return []int(at), nil
	case Intser:
		return at.Ints(), nil
	}
	return nil, errors.Errorf(extractionFail, "[]int", a)
}

func getInt(a interface{}) (retVal int, err error) {
	if b, ok := a.(int); ok {
		return b, nil
	}
	err = errors.Errorf(extractionFail, "int", a)
	return
}

func getInt64s(a Array) ([]int64, error) {
	switch at := a.(type) {
	case i64s:
		return []int64(at), nil
	case Int64ser:
		return at.Int64s(), nil
	}
	return nil, errors.Errorf(extractionFail, "[]int64", a)
}

func getInt64(a interface{}) (retVal int64, err error) {
	if b, ok := a.(int64); ok {
		return b, nil
	}
	err = errors.Errorf(extractionFail, "int64", a)
	return
}

func getInt32s(a Array) ([]int32, error) {
	switch at := a.(type) {
	case i32s:
		return []int32(at), nil
	case Int32ser:
		return at.Int32s(), nil
	}
	return nil, errors.Errorf(extractionFail, "[]int32", a)
}

func getInt32(a interface{}) (retVal int32, err error) {
	if b, ok := a.(int32); ok {
		return b, nil
	}
	err = errors.Errorf(extractionFail, "int32", a)
	return
}

func getBytes(a Array) ([]byte, error) {
	switch at := a.(type) {
	case u8s:
		return []byte(at), nil
	case Byteser:
		return at.Bytes(), nil
	}
	return nil, errors.Errorf(extractionFail, "[]byte", a)
}

func getByte(a interface{}) (retVal byte, err error) {
	if b, ok := a.(byte); ok {
		return b, nil
	}
	err = errors.Errorf(extractionFail, "byte", a)
	return
}

func getBools(a Array) ([]bool, error) {
	switch at := a.(type) {
	case bs:
		return []bool(at), nil
	case Boolser:
		return at.Bools(), nil
	}
	return nil, errors.Errorf(extractionFail, "[]bool", a)
}

func getBool(a interface{}) (retVal bool, err error) {
	if b, ok := a.(bool); ok {
		return b, nil
	}
	err = errors.Errorf(extractionFail, "bool", a)
	return
}

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

/* Map */

func (a f64s) Map(fn interface{}) error {
	if f, ok := fn.(func(float64) float64); ok {
		for i, v := range a {
			a[i] = f(v)
		}
		return nil
	}
	return errors.Errorf(extractionFail, "func(x float64)float64", fn)
}

func (a f32s) Map(fn interface{}) error {
	if f, ok := fn.(func(float32) float32); ok {
		for i, v := range a {
			a[i] = f(v)
		}
		return nil
	}
	return errors.Errorf(extractionFail, "func(x float32)float32", fn)
}

func (a ints) Map(fn interface{}) error {
	if f, ok := fn.(func(int) int); ok {
		for i, v := range a {
			a[i] = f(v)
		}
		return nil
	}
	return errors.Errorf(extractionFail, "func(x int)int", fn)
}

func (a i64s) Map(fn interface{}) error {
	if f, ok := fn.(func(int64) int64); ok {
		for i, v := range a {
			a[i] = f(v)
		}
		return nil
	}
	return errors.Errorf(extractionFail, "func(x int64)int64", fn)
}

func (a i32s) Map(fn interface{}) error {
	if f, ok := fn.(func(int32) int32); ok {
		for i, v := range a {
			a[i] = f(v)
		}
		return nil
	}
	return errors.Errorf(extractionFail, "func(x int32)int32", fn)
}

func (a u8s) Map(fn interface{}) error {
	if f, ok := fn.(func(byte) byte); ok {
		for i, v := range a {
			a[i] = f(v)
		}
		return nil
	}
	return errors.Errorf(extractionFail, "func(x byte)byte", fn)
}

func (a bs) Map(fn interface{}) error {
	if f, ok := fn.(func(bool) bool); ok {
		for i, v := range a {
			a[i] = f(v)
		}
		return nil
	}
	return errors.Errorf(extractionFail, "func(x bool)bool", fn)
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

/* IncrMapper specialization */

func (a f64s) MapIncr(fn interface{}) error {
	if f, ok := fn.(func(float64) float64); ok {
		for i, v := range a {
			a[i] += f(v)
		}
		return nil
	}
	return errors.Errorf(extractionFail, "func(x float64)float64", fn)
}

func (a f32s) MapIncr(fn interface{}) error {
	if f, ok := fn.(func(float32) float32); ok {
		for i, v := range a {
			a[i] += f(v)
		}
		return nil
	}
	return errors.Errorf(extractionFail, "func(x float32)float32", fn)
}

func (a ints) MapIncr(fn interface{}) error {
	if f, ok := fn.(func(int) int); ok {
		for i, v := range a {
			a[i] += f(v)
		}
		return nil
	}
	return errors.Errorf(extractionFail, "func(x int)int", fn)
}

func (a i64s) MapIncr(fn interface{}) error {
	if f, ok := fn.(func(int64) int64); ok {
		for i, v := range a {
			a[i] += f(v)
		}
		return nil
	}
	return errors.Errorf(extractionFail, "func(x int64)int64", fn)
}

func (a i32s) MapIncr(fn interface{}) error {
	if f, ok := fn.(func(int32) int32); ok {
		for i, v := range a {
			a[i] += f(v)
		}
		return nil
	}
	return errors.Errorf(extractionFail, "func(x int32)int32", fn)
}

func (a u8s) MapIncr(fn interface{}) error {
	if f, ok := fn.(func(byte) byte); ok {
		for i, v := range a {
			a[i] += f(v)
		}
		return nil
	}
	return errors.Errorf(extractionFail, "func(x byte)byte", fn)
}

/* IterMapper specialization */

func (a f64s) IterMap(other Array, it, ot *FlatIterator, fn interface{}, incr bool) (err error) {
	// check noop
	if other == nil && ot == nil && fn == nil {
		return nil
	}

	// check types first
	var b []float64
	if other != nil {
		if b, err = getFloat64s(other); err != nil {
			return
		}
	}

	var f func(float64) float64
	var ok bool
	if fn != nil {
		if f, ok = fn.(func(float64) float64); !ok {
			return errors.Errorf(extractionFail, "func(float64)float64", f)
		}
	}

	switch {
	case other == nil && it == nil && ot == nil && fn != nil:
		// basic case: this is just a.Map(fn)
		return a.Map(f)
	case other == nil && it != nil && ot == nil && fn != nil:
		// basically this is apply function with iterator guidance
		var next int
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			if _, noop := err.(NoOpError); err != nil && !noop {
				return
			}
			if incr {
				a[next] += f(a[next])
			} else {
				a[next] = f(a[next])
			}
		}
		return nil
	case other != nil && it == nil && ot == nil:
		// the case where a[i] = b[i]
		if len(a) != len(b) {
			return errors.Errorf(sizeMismatch, len(a), len(b))
		}
		a = a[:len(a)] // optim for BCE
		b = b[:len(a)] // optim for BCE

		switch {
		case incr && fn == nil:
			for i, v := range b {
				a[i] += v
			}
			return nil
		case incr && fn != nil:
			for i, v := range b {
				a[i] += f(v)
			}
			return nil
		case !incr && fn == nil:
			for i, v := range b {
				a[i] = v
			}
		case !incr && fn != nil:
			for i, v := range b {
				a[i] = f(v)
			}
		}
	case other != nil && it != nil && ot == nil:
		// case where assignment of a = b; where a is guided by it
		var next, j int
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			if _, noop := err.(NoOpError); err != nil && !noop {
				return
			}

			switch {
			case incr && fn == nil:
				a[next] += b[j]
			case incr && fn != nil:
				a[next] += f(b[j])
			case !incr && fn == nil:
				a[next] = b[j]
			case !incr && fn != nil:
				a[next] = f(b[j])
			}

			j++
		}
		return nil
	case other != nil && it == nil && ot != nil:
		// case where assignment of a = b; where b is guided by ot
		var next, i int
		for next, err = ot.Next(); err == nil; next, err = ot.Next() {
			if _, noop := err.(NoOpError); err != nil && !noop {
				return
			}

			switch {
			case incr && fn == nil:
				a[i] += b[next]
			case incr && fn != nil:
				a[i] += f(b[next])
			case !incr && fn == nil:
				a[i] = b[next]
			case !incr && fn != nil:
				a[i] = f(b[next])
			}

			i++
		}
		return nil
	case other != nil && it != nil && ot != nil:
		// case where assignment of a = b; and both a and b are guided by it and ot respectively
		var i, j int
		for {
			if i, err = it.Next(); err != nil {
				if _, ok := err.(NoOpError); !ok {
					return err
				}
				err = nil
				break
			}
			if j, err = ot.Next(); err != nil {
				if _, ok := err.(NoOpError); !ok {
					return err
				}
				err = nil
				break
			}

			switch {
			case incr && fn == nil:
				a[i] += b[j]
			case incr && fn != nil:
				a[i] += f(b[j])
			case !incr && fn == nil:
				a[i] = b[j]
			case !incr && fn != nil:
				a[i] = f(b[j])
			}

			return nil
		}
	case other == nil && ot != nil:
		// error - stupid
		return errors.Errorf("Meaningless state - other is nil, ot is not")
	}
	return
}

func (a f32s) IterMap(other Array, it, ot *FlatIterator, fn interface{}, incr bool) (err error) {
	// check noop
	if other == nil && ot == nil && fn == nil {
		return nil
	}

	// check types first
	var b []float32
	if other != nil {
		if b, err = getFloat32s(other); err != nil {
			return
		}
	}

	var f func(float32) float32
	var ok bool
	if fn != nil {
		if f, ok = fn.(func(float32) float32); !ok {
			return errors.Errorf(extractionFail, "func(float32)float32", f)
		}
	}

	switch {
	case other == nil && it == nil && ot == nil && fn != nil:
		// basic case: this is just a.Map(fn)
		return a.Map(f)
	case other == nil && it != nil && ot == nil && fn != nil:
		// basically this is apply function with iterator guidance
		var next int
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			if _, noop := err.(NoOpError); err != nil && !noop {
				return
			}
			if incr {
				a[next] += f(a[next])
			} else {
				a[next] = f(a[next])
			}
		}
		return nil
	case other != nil && it == nil && ot == nil:
		// the case where a[i] = b[i]
		if len(a) != len(b) {
			return errors.Errorf(sizeMismatch, len(a), len(b))
		}
		a = a[:len(a)] // optim for BCE
		b = b[:len(a)] // optim for BCE

		switch {
		case incr && fn == nil:
			for i, v := range b {
				a[i] += v
			}
			return nil
		case incr && fn != nil:
			for i, v := range b {
				a[i] += f(v)
			}
			return nil
		case !incr && fn == nil:
			for i, v := range b {
				a[i] = v
			}
		case !incr && fn != nil:
			for i, v := range b {
				a[i] = f(v)
			}
		}
	case other != nil && it != nil && ot == nil:
		// case where assignment of a = b; where a is guided by it
		var next, j int
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			if _, noop := err.(NoOpError); err != nil && !noop {
				return
			}

			switch {
			case incr && fn == nil:
				a[next] += b[j]
			case incr && fn != nil:
				a[next] += f(b[j])
			case !incr && fn == nil:
				a[next] = b[j]
			case !incr && fn != nil:
				a[next] = f(b[j])
			}

			j++
		}
		return nil
	case other != nil && it == nil && ot != nil:
		// case where assignment of a = b; where b is guided by ot
		var next, i int
		for next, err = ot.Next(); err == nil; next, err = ot.Next() {
			if _, noop := err.(NoOpError); err != nil && !noop {
				return
			}

			switch {
			case incr && fn == nil:
				a[i] += b[next]
			case incr && fn != nil:
				a[i] += f(b[next])
			case !incr && fn == nil:
				a[i] = b[next]
			case !incr && fn != nil:
				a[i] = f(b[next])
			}

			i++
		}
		return nil
	case other != nil && it != nil && ot != nil:
		// case where assignment of a = b; and both a and b are guided by it and ot respectively
		var i, j int
		for {
			if i, err = it.Next(); err != nil {
				if _, ok := err.(NoOpError); !ok {
					return err
				}
				err = nil
				break
			}
			if j, err = ot.Next(); err != nil {
				if _, ok := err.(NoOpError); !ok {
					return err
				}
				err = nil
				break
			}

			switch {
			case incr && fn == nil:
				a[i] += b[j]
			case incr && fn != nil:
				a[i] += f(b[j])
			case !incr && fn == nil:
				a[i] = b[j]
			case !incr && fn != nil:
				a[i] = f(b[j])
			}

			return nil
		}
	case other == nil && ot != nil:
		// error - stupid
		return errors.Errorf("Meaningless state - other is nil, ot is not")
	}
	return
}

func (a ints) IterMap(other Array, it, ot *FlatIterator, fn interface{}, incr bool) (err error) {
	// check noop
	if other == nil && ot == nil && fn == nil {
		return nil
	}

	// check types first
	var b []int
	if other != nil {
		if b, err = getInts(other); err != nil {
			return
		}
	}

	var f func(int) int
	var ok bool
	if fn != nil {
		if f, ok = fn.(func(int) int); !ok {
			return errors.Errorf(extractionFail, "func(int)int", f)
		}
	}

	switch {
	case other == nil && it == nil && ot == nil && fn != nil:
		// basic case: this is just a.Map(fn)
		return a.Map(f)
	case other == nil && it != nil && ot == nil && fn != nil:
		// basically this is apply function with iterator guidance
		var next int
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			if _, noop := err.(NoOpError); err != nil && !noop {
				return
			}
			if incr {
				a[next] += f(a[next])
			} else {
				a[next] = f(a[next])
			}
		}
		return nil
	case other != nil && it == nil && ot == nil:
		// the case where a[i] = b[i]
		if len(a) != len(b) {
			return errors.Errorf(sizeMismatch, len(a), len(b))
		}
		a = a[:len(a)] // optim for BCE
		b = b[:len(a)] // optim for BCE

		switch {
		case incr && fn == nil:
			for i, v := range b {
				a[i] += v
			}
			return nil
		case incr && fn != nil:
			for i, v := range b {
				a[i] += f(v)
			}
			return nil
		case !incr && fn == nil:
			for i, v := range b {
				a[i] = v
			}
		case !incr && fn != nil:
			for i, v := range b {
				a[i] = f(v)
			}
		}
	case other != nil && it != nil && ot == nil:
		// case where assignment of a = b; where a is guided by it
		var next, j int
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			if _, noop := err.(NoOpError); err != nil && !noop {
				return
			}

			switch {
			case incr && fn == nil:
				a[next] += b[j]
			case incr && fn != nil:
				a[next] += f(b[j])
			case !incr && fn == nil:
				a[next] = b[j]
			case !incr && fn != nil:
				a[next] = f(b[j])
			}

			j++
		}
		return nil
	case other != nil && it == nil && ot != nil:
		// case where assignment of a = b; where b is guided by ot
		var next, i int
		for next, err = ot.Next(); err == nil; next, err = ot.Next() {
			if _, noop := err.(NoOpError); err != nil && !noop {
				return
			}

			switch {
			case incr && fn == nil:
				a[i] += b[next]
			case incr && fn != nil:
				a[i] += f(b[next])
			case !incr && fn == nil:
				a[i] = b[next]
			case !incr && fn != nil:
				a[i] = f(b[next])
			}

			i++
		}
		return nil
	case other != nil && it != nil && ot != nil:
		// case where assignment of a = b; and both a and b are guided by it and ot respectively
		var i, j int
		for {
			if i, err = it.Next(); err != nil {
				if _, ok := err.(NoOpError); !ok {
					return err
				}
				err = nil
				break
			}
			if j, err = ot.Next(); err != nil {
				if _, ok := err.(NoOpError); !ok {
					return err
				}
				err = nil
				break
			}

			switch {
			case incr && fn == nil:
				a[i] += b[j]
			case incr && fn != nil:
				a[i] += f(b[j])
			case !incr && fn == nil:
				a[i] = b[j]
			case !incr && fn != nil:
				a[i] = f(b[j])
			}

			return nil
		}
	case other == nil && ot != nil:
		// error - stupid
		return errors.Errorf("Meaningless state - other is nil, ot is not")
	}
	return
}

func (a i64s) IterMap(other Array, it, ot *FlatIterator, fn interface{}, incr bool) (err error) {
	// check noop
	if other == nil && ot == nil && fn == nil {
		return nil
	}

	// check types first
	var b []int64
	if other != nil {
		if b, err = getInt64s(other); err != nil {
			return
		}
	}

	var f func(int64) int64
	var ok bool
	if fn != nil {
		if f, ok = fn.(func(int64) int64); !ok {
			return errors.Errorf(extractionFail, "func(int64)int64", f)
		}
	}

	switch {
	case other == nil && it == nil && ot == nil && fn != nil:
		// basic case: this is just a.Map(fn)
		return a.Map(f)
	case other == nil && it != nil && ot == nil && fn != nil:
		// basically this is apply function with iterator guidance
		var next int
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			if _, noop := err.(NoOpError); err != nil && !noop {
				return
			}
			if incr {
				a[next] += f(a[next])
			} else {
				a[next] = f(a[next])
			}
		}
		return nil
	case other != nil && it == nil && ot == nil:
		// the case where a[i] = b[i]
		if len(a) != len(b) {
			return errors.Errorf(sizeMismatch, len(a), len(b))
		}
		a = a[:len(a)] // optim for BCE
		b = b[:len(a)] // optim for BCE

		switch {
		case incr && fn == nil:
			for i, v := range b {
				a[i] += v
			}
			return nil
		case incr && fn != nil:
			for i, v := range b {
				a[i] += f(v)
			}
			return nil
		case !incr && fn == nil:
			for i, v := range b {
				a[i] = v
			}
		case !incr && fn != nil:
			for i, v := range b {
				a[i] = f(v)
			}
		}
	case other != nil && it != nil && ot == nil:
		// case where assignment of a = b; where a is guided by it
		var next, j int
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			if _, noop := err.(NoOpError); err != nil && !noop {
				return
			}

			switch {
			case incr && fn == nil:
				a[next] += b[j]
			case incr && fn != nil:
				a[next] += f(b[j])
			case !incr && fn == nil:
				a[next] = b[j]
			case !incr && fn != nil:
				a[next] = f(b[j])
			}

			j++
		}
		return nil
	case other != nil && it == nil && ot != nil:
		// case where assignment of a = b; where b is guided by ot
		var next, i int
		for next, err = ot.Next(); err == nil; next, err = ot.Next() {
			if _, noop := err.(NoOpError); err != nil && !noop {
				return
			}

			switch {
			case incr && fn == nil:
				a[i] += b[next]
			case incr && fn != nil:
				a[i] += f(b[next])
			case !incr && fn == nil:
				a[i] = b[next]
			case !incr && fn != nil:
				a[i] = f(b[next])
			}

			i++
		}
		return nil
	case other != nil && it != nil && ot != nil:
		// case where assignment of a = b; and both a and b are guided by it and ot respectively
		var i, j int
		for {
			if i, err = it.Next(); err != nil {
				if _, ok := err.(NoOpError); !ok {
					return err
				}
				err = nil
				break
			}
			if j, err = ot.Next(); err != nil {
				if _, ok := err.(NoOpError); !ok {
					return err
				}
				err = nil
				break
			}

			switch {
			case incr && fn == nil:
				a[i] += b[j]
			case incr && fn != nil:
				a[i] += f(b[j])
			case !incr && fn == nil:
				a[i] = b[j]
			case !incr && fn != nil:
				a[i] = f(b[j])
			}

			return nil
		}
	case other == nil && ot != nil:
		// error - stupid
		return errors.Errorf("Meaningless state - other is nil, ot is not")
	}
	return
}

func (a i32s) IterMap(other Array, it, ot *FlatIterator, fn interface{}, incr bool) (err error) {
	// check noop
	if other == nil && ot == nil && fn == nil {
		return nil
	}

	// check types first
	var b []int32
	if other != nil {
		if b, err = getInt32s(other); err != nil {
			return
		}
	}

	var f func(int32) int32
	var ok bool
	if fn != nil {
		if f, ok = fn.(func(int32) int32); !ok {
			return errors.Errorf(extractionFail, "func(int32)int32", f)
		}
	}

	switch {
	case other == nil && it == nil && ot == nil && fn != nil:
		// basic case: this is just a.Map(fn)
		return a.Map(f)
	case other == nil && it != nil && ot == nil && fn != nil:
		// basically this is apply function with iterator guidance
		var next int
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			if _, noop := err.(NoOpError); err != nil && !noop {
				return
			}
			if incr {
				a[next] += f(a[next])
			} else {
				a[next] = f(a[next])
			}
		}
		return nil
	case other != nil && it == nil && ot == nil:
		// the case where a[i] = b[i]
		if len(a) != len(b) {
			return errors.Errorf(sizeMismatch, len(a), len(b))
		}
		a = a[:len(a)] // optim for BCE
		b = b[:len(a)] // optim for BCE

		switch {
		case incr && fn == nil:
			for i, v := range b {
				a[i] += v
			}
			return nil
		case incr && fn != nil:
			for i, v := range b {
				a[i] += f(v)
			}
			return nil
		case !incr && fn == nil:
			for i, v := range b {
				a[i] = v
			}
		case !incr && fn != nil:
			for i, v := range b {
				a[i] = f(v)
			}
		}
	case other != nil && it != nil && ot == nil:
		// case where assignment of a = b; where a is guided by it
		var next, j int
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			if _, noop := err.(NoOpError); err != nil && !noop {
				return
			}

			switch {
			case incr && fn == nil:
				a[next] += b[j]
			case incr && fn != nil:
				a[next] += f(b[j])
			case !incr && fn == nil:
				a[next] = b[j]
			case !incr && fn != nil:
				a[next] = f(b[j])
			}

			j++
		}
		return nil
	case other != nil && it == nil && ot != nil:
		// case where assignment of a = b; where b is guided by ot
		var next, i int
		for next, err = ot.Next(); err == nil; next, err = ot.Next() {
			if _, noop := err.(NoOpError); err != nil && !noop {
				return
			}

			switch {
			case incr && fn == nil:
				a[i] += b[next]
			case incr && fn != nil:
				a[i] += f(b[next])
			case !incr && fn == nil:
				a[i] = b[next]
			case !incr && fn != nil:
				a[i] = f(b[next])
			}

			i++
		}
		return nil
	case other != nil && it != nil && ot != nil:
		// case where assignment of a = b; and both a and b are guided by it and ot respectively
		var i, j int
		for {
			if i, err = it.Next(); err != nil {
				if _, ok := err.(NoOpError); !ok {
					return err
				}
				err = nil
				break
			}
			if j, err = ot.Next(); err != nil {
				if _, ok := err.(NoOpError); !ok {
					return err
				}
				err = nil
				break
			}

			switch {
			case incr && fn == nil:
				a[i] += b[j]
			case incr && fn != nil:
				a[i] += f(b[j])
			case !incr && fn == nil:
				a[i] = b[j]
			case !incr && fn != nil:
				a[i] = f(b[j])
			}

			return nil
		}
	case other == nil && ot != nil:
		// error - stupid
		return errors.Errorf("Meaningless state - other is nil, ot is not")
	}
	return
}

func (a u8s) IterMap(other Array, it, ot *FlatIterator, fn interface{}, incr bool) (err error) {
	// check noop
	if other == nil && ot == nil && fn == nil {
		return nil
	}

	// check types first
	var b []byte
	if other != nil {
		if b, err = getBytes(other); err != nil {
			return
		}
	}

	var f func(byte) byte
	var ok bool
	if fn != nil {
		if f, ok = fn.(func(byte) byte); !ok {
			return errors.Errorf(extractionFail, "func(byte)byte", f)
		}
	}

	switch {
	case other == nil && it == nil && ot == nil && fn != nil:
		// basic case: this is just a.Map(fn)
		return a.Map(f)
	case other == nil && it != nil && ot == nil && fn != nil:
		// basically this is apply function with iterator guidance
		var next int
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			if _, noop := err.(NoOpError); err != nil && !noop {
				return
			}
			if incr {
				a[next] += f(a[next])
			} else {
				a[next] = f(a[next])
			}
		}
		return nil
	case other != nil && it == nil && ot == nil:
		// the case where a[i] = b[i]
		if len(a) != len(b) {
			return errors.Errorf(sizeMismatch, len(a), len(b))
		}
		a = a[:len(a)] // optim for BCE
		b = b[:len(a)] // optim for BCE

		switch {
		case incr && fn == nil:
			for i, v := range b {
				a[i] += v
			}
			return nil
		case incr && fn != nil:
			for i, v := range b {
				a[i] += f(v)
			}
			return nil
		case !incr && fn == nil:
			for i, v := range b {
				a[i] = v
			}
		case !incr && fn != nil:
			for i, v := range b {
				a[i] = f(v)
			}
		}
	case other != nil && it != nil && ot == nil:
		// case where assignment of a = b; where a is guided by it
		var next, j int
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			if _, noop := err.(NoOpError); err != nil && !noop {
				return
			}

			switch {
			case incr && fn == nil:
				a[next] += b[j]
			case incr && fn != nil:
				a[next] += f(b[j])
			case !incr && fn == nil:
				a[next] = b[j]
			case !incr && fn != nil:
				a[next] = f(b[j])
			}

			j++
		}
		return nil
	case other != nil && it == nil && ot != nil:
		// case where assignment of a = b; where b is guided by ot
		var next, i int
		for next, err = ot.Next(); err == nil; next, err = ot.Next() {
			if _, noop := err.(NoOpError); err != nil && !noop {
				return
			}

			switch {
			case incr && fn == nil:
				a[i] += b[next]
			case incr && fn != nil:
				a[i] += f(b[next])
			case !incr && fn == nil:
				a[i] = b[next]
			case !incr && fn != nil:
				a[i] = f(b[next])
			}

			i++
		}
		return nil
	case other != nil && it != nil && ot != nil:
		// case where assignment of a = b; and both a and b are guided by it and ot respectively
		var i, j int
		for {
			if i, err = it.Next(); err != nil {
				if _, ok := err.(NoOpError); !ok {
					return err
				}
				err = nil
				break
			}
			if j, err = ot.Next(); err != nil {
				if _, ok := err.(NoOpError); !ok {
					return err
				}
				err = nil
				break
			}

			switch {
			case incr && fn == nil:
				a[i] += b[j]
			case incr && fn != nil:
				a[i] += f(b[j])
			case !incr && fn == nil:
				a[i] = b[j]
			case !incr && fn != nil:
				a[i] = f(b[j])
			}

			return nil
		}
	case other == nil && ot != nil:
		// error - stupid
		return errors.Errorf("Meaningless state - other is nil, ot is not")
	}
	return
}

func (a bs) IterMap(other Array, it, ot *FlatIterator, fn interface{}, incr bool) (err error) {
	// check noop
	if other == nil && ot == nil && fn == nil {
		return nil
	}

	// check types first
	var b []bool
	if other != nil {
		if b, err = getBools(other); err != nil {
			return
		}
	}

	var f func(bool) bool
	var ok bool
	if fn != nil {
		if f, ok = fn.(func(bool) bool); !ok {
			return errors.Errorf(extractionFail, "func(bool)bool", f)
		}
	}

	switch {
	case other == nil && it == nil && ot == nil && fn != nil:
		// basic case: this is just a.Map(fn)
		return a.Map(f)
	case other == nil && it != nil && ot == nil && fn != nil:
		// basically this is apply function with iterator guidance
		var next int
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			if _, noop := err.(NoOpError); err != nil && !noop {
				return
			}
			a[next] = f(a[next])
		}
		return nil
	case other != nil && it == nil && ot == nil:
		// the case where a[i] = b[i]
		if len(a) != len(b) {
			return errors.Errorf(sizeMismatch, len(a), len(b))
		}
		a = a[:len(a)] // optim for BCE
		b = b[:len(a)] // optim for BCE

		if fn == nil {
			for i, v := range b {
				a[i] = v
			}
		} else {
			for i, v := range b {
				a[i] = f(v)
			}
		}
		return nil
	case other != nil && it != nil && ot == nil:
		// case where assignment of a = b; where a is guided by it
		var next, j int
		for next, err = it.Next(); err == nil; next, err = it.Next() {
			if _, noop := err.(NoOpError); err != nil && !noop {
				return
			}

			if fn == nil {
				a[next] = b[j]
			} else {
				a[next] = f(b[j])
			}

			j++
		}
		return nil
	case other != nil && it == nil && ot != nil:
		// case where assignment of a = b; where b is guided by ot
		var next, i int
		for next, err = ot.Next(); err == nil; next, err = ot.Next() {
			if _, noop := err.(NoOpError); err != nil && !noop {
				return
			}

			if fn == nil {
				a[i] = b[next]
			} else {
				a[i] = f(b[next])
			}

			i++
		}
		return nil
	case other != nil && it != nil && ot != nil:
		// case where assignment of a = b; and both a and b are guided by it and ot respectively
		var i, j int
		for {
			if i, err = it.Next(); err != nil {
				if _, ok := err.(NoOpError); !ok {
					return err
				}
				err = nil
				break
			}
			if j, err = ot.Next(); err != nil {
				if _, ok := err.(NoOpError); !ok {
					return err
				}
				err = nil
				break
			}

			if fn == nil {
				a[i] = b[j]
			} else {
				a[i] = f(b[j])
			}

			return nil
		}
	case other == nil && ot != nil:
		// error - stupid
		return errors.Errorf("Meaningless state - other is nil, ot is not")
	}
	return
}
