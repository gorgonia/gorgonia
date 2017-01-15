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

/* Compat */

func (a f64sDummy) Float64s() []float64 { return []float64(a) }
func (a f32sDummy) Float32s() []float32 { return []float32(a) }
func (a intsDummy) Ints() []int         { return []int(a) }
func (a i64sDummy) Int64s() []int64     { return []int64(a) }
func (a i32sDummy) Int32s() []int32     { return []int32(a) }
func (a u8sDummy) Bytes() []byte        { return []byte(a) }
func (a bsDummy) Bools() []bool         { return []bool(a) }

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

/* Map */

func (a f64sDummy) Map(fn interface{}) error {
	if f, ok := fn.(func(float64) float64); ok {
		for i, v := range a {
			a[i] = f(v)
		}
		return nil
	}
	return errors.Errorf(extractionFail, "func(x float64)float64", fn)
}

func (a f32sDummy) Map(fn interface{}) error {
	if f, ok := fn.(func(float32) float32); ok {
		for i, v := range a {
			a[i] = f(v)
		}
		return nil
	}
	return errors.Errorf(extractionFail, "func(x float32)float32", fn)
}

func (a intsDummy) Map(fn interface{}) error {
	if f, ok := fn.(func(int) int); ok {
		for i, v := range a {
			a[i] = f(v)
		}
		return nil
	}
	return errors.Errorf(extractionFail, "func(x int)int", fn)
}

func (a i64sDummy) Map(fn interface{}) error {
	if f, ok := fn.(func(int64) int64); ok {
		for i, v := range a {
			a[i] = f(v)
		}
		return nil
	}
	return errors.Errorf(extractionFail, "func(x int64)int64", fn)
}

func (a i32sDummy) Map(fn interface{}) error {
	if f, ok := fn.(func(int32) int32); ok {
		for i, v := range a {
			a[i] = f(v)
		}
		return nil
	}
	return errors.Errorf(extractionFail, "func(x int32)int32", fn)
}

func (a u8sDummy) Map(fn interface{}) error {
	if f, ok := fn.(func(byte) byte); ok {
		for i, v := range a {
			a[i] = f(v)
		}
		return nil
	}
	return errors.Errorf(extractionFail, "func(x byte)byte", fn)
}

func (a bsDummy) Map(fn interface{}) error {
	if f, ok := fn.(func(bool) bool); ok {
		for i, v := range a {
			a[i] = f(v)
		}
		return nil
	}
	return errors.Errorf(extractionFail, "func(x bool)bool", fn)
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
	case []float64:
		return copy(a, b), nil
	case Float64ser:
		return copy(a, b.Float64s()), nil
	}

	return 0, errors.Errorf("Cannot copy from %T", other)
}

func (a f32sDummy) CopyFrom(other interface{}) (int, error) {
	switch b := other.(type) {
	case []float32:
		return copy(a, b), nil
	case Float32ser:
		return copy(a, b.Float32s()), nil
	}

	return 0, errors.Errorf("Cannot copy from %T", other)
}

func (a intsDummy) CopyFrom(other interface{}) (int, error) {
	switch b := other.(type) {
	case []int:
		return copy(a, b), nil
	case Intser:
		return copy(a, b.Ints()), nil
	}

	return 0, errors.Errorf("Cannot copy from %T", other)
}

func (a i64sDummy) CopyFrom(other interface{}) (int, error) {
	switch b := other.(type) {
	case []int64:
		return copy(a, b), nil
	case Int64ser:
		return copy(a, b.Int64s()), nil
	}

	return 0, errors.Errorf("Cannot copy from %T", other)
}

func (a i32sDummy) CopyFrom(other interface{}) (int, error) {
	switch b := other.(type) {
	case []int32:
		return copy(a, b), nil
	case Int32ser:
		return copy(a, b.Int32s()), nil
	}

	return 0, errors.Errorf("Cannot copy from %T", other)
}

func (a u8sDummy) CopyFrom(other interface{}) (int, error) {
	switch b := other.(type) {
	case []byte:
		return copy(a, b), nil
	case Byteser:
		return copy(a, b.Bytes()), nil
	}

	return 0, errors.Errorf("Cannot copy from %T", other)
}

func (a bsDummy) CopyFrom(other interface{}) (int, error) {
	switch b := other.(type) {
	case []bool:
		return copy(a, b), nil
	case Boolser:
		return copy(a, b.Bools()), nil
	}

	return 0, errors.Errorf("Cannot copy from %T", other)
}

/* Transpose Specialization */

func (a f64sDummy) Transpose(oldShape, oldStrides, axes, newStrides []int) {
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

func (a f32sDummy) Transpose(oldShape, oldStrides, axes, newStrides []int) {
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

func (a intsDummy) Transpose(oldShape, oldStrides, axes, newStrides []int) {
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

func (a i64sDummy) Transpose(oldShape, oldStrides, axes, newStrides []int) {
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

func (a i32sDummy) Transpose(oldShape, oldStrides, axes, newStrides []int) {
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

func (a u8sDummy) Transpose(oldShape, oldStrides, axes, newStrides []int) {
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

func (a bsDummy) Transpose(oldShape, oldStrides, axes, newStrides []int) {
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

func (a f64sDummy) MapIncr(fn interface{}) error {
	if f, ok := fn.(func(float64) float64); ok {
		for i, v := range a {
			a[i] += f(v)
		}
		return nil
	}
	return errors.Errorf(extractionFail, "func(x float64)float64", fn)
}

func (a f32sDummy) MapIncr(fn interface{}) error {
	if f, ok := fn.(func(float32) float32); ok {
		for i, v := range a {
			a[i] += f(v)
		}
		return nil
	}
	return errors.Errorf(extractionFail, "func(x float32)float32", fn)
}

func (a intsDummy) MapIncr(fn interface{}) error {
	if f, ok := fn.(func(int) int); ok {
		for i, v := range a {
			a[i] += f(v)
		}
		return nil
	}
	return errors.Errorf(extractionFail, "func(x int)int", fn)
}

func (a i64sDummy) MapIncr(fn interface{}) error {
	if f, ok := fn.(func(int64) int64); ok {
		for i, v := range a {
			a[i] += f(v)
		}
		return nil
	}
	return errors.Errorf(extractionFail, "func(x int64)int64", fn)
}

func (a i32sDummy) MapIncr(fn interface{}) error {
	if f, ok := fn.(func(int32) int32); ok {
		for i, v := range a {
			a[i] += f(v)
		}
		return nil
	}
	return errors.Errorf(extractionFail, "func(x int32)int32", fn)
}

func (a u8sDummy) MapIncr(fn interface{}) error {
	if f, ok := fn.(func(byte) byte); ok {
		for i, v := range a {
			a[i] += f(v)
		}
		return nil
	}
	return errors.Errorf(extractionFail, "func(x byte)byte", fn)
}

/* IterMapper specialization */

func (a f64sDummy) IterMap(other Array, it, ot *FlatIterator, fn interface{}, incr bool) (err error) {
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

		}
		return nil
	case other == nil && ot != nil:
		// error - stupid
		return errors.Errorf("Meaningless state - other is nil, ot is not")
	}
	return
}

func (a f32sDummy) IterMap(other Array, it, ot *FlatIterator, fn interface{}, incr bool) (err error) {
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

		}
		return nil
	case other == nil && ot != nil:
		// error - stupid
		return errors.Errorf("Meaningless state - other is nil, ot is not")
	}
	return
}

func (a intsDummy) IterMap(other Array, it, ot *FlatIterator, fn interface{}, incr bool) (err error) {
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

		}
		return nil
	case other == nil && ot != nil:
		// error - stupid
		return errors.Errorf("Meaningless state - other is nil, ot is not")
	}
	return
}

func (a i64sDummy) IterMap(other Array, it, ot *FlatIterator, fn interface{}, incr bool) (err error) {
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

		}
		return nil
	case other == nil && ot != nil:
		// error - stupid
		return errors.Errorf("Meaningless state - other is nil, ot is not")
	}
	return
}

func (a i32sDummy) IterMap(other Array, it, ot *FlatIterator, fn interface{}, incr bool) (err error) {
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

		}
		return nil
	case other == nil && ot != nil:
		// error - stupid
		return errors.Errorf("Meaningless state - other is nil, ot is not")
	}
	return
}

func (a u8sDummy) IterMap(other Array, it, ot *FlatIterator, fn interface{}, incr bool) (err error) {
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

		}
		return nil
	case other == nil && ot != nil:
		// error - stupid
		return errors.Errorf("Meaningless state - other is nil, ot is not")
	}
	return
}

func (a bsDummy) IterMap(other Array, it, ot *FlatIterator, fn interface{}, incr bool) (err error) {
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

		}
		return nil
	case other == nil && ot != nil:
		// error - stupid
		return errors.Errorf("Meaningless state - other is nil, ot is not")
	}
	return
}

/* Add */

func (a f64sDummy) Add(other Number) error {
	b, err := getFloat64s(other)
	if err != nil {
		return errors.Wrapf(err, opFail, "Add")
	}

	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
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
		return errors.Errorf(lenMismatch, len(a), len(b))
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
		return errors.Errorf(lenMismatch, len(a), len(b))
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
		return errors.Errorf(lenMismatch, len(a), len(b))
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
		return errors.Errorf(lenMismatch, len(a), len(b))
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
		return errors.Errorf(lenMismatch, len(a), len(b))
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
		return errors.Errorf(lenMismatch, len(a), len(b))
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
		return errors.Errorf(lenMismatch, len(a), len(b))
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
		return errors.Errorf(lenMismatch, len(a), len(b))
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
		return errors.Errorf(lenMismatch, len(a), len(b))
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
		return errors.Errorf(lenMismatch, len(a), len(b))
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
		return errors.Errorf(lenMismatch, len(a), len(b))
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
		return errors.Errorf(lenMismatch, len(a), len(b))
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
		return errors.Errorf(lenMismatch, len(a), len(b))
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
		return errors.Errorf(lenMismatch, len(a), len(b))
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
		return errors.Errorf(lenMismatch, len(a), len(b))
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
		return errors.Errorf(lenMismatch, len(a), len(b))
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
		return errors.Errorf(lenMismatch, len(a), len(b))
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
		return errors.Errorf(lenMismatch, len(a), len(b))
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
		return errors.Errorf(lenMismatch, len(a), len(b))
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

func (a i64sDummy) Div(other Number) error {
	b, err := getInt64s(other)
	if err != nil {
		return errors.Wrapf(err, opFail, "Div")
	}

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

func (a i32sDummy) Div(other Number) error {
	b, err := getInt32s(other)
	if err != nil {
		return errors.Wrapf(err, opFail, "Div")
	}

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

func (a u8sDummy) Div(other Number) error {
	b, err := getBytes(other)
	if err != nil {
		return errors.Wrapf(err, opFail, "Div")
	}

	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
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

func (a f64sDummy) Pow(other Number) error {
	b, err := getFloat64s(other)
	if err != nil {
		return errors.Wrapf(err, opFail, "Pow")
	}

	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
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
		return errors.Errorf(lenMismatch, len(a), len(b))
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
		return errors.Errorf(lenMismatch, len(a), len(b))
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
		return errors.Errorf(lenMismatch, len(a), len(b))
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
		return errors.Errorf(lenMismatch, len(a), len(b))
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
		return errors.Errorf(lenMismatch, len(a), len(b))
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

/* Add */

func (a f64sDummy) IncrAdd(other, incrArr Number) (err error) {
	var b, incr []float64
	if b, err = getFloat64s(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrAdd")
	}

	if incr, err = getFloat64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrAdd")
	}

	if len(b) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	vecf64.IncrAdd([]float64(a), b, incr)
	return nil
}

func (a f32sDummy) IncrAdd(other, incrArr Number) (err error) {
	var b, incr []float32
	if b, err = getFloat32s(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrAdd")
	}

	if incr, err = getFloat32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrAdd")
	}

	if len(b) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	vecf32.IncrAdd([]float32(a), b, incr)
	return nil
}

func (a intsDummy) IncrAdd(other, incrArr Number) (err error) {
	var b, incr []int
	if b, err = getInts(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrAdd")
	}

	if incr, err = getInts(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrAdd")
	}

	if len(b) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	for i, v := range b {
		incr[i] += a[i] + v
	}

	return nil
}

func (a i64sDummy) IncrAdd(other, incrArr Number) (err error) {
	var b, incr []int64
	if b, err = getInt64s(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrAdd")
	}

	if incr, err = getInt64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrAdd")
	}

	if len(b) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	for i, v := range b {
		incr[i] += a[i] + v
	}

	return nil
}

func (a i32sDummy) IncrAdd(other, incrArr Number) (err error) {
	var b, incr []int32
	if b, err = getInt32s(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrAdd")
	}

	if incr, err = getInt32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrAdd")
	}

	if len(b) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	for i, v := range b {
		incr[i] += a[i] + v
	}

	return nil
}

func (a u8sDummy) IncrAdd(other, incrArr Number) (err error) {
	var b, incr []byte
	if b, err = getBytes(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrAdd")
	}

	if incr, err = getBytes(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrAdd")
	}

	if len(b) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	for i, v := range b {
		incr[i] += a[i] + v
	}

	return nil
}

/* Sub */

func (a f64sDummy) IncrSub(other, incrArr Number) (err error) {
	var b, incr []float64
	if b, err = getFloat64s(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrSub")
	}

	if incr, err = getFloat64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrSub")
	}

	if len(b) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	vecf64.IncrSub([]float64(a), b, incr)
	return nil
}

func (a f32sDummy) IncrSub(other, incrArr Number) (err error) {
	var b, incr []float32
	if b, err = getFloat32s(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrSub")
	}

	if incr, err = getFloat32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrSub")
	}

	if len(b) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	vecf32.IncrSub([]float32(a), b, incr)
	return nil
}

func (a intsDummy) IncrSub(other, incrArr Number) (err error) {
	var b, incr []int
	if b, err = getInts(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrSub")
	}

	if incr, err = getInts(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrSub")
	}

	if len(b) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	for i, v := range b {
		incr[i] += a[i] - v
	}

	return nil
}

func (a i64sDummy) IncrSub(other, incrArr Number) (err error) {
	var b, incr []int64
	if b, err = getInt64s(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrSub")
	}

	if incr, err = getInt64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrSub")
	}

	if len(b) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	for i, v := range b {
		incr[i] += a[i] - v
	}

	return nil
}

func (a i32sDummy) IncrSub(other, incrArr Number) (err error) {
	var b, incr []int32
	if b, err = getInt32s(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrSub")
	}

	if incr, err = getInt32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrSub")
	}

	if len(b) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	for i, v := range b {
		incr[i] += a[i] - v
	}

	return nil
}

func (a u8sDummy) IncrSub(other, incrArr Number) (err error) {
	var b, incr []byte
	if b, err = getBytes(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrSub")
	}

	if incr, err = getBytes(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrSub")
	}

	if len(b) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	for i, v := range b {
		incr[i] += a[i] - v
	}

	return nil
}

/* Mul */

func (a f64sDummy) IncrMul(other, incrArr Number) (err error) {
	var b, incr []float64
	if b, err = getFloat64s(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrMul")
	}

	if incr, err = getFloat64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrMul")
	}

	if len(b) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	vecf64.IncrMul([]float64(a), b, incr)
	return nil
}

func (a f32sDummy) IncrMul(other, incrArr Number) (err error) {
	var b, incr []float32
	if b, err = getFloat32s(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrMul")
	}

	if incr, err = getFloat32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrMul")
	}

	if len(b) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	vecf32.IncrMul([]float32(a), b, incr)
	return nil
}

func (a intsDummy) IncrMul(other, incrArr Number) (err error) {
	var b, incr []int
	if b, err = getInts(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrMul")
	}

	if incr, err = getInts(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrMul")
	}

	if len(b) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	for i, v := range b {
		incr[i] += a[i] * v
	}

	return nil
}

func (a i64sDummy) IncrMul(other, incrArr Number) (err error) {
	var b, incr []int64
	if b, err = getInt64s(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrMul")
	}

	if incr, err = getInt64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrMul")
	}

	if len(b) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	for i, v := range b {
		incr[i] += a[i] * v
	}

	return nil
}

func (a i32sDummy) IncrMul(other, incrArr Number) (err error) {
	var b, incr []int32
	if b, err = getInt32s(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrMul")
	}

	if incr, err = getInt32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrMul")
	}

	if len(b) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	for i, v := range b {
		incr[i] += a[i] * v
	}

	return nil
}

func (a u8sDummy) IncrMul(other, incrArr Number) (err error) {
	var b, incr []byte
	if b, err = getBytes(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrMul")
	}

	if incr, err = getBytes(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrMul")
	}

	if len(b) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	for i, v := range b {
		incr[i] += a[i] * v
	}

	return nil
}

/* Div */

func (a f64sDummy) IncrDiv(other, incrArr Number) (err error) {
	var b, incr []float64
	if b, err = getFloat64s(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrDiv")
	}

	if incr, err = getFloat64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrDiv")
	}

	if len(b) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	vecf64.IncrDiv([]float64(a), b, incr)
	return nil
}

func (a f32sDummy) IncrDiv(other, incrArr Number) (err error) {
	var b, incr []float32
	if b, err = getFloat32s(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrDiv")
	}

	if incr, err = getFloat32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrDiv")
	}

	if len(b) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	vecf32.IncrDiv([]float32(a), b, incr)
	return nil
}

func (a intsDummy) IncrDiv(other, incrArr Number) (err error) {
	var b, incr []int
	if b, err = getInts(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrDiv")
	}

	if incr, err = getInts(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrDiv")
	}

	if len(b) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
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

func (a i64sDummy) IncrDiv(other, incrArr Number) (err error) {
	var b, incr []int64
	if b, err = getInt64s(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrDiv")
	}

	if incr, err = getInt64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrDiv")
	}

	if len(b) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
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

func (a i32sDummy) IncrDiv(other, incrArr Number) (err error) {
	var b, incr []int32
	if b, err = getInt32s(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrDiv")
	}

	if incr, err = getInt32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrDiv")
	}

	if len(b) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
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

func (a u8sDummy) IncrDiv(other, incrArr Number) (err error) {
	var b, incr []byte
	if b, err = getBytes(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrDiv")
	}

	if incr, err = getBytes(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrDiv")
	}

	if len(b) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
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

func (a f64sDummy) IncrPow(other, incrArr Number) (err error) {
	var b, incr []float64
	if b, err = getFloat64s(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrPow")
	}

	if incr, err = getFloat64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrPow")
	}

	if len(b) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	vecf64.IncrPow([]float64(a), b, incr)
	return nil
}

func (a f32sDummy) IncrPow(other, incrArr Number) (err error) {
	var b, incr []float32
	if b, err = getFloat32s(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrPow")
	}

	if incr, err = getFloat32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrPow")
	}

	if len(b) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	vecf32.IncrPow([]float32(a), b, incr)
	return nil
}

func (a intsDummy) IncrPow(other, incrArr Number) (err error) {
	var b, incr []int
	if b, err = getInts(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrPow")
	}

	if incr, err = getInts(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrPow")
	}

	if len(b) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	for i, v := range b {
		incr[i] += int(math.Pow(float64(a[i]), float64(v)))
	}

	return nil
}

func (a i64sDummy) IncrPow(other, incrArr Number) (err error) {
	var b, incr []int64
	if b, err = getInt64s(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrPow")
	}

	if incr, err = getInt64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrPow")
	}

	if len(b) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	for i, v := range b {
		incr[i] += int64(math.Pow(float64(a[i]), float64(v)))
	}

	return nil
}

func (a i32sDummy) IncrPow(other, incrArr Number) (err error) {
	var b, incr []int32
	if b, err = getInt32s(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrPow")
	}

	if incr, err = getInt32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrPow")
	}

	if len(b) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	for i, v := range b {
		incr[i] += int32(math.Pow(float64(a[i]), float64(v)))
	}

	return nil
}

func (a u8sDummy) IncrPow(other, incrArr Number) (err error) {
	var b, incr []byte
	if b, err = getBytes(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrPow")
	}

	if incr, err = getBytes(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrPow")
	}

	if len(b) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	for i, v := range b {
		incr[i] += byte(math.Pow(float64(a[i]), float64(v)))
	}

	return nil
}

/* Trans */

func (a f64sDummy) IncrTrans(other interface{}, incrArr Number) (err error) {
	var b float64
	if b, err = getFloat64(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrTrans")
	}

	var incr []float64
	if incr, err = getFloat64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrTrans")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	vecf64.IncrTrans([]float64(a), b, incr)
	return nil
}

func (a f32sDummy) IncrTrans(other interface{}, incrArr Number) (err error) {
	var b float32
	if b, err = getFloat32(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrTrans")
	}

	var incr []float32
	if incr, err = getFloat32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrTrans")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	vecf32.IncrTrans([]float32(a), b, incr)
	return nil
}

func (a intsDummy) IncrTrans(other interface{}, incrArr Number) (err error) {
	var b int
	if b, err = getInt(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrTrans")
	}

	var incr []int
	if incr, err = getInts(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrTrans")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += v + b
	}
	return nil
}

func (a i64sDummy) IncrTrans(other interface{}, incrArr Number) (err error) {
	var b int64
	if b, err = getInt64(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrTrans")
	}

	var incr []int64
	if incr, err = getInt64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrTrans")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += v + b
	}
	return nil
}

func (a i32sDummy) IncrTrans(other interface{}, incrArr Number) (err error) {
	var b int32
	if b, err = getInt32(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrTrans")
	}

	var incr []int32
	if incr, err = getInt32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrTrans")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += v + b
	}
	return nil
}

func (a u8sDummy) IncrTrans(other interface{}, incrArr Number) (err error) {
	var b byte
	if b, err = getByte(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrTrans")
	}

	var incr []byte
	if incr, err = getBytes(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrTrans")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += v + b
	}
	return nil
}

/* TransInv */

func (a f64sDummy) IncrTransInv(other interface{}, incrArr Number) (err error) {
	var b float64
	if b, err = getFloat64(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInv")
	}

	var incr []float64
	if incr, err = getFloat64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInv")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	vecf64.IncrTransInv([]float64(a), b, incr)
	return nil
}

func (a f32sDummy) IncrTransInv(other interface{}, incrArr Number) (err error) {
	var b float32
	if b, err = getFloat32(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInv")
	}

	var incr []float32
	if incr, err = getFloat32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInv")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	vecf32.IncrTransInv([]float32(a), b, incr)
	return nil
}

func (a intsDummy) IncrTransInv(other interface{}, incrArr Number) (err error) {
	var b int
	if b, err = getInt(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInv")
	}

	var incr []int
	if incr, err = getInts(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInv")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += v - b
	}
	return nil
}

func (a i64sDummy) IncrTransInv(other interface{}, incrArr Number) (err error) {
	var b int64
	if b, err = getInt64(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInv")
	}

	var incr []int64
	if incr, err = getInt64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInv")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += v - b
	}
	return nil
}

func (a i32sDummy) IncrTransInv(other interface{}, incrArr Number) (err error) {
	var b int32
	if b, err = getInt32(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInv")
	}

	var incr []int32
	if incr, err = getInt32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInv")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += v - b
	}
	return nil
}

func (a u8sDummy) IncrTransInv(other interface{}, incrArr Number) (err error) {
	var b byte
	if b, err = getByte(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInv")
	}

	var incr []byte
	if incr, err = getBytes(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInv")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += v - b
	}
	return nil
}

/* TransInvR */

func (a f64sDummy) IncrTransInvR(other interface{}, incrArr Number) (err error) {
	var b float64
	if b, err = getFloat64(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInvR")
	}

	var incr []float64
	if incr, err = getFloat64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInvR")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	vecf64.IncrTransInvR([]float64(a), b, incr)
	return nil
}

func (a f32sDummy) IncrTransInvR(other interface{}, incrArr Number) (err error) {
	var b float32
	if b, err = getFloat32(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInvR")
	}

	var incr []float32
	if incr, err = getFloat32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInvR")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	vecf32.IncrTransInvR([]float32(a), b, incr)
	return nil
}

func (a intsDummy) IncrTransInvR(other interface{}, incrArr Number) (err error) {
	var b int
	if b, err = getInt(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInvR")
	}

	var incr []int
	if incr, err = getInts(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInvR")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += b - v
	}
	return nil
}

func (a i64sDummy) IncrTransInvR(other interface{}, incrArr Number) (err error) {
	var b int64
	if b, err = getInt64(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInvR")
	}

	var incr []int64
	if incr, err = getInt64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInvR")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += b - v
	}
	return nil
}

func (a i32sDummy) IncrTransInvR(other interface{}, incrArr Number) (err error) {
	var b int32
	if b, err = getInt32(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInvR")
	}

	var incr []int32
	if incr, err = getInt32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInvR")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += b - v
	}
	return nil
}

func (a u8sDummy) IncrTransInvR(other interface{}, incrArr Number) (err error) {
	var b byte
	if b, err = getByte(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInvR")
	}

	var incr []byte
	if incr, err = getBytes(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrTransInvR")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += b - v
	}
	return nil
}

/* Scale */

func (a f64sDummy) IncrScale(other interface{}, incrArr Number) (err error) {
	var b float64
	if b, err = getFloat64(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrScale")
	}

	var incr []float64
	if incr, err = getFloat64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrScale")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	vecf64.IncrScale([]float64(a), b, incr)
	return nil
}

func (a f32sDummy) IncrScale(other interface{}, incrArr Number) (err error) {
	var b float32
	if b, err = getFloat32(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrScale")
	}

	var incr []float32
	if incr, err = getFloat32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrScale")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	vecf32.IncrScale([]float32(a), b, incr)
	return nil
}

func (a intsDummy) IncrScale(other interface{}, incrArr Number) (err error) {
	var b int
	if b, err = getInt(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrScale")
	}

	var incr []int
	if incr, err = getInts(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrScale")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += v * b
	}
	return nil
}

func (a i64sDummy) IncrScale(other interface{}, incrArr Number) (err error) {
	var b int64
	if b, err = getInt64(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrScale")
	}

	var incr []int64
	if incr, err = getInt64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrScale")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += v * b
	}
	return nil
}

func (a i32sDummy) IncrScale(other interface{}, incrArr Number) (err error) {
	var b int32
	if b, err = getInt32(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrScale")
	}

	var incr []int32
	if incr, err = getInt32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrScale")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += v * b
	}
	return nil
}

func (a u8sDummy) IncrScale(other interface{}, incrArr Number) (err error) {
	var b byte
	if b, err = getByte(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrScale")
	}

	var incr []byte
	if incr, err = getBytes(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrScale")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += v * b
	}
	return nil
}

/* ScaleInv */

func (a f64sDummy) IncrScaleInv(other interface{}, incrArr Number) (err error) {
	var b float64
	if b, err = getFloat64(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInv")
	}

	var incr []float64
	if incr, err = getFloat64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInv")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	vecf64.IncrScaleInv([]float64(a), b, incr)
	return nil
}

func (a f32sDummy) IncrScaleInv(other interface{}, incrArr Number) (err error) {
	var b float32
	if b, err = getFloat32(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInv")
	}

	var incr []float32
	if incr, err = getFloat32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInv")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	vecf32.IncrScaleInv([]float32(a), b, incr)
	return nil
}

func (a intsDummy) IncrScaleInv(other interface{}, incrArr Number) (err error) {
	var b int
	if b, err = getInt(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInv")
	}

	var incr []int
	if incr, err = getInts(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInv")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
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

func (a i64sDummy) IncrScaleInv(other interface{}, incrArr Number) (err error) {
	var b int64
	if b, err = getInt64(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInv")
	}

	var incr []int64
	if incr, err = getInt64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInv")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
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

func (a i32sDummy) IncrScaleInv(other interface{}, incrArr Number) (err error) {
	var b int32
	if b, err = getInt32(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInv")
	}

	var incr []int32
	if incr, err = getInt32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInv")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
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

func (a u8sDummy) IncrScaleInv(other interface{}, incrArr Number) (err error) {
	var b byte
	if b, err = getByte(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInv")
	}

	var incr []byte
	if incr, err = getBytes(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInv")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
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

func (a f64sDummy) IncrScaleInvR(other interface{}, incrArr Number) (err error) {
	var b float64
	if b, err = getFloat64(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInvR")
	}

	var incr []float64
	if incr, err = getFloat64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInvR")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	vecf64.IncrScaleInvR([]float64(a), b, incr)
	return nil
}

func (a f32sDummy) IncrScaleInvR(other interface{}, incrArr Number) (err error) {
	var b float32
	if b, err = getFloat32(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInvR")
	}

	var incr []float32
	if incr, err = getFloat32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInvR")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	vecf32.IncrScaleInvR([]float32(a), b, incr)
	return nil
}

func (a intsDummy) IncrScaleInvR(other interface{}, incrArr Number) (err error) {
	var b int
	if b, err = getInt(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInvR")
	}

	var incr []int
	if incr, err = getInts(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInvR")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
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

func (a i64sDummy) IncrScaleInvR(other interface{}, incrArr Number) (err error) {
	var b int64
	if b, err = getInt64(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInvR")
	}

	var incr []int64
	if incr, err = getInt64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInvR")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
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

func (a i32sDummy) IncrScaleInvR(other interface{}, incrArr Number) (err error) {
	var b int32
	if b, err = getInt32(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInvR")
	}

	var incr []int32
	if incr, err = getInt32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInvR")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
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

func (a u8sDummy) IncrScaleInvR(other interface{}, incrArr Number) (err error) {
	var b byte
	if b, err = getByte(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInvR")
	}

	var incr []byte
	if incr, err = getBytes(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrScaleInvR")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
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

func (a f64sDummy) IncrPowOf(other interface{}, incrArr Number) (err error) {
	var b float64
	if b, err = getFloat64(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOf")
	}

	var incr []float64
	if incr, err = getFloat64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOf")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	vecf64.IncrPowOf([]float64(a), b, incr)
	return nil
}

func (a f32sDummy) IncrPowOf(other interface{}, incrArr Number) (err error) {
	var b float32
	if b, err = getFloat32(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOf")
	}

	var incr []float32
	if incr, err = getFloat32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOf")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	vecf32.IncrPowOf([]float32(a), b, incr)
	return nil
}

func (a intsDummy) IncrPowOf(other interface{}, incrArr Number) (err error) {
	var b int
	if b, err = getInt(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOf")
	}

	var incr []int
	if incr, err = getInts(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOf")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += int(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func (a i64sDummy) IncrPowOf(other interface{}, incrArr Number) (err error) {
	var b int64
	if b, err = getInt64(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOf")
	}

	var incr []int64
	if incr, err = getInt64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOf")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += int64(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func (a i32sDummy) IncrPowOf(other interface{}, incrArr Number) (err error) {
	var b int32
	if b, err = getInt32(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOf")
	}

	var incr []int32
	if incr, err = getInt32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOf")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += int32(math.Pow(float64(v), float64(b)))
	}
	return nil
}

func (a u8sDummy) IncrPowOf(other interface{}, incrArr Number) (err error) {
	var b byte
	if b, err = getByte(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOf")
	}

	var incr []byte
	if incr, err = getBytes(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOf")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += byte(math.Pow(float64(v), float64(b)))
	}
	return nil
}

/* PowOfR */

func (a f64sDummy) IncrPowOfR(other interface{}, incrArr Number) (err error) {
	var b float64
	if b, err = getFloat64(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOfR")
	}

	var incr []float64
	if incr, err = getFloat64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOfR")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	vecf64.IncrPowOfR([]float64(a), b, incr)
	return nil
}

func (a f32sDummy) IncrPowOfR(other interface{}, incrArr Number) (err error) {
	var b float32
	if b, err = getFloat32(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOfR")
	}

	var incr []float32
	if incr, err = getFloat32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOfR")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	vecf32.IncrPowOfR([]float32(a), b, incr)
	return nil
}

func (a intsDummy) IncrPowOfR(other interface{}, incrArr Number) (err error) {
	var b int
	if b, err = getInt(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOfR")
	}

	var incr []int
	if incr, err = getInts(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOfR")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += int(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func (a i64sDummy) IncrPowOfR(other interface{}, incrArr Number) (err error) {
	var b int64
	if b, err = getInt64(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOfR")
	}

	var incr []int64
	if incr, err = getInt64s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOfR")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += int64(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func (a i32sDummy) IncrPowOfR(other interface{}, incrArr Number) (err error) {
	var b int32
	if b, err = getInt32(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOfR")
	}

	var incr []int32
	if incr, err = getInt32s(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOfR")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += int32(math.Pow(float64(b), float64(v)))
	}
	return nil
}

func (a u8sDummy) IncrPowOfR(other interface{}, incrArr Number) (err error) {
	var b byte
	if b, err = getByte(other); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOfR")
	}

	var incr []byte
	if incr, err = getBytes(incrArr); err != nil {
		return errors.Wrapf(err, opFail, "IncrPowOfR")
	}

	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}

	for i, v := range a {
		incr[i] += byte(math.Pow(float64(b), float64(v)))
	}
	return nil
}

/* ElEq */

func (a f64sDummy) ElEq(other ElemEq, same bool) (Array, error) {
	var compat Float64ser
	var ok bool
	if compat, ok = other.(Float64ser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Float64s()

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

func (a f32sDummy) ElEq(other ElemEq, same bool) (Array, error) {
	var compat Float32ser
	var ok bool
	if compat, ok = other.(Float32ser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Float32s()

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

func (a intsDummy) ElEq(other ElemEq, same bool) (Array, error) {
	var compat Intser
	var ok bool
	if compat, ok = other.(Intser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Ints()

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

func (a i64sDummy) ElEq(other ElemEq, same bool) (Array, error) {
	var compat Int64ser
	var ok bool
	if compat, ok = other.(Int64ser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Int64s()

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

func (a i32sDummy) ElEq(other ElemEq, same bool) (Array, error) {
	var compat Int32ser
	var ok bool
	if compat, ok = other.(Int32ser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Int32s()

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

func (a u8sDummy) ElEq(other ElemEq, same bool) (Array, error) {
	var compat Byteser
	var ok bool
	if compat, ok = other.(Byteser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Bytes()

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

func (a bsDummy) ElEq(other ElemEq, same bool) (Array, error) {
	var compat Boolser
	var ok bool
	if compat, ok = other.(Boolser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Bools()

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

func (a f64sDummy) Gt(other ElemOrd, same bool) (Array, error) {
	var compat Float64ser
	var ok bool
	if compat, ok = other.(Float64ser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Float64s()

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

func (a f32sDummy) Gt(other ElemOrd, same bool) (Array, error) {
	var compat Float32ser
	var ok bool
	if compat, ok = other.(Float32ser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Float32s()

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

func (a intsDummy) Gt(other ElemOrd, same bool) (Array, error) {
	var compat Intser
	var ok bool
	if compat, ok = other.(Intser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Ints()

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

func (a i64sDummy) Gt(other ElemOrd, same bool) (Array, error) {
	var compat Int64ser
	var ok bool
	if compat, ok = other.(Int64ser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Int64s()

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

func (a i32sDummy) Gt(other ElemOrd, same bool) (Array, error) {
	var compat Int32ser
	var ok bool
	if compat, ok = other.(Int32ser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Int32s()

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

func (a u8sDummy) Gt(other ElemOrd, same bool) (Array, error) {
	var compat Byteser
	var ok bool
	if compat, ok = other.(Byteser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Bytes()

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

func (a f64sDummy) Gte(other ElemOrd, same bool) (Array, error) {
	var compat Float64ser
	var ok bool
	if compat, ok = other.(Float64ser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Float64s()

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

func (a f32sDummy) Gte(other ElemOrd, same bool) (Array, error) {
	var compat Float32ser
	var ok bool
	if compat, ok = other.(Float32ser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Float32s()

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

func (a intsDummy) Gte(other ElemOrd, same bool) (Array, error) {
	var compat Intser
	var ok bool
	if compat, ok = other.(Intser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Ints()

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

func (a i64sDummy) Gte(other ElemOrd, same bool) (Array, error) {
	var compat Int64ser
	var ok bool
	if compat, ok = other.(Int64ser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Int64s()

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

func (a i32sDummy) Gte(other ElemOrd, same bool) (Array, error) {
	var compat Int32ser
	var ok bool
	if compat, ok = other.(Int32ser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Int32s()

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

func (a u8sDummy) Gte(other ElemOrd, same bool) (Array, error) {
	var compat Byteser
	var ok bool
	if compat, ok = other.(Byteser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Bytes()

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

func (a f64sDummy) Lt(other ElemOrd, same bool) (Array, error) {
	var compat Float64ser
	var ok bool
	if compat, ok = other.(Float64ser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Float64s()

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

func (a f32sDummy) Lt(other ElemOrd, same bool) (Array, error) {
	var compat Float32ser
	var ok bool
	if compat, ok = other.(Float32ser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Float32s()

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

func (a intsDummy) Lt(other ElemOrd, same bool) (Array, error) {
	var compat Intser
	var ok bool
	if compat, ok = other.(Intser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Ints()

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

func (a i64sDummy) Lt(other ElemOrd, same bool) (Array, error) {
	var compat Int64ser
	var ok bool
	if compat, ok = other.(Int64ser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Int64s()

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

func (a i32sDummy) Lt(other ElemOrd, same bool) (Array, error) {
	var compat Int32ser
	var ok bool
	if compat, ok = other.(Int32ser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Int32s()

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

func (a u8sDummy) Lt(other ElemOrd, same bool) (Array, error) {
	var compat Byteser
	var ok bool
	if compat, ok = other.(Byteser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Bytes()

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

func (a f64sDummy) Lte(other ElemOrd, same bool) (Array, error) {
	var compat Float64ser
	var ok bool
	if compat, ok = other.(Float64ser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Float64s()

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

func (a f32sDummy) Lte(other ElemOrd, same bool) (Array, error) {
	var compat Float32ser
	var ok bool
	if compat, ok = other.(Float32ser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Float32s()

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

func (a intsDummy) Lte(other ElemOrd, same bool) (Array, error) {
	var compat Intser
	var ok bool
	if compat, ok = other.(Intser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Ints()

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

func (a i64sDummy) Lte(other ElemOrd, same bool) (Array, error) {
	var compat Int64ser
	var ok bool
	if compat, ok = other.(Int64ser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Int64s()

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

func (a i32sDummy) Lte(other ElemOrd, same bool) (Array, error) {
	var compat Int32ser
	var ok bool
	if compat, ok = other.(Int32ser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Int32s()

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

func (a u8sDummy) Lte(other ElemOrd, same bool) (Array, error) {
	var compat Byteser
	var ok bool
	if compat, ok = other.(Byteser); !ok {
		return nil, errors.Errorf(typeMismatch, a, other)
	}
	b := compat.Bytes()

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

func (a f64sDummy) Slice(start, end int) (Array, error) {
	if end >= len(a) || start < 0 {
		return nil, errors.Errorf(sliceIndexOOB, start, end, len(a))
	}

	return a[start:end], nil
}

func (a f64sDummy) Dtype() Dtype { return Float64 }

func (a f64sDummy) HasNaN() bool { return false }
func (a f64sDummy) HasInf() bool { return false }
func (a f32sDummy) Slice(start, end int) (Array, error) {
	if end >= len(a) || start < 0 {
		return nil, errors.Errorf(sliceIndexOOB, start, end, len(a))
	}

	return a[start:end], nil
}

func (a f32sDummy) Dtype() Dtype { return Float32 }

func (a f32sDummy) HasNaN() bool { return false }
func (a f32sDummy) HasInf() bool { return false }
func (a intsDummy) Slice(start, end int) (Array, error) {
	if end >= len(a) || start < 0 {
		return nil, errors.Errorf(sliceIndexOOB, start, end, len(a))
	}

	return a[start:end], nil
}

func (a intsDummy) Dtype() Dtype { return Int }

func (a i64sDummy) Slice(start, end int) (Array, error) {
	if end >= len(a) || start < 0 {
		return nil, errors.Errorf(sliceIndexOOB, start, end, len(a))
	}

	return a[start:end], nil
}

func (a i64sDummy) Dtype() Dtype { return Int64 }

func (a i32sDummy) Slice(start, end int) (Array, error) {
	if end >= len(a) || start < 0 {
		return nil, errors.Errorf(sliceIndexOOB, start, end, len(a))
	}

	return a[start:end], nil
}

func (a i32sDummy) Dtype() Dtype { return Int32 }

func (a u8sDummy) Slice(start, end int) (Array, error) {
	if end >= len(a) || start < 0 {
		return nil, errors.Errorf(sliceIndexOOB, start, end, len(a))
	}

	return a[start:end], nil
}

func (a u8sDummy) Dtype() Dtype { return Byte }

func (a bsDummy) Slice(start, end int) (Array, error) {
	if end >= len(a) || start < 0 {
		return nil, errors.Errorf(sliceIndexOOB, start, end, len(a))
	}

	return a[start:end], nil
}

func (a bsDummy) Dtype() Dtype { return Bool }
