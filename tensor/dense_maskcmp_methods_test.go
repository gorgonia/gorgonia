package tensor

import (
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"
)

/*
GENERATED FILE. DO NOT EDIT
*/

/* MaskedEqual */

func TestDense_MaskedEqual_I(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Int), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.ints()
	for i := range data {
		data[i] = int(i)
	}
	T.MaskedEqual(int(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(int(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(int(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(int(1), int(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(int(1), int(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(int(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedEqual_I8(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Int8), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.int8s()
	for i := range data {
		data[i] = int8(i)
	}
	T.MaskedEqual(int8(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(int8(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(int8(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(int8(1), int8(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(int8(1), int8(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(int8(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedEqual_I16(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Int16), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.int16s()
	for i := range data {
		data[i] = int16(i)
	}
	T.MaskedEqual(int16(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(int16(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(int16(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(int16(1), int16(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(int16(1), int16(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(int16(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedEqual_I32(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Int32), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.int32s()
	for i := range data {
		data[i] = int32(i)
	}
	T.MaskedEqual(int32(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(int32(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(int32(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(int32(1), int32(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(int32(1), int32(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(int32(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedEqual_I64(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Int64), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.int64s()
	for i := range data {
		data[i] = int64(i)
	}
	T.MaskedEqual(int64(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(int64(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(int64(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(int64(1), int64(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(int64(1), int64(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(int64(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedEqual_U(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Uint), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.uints()
	for i := range data {
		data[i] = uint(i)
	}
	T.MaskedEqual(uint(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(uint(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(uint(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(uint(1), uint(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(uint(1), uint(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(uint(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedEqual_U8(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Uint8), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.uint8s()
	for i := range data {
		data[i] = uint8(i)
	}
	T.MaskedEqual(uint8(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(uint8(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(uint8(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(uint8(1), uint8(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(uint8(1), uint8(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(uint8(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedEqual_U16(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Uint16), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.uint16s()
	for i := range data {
		data[i] = uint16(i)
	}
	T.MaskedEqual(uint16(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(uint16(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(uint16(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(uint16(1), uint16(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(uint16(1), uint16(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(uint16(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedEqual_U32(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Uint32), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.uint32s()
	for i := range data {
		data[i] = uint32(i)
	}
	T.MaskedEqual(uint32(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(uint32(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(uint32(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(uint32(1), uint32(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(uint32(1), uint32(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(uint32(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedEqual_U64(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Uint64), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.uint64s()
	for i := range data {
		data[i] = uint64(i)
	}
	T.MaskedEqual(uint64(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(uint64(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(uint64(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(uint64(1), uint64(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(uint64(1), uint64(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(uint64(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedEqual_F32(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Float32), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.float32s()
	for i := range data {
		data[i] = float32(i)
	}
	T.MaskedEqual(float32(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(float32(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(float32(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(float32(1), float32(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(float32(1), float32(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(float32(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedEqual_F64(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Float64), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.float64s()
	for i := range data {
		data[i] = float64(i)
	}
	T.MaskedEqual(float64(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(float64(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(float64(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(float64(1), float64(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(float64(1), float64(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(float64(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedEqual_Str(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(String), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.strings()
	for i := range data {
		data[i] = string(i)
	}
	T.MaskedEqual(string(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(string(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(string(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(string(1), string(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(string(1), string(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(string(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}

/* MaskedNotEqual */

func TestDense_MaskedNotEqual_I(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Int), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.ints()
	for i := range data {
		data[i] = int(i)
	}
	T.MaskedEqual(int(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(int(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(int(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(int(1), int(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(int(1), int(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(int(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedNotEqual_I8(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Int8), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.int8s()
	for i := range data {
		data[i] = int8(i)
	}
	T.MaskedEqual(int8(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(int8(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(int8(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(int8(1), int8(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(int8(1), int8(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(int8(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedNotEqual_I16(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Int16), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.int16s()
	for i := range data {
		data[i] = int16(i)
	}
	T.MaskedEqual(int16(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(int16(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(int16(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(int16(1), int16(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(int16(1), int16(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(int16(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedNotEqual_I32(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Int32), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.int32s()
	for i := range data {
		data[i] = int32(i)
	}
	T.MaskedEqual(int32(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(int32(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(int32(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(int32(1), int32(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(int32(1), int32(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(int32(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedNotEqual_I64(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Int64), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.int64s()
	for i := range data {
		data[i] = int64(i)
	}
	T.MaskedEqual(int64(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(int64(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(int64(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(int64(1), int64(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(int64(1), int64(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(int64(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedNotEqual_U(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Uint), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.uints()
	for i := range data {
		data[i] = uint(i)
	}
	T.MaskedEqual(uint(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(uint(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(uint(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(uint(1), uint(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(uint(1), uint(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(uint(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedNotEqual_U8(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Uint8), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.uint8s()
	for i := range data {
		data[i] = uint8(i)
	}
	T.MaskedEqual(uint8(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(uint8(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(uint8(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(uint8(1), uint8(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(uint8(1), uint8(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(uint8(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedNotEqual_U16(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Uint16), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.uint16s()
	for i := range data {
		data[i] = uint16(i)
	}
	T.MaskedEqual(uint16(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(uint16(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(uint16(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(uint16(1), uint16(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(uint16(1), uint16(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(uint16(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedNotEqual_U32(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Uint32), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.uint32s()
	for i := range data {
		data[i] = uint32(i)
	}
	T.MaskedEqual(uint32(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(uint32(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(uint32(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(uint32(1), uint32(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(uint32(1), uint32(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(uint32(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedNotEqual_U64(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Uint64), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.uint64s()
	for i := range data {
		data[i] = uint64(i)
	}
	T.MaskedEqual(uint64(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(uint64(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(uint64(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(uint64(1), uint64(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(uint64(1), uint64(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(uint64(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedNotEqual_F32(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Float32), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.float32s()
	for i := range data {
		data[i] = float32(i)
	}
	T.MaskedEqual(float32(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(float32(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(float32(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(float32(1), float32(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(float32(1), float32(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(float32(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedNotEqual_F64(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Float64), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.float64s()
	for i := range data {
		data[i] = float64(i)
	}
	T.MaskedEqual(float64(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(float64(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(float64(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(float64(1), float64(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(float64(1), float64(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(float64(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedNotEqual_Str(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(String), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.strings()
	for i := range data {
		data[i] = string(i)
	}
	T.MaskedEqual(string(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(string(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(string(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(string(1), string(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(string(1), string(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(string(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}

/* MaskedValues */

func TestDense_MaskedValues_F32(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Float32), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.float32s()
	for i := range data {
		data[i] = float32(i)
	}
	T.MaskedEqual(float32(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(float32(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(float32(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(float32(1), float32(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(float32(1), float32(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(float32(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedValues_F64(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Float64), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.float64s()
	for i := range data {
		data[i] = float64(i)
	}
	T.MaskedEqual(float64(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(float64(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(float64(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(float64(1), float64(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(float64(1), float64(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(float64(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}

/* MaskedGreater */

func TestDense_MaskedGreater_I(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Int), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.ints()
	for i := range data {
		data[i] = int(i)
	}
	T.MaskedEqual(int(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(int(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(int(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(int(1), int(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(int(1), int(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(int(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedGreater_I8(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Int8), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.int8s()
	for i := range data {
		data[i] = int8(i)
	}
	T.MaskedEqual(int8(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(int8(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(int8(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(int8(1), int8(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(int8(1), int8(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(int8(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedGreater_I16(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Int16), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.int16s()
	for i := range data {
		data[i] = int16(i)
	}
	T.MaskedEqual(int16(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(int16(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(int16(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(int16(1), int16(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(int16(1), int16(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(int16(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedGreater_I32(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Int32), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.int32s()
	for i := range data {
		data[i] = int32(i)
	}
	T.MaskedEqual(int32(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(int32(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(int32(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(int32(1), int32(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(int32(1), int32(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(int32(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedGreater_I64(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Int64), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.int64s()
	for i := range data {
		data[i] = int64(i)
	}
	T.MaskedEqual(int64(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(int64(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(int64(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(int64(1), int64(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(int64(1), int64(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(int64(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedGreater_U(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Uint), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.uints()
	for i := range data {
		data[i] = uint(i)
	}
	T.MaskedEqual(uint(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(uint(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(uint(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(uint(1), uint(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(uint(1), uint(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(uint(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedGreater_U8(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Uint8), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.uint8s()
	for i := range data {
		data[i] = uint8(i)
	}
	T.MaskedEqual(uint8(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(uint8(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(uint8(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(uint8(1), uint8(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(uint8(1), uint8(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(uint8(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedGreater_U16(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Uint16), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.uint16s()
	for i := range data {
		data[i] = uint16(i)
	}
	T.MaskedEqual(uint16(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(uint16(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(uint16(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(uint16(1), uint16(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(uint16(1), uint16(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(uint16(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedGreater_U32(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Uint32), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.uint32s()
	for i := range data {
		data[i] = uint32(i)
	}
	T.MaskedEqual(uint32(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(uint32(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(uint32(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(uint32(1), uint32(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(uint32(1), uint32(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(uint32(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedGreater_U64(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Uint64), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.uint64s()
	for i := range data {
		data[i] = uint64(i)
	}
	T.MaskedEqual(uint64(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(uint64(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(uint64(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(uint64(1), uint64(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(uint64(1), uint64(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(uint64(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedGreater_F32(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Float32), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.float32s()
	for i := range data {
		data[i] = float32(i)
	}
	T.MaskedEqual(float32(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(float32(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(float32(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(float32(1), float32(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(float32(1), float32(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(float32(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedGreater_F64(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Float64), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.float64s()
	for i := range data {
		data[i] = float64(i)
	}
	T.MaskedEqual(float64(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(float64(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(float64(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(float64(1), float64(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(float64(1), float64(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(float64(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedGreater_Str(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(String), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.strings()
	for i := range data {
		data[i] = string(i)
	}
	T.MaskedEqual(string(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(string(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(string(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(string(1), string(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(string(1), string(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(string(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}

/* MaskedGreaterEqual */

func TestDense_MaskedGreaterEqual_I(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Int), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.ints()
	for i := range data {
		data[i] = int(i)
	}
	T.MaskedEqual(int(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(int(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(int(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(int(1), int(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(int(1), int(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(int(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedGreaterEqual_I8(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Int8), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.int8s()
	for i := range data {
		data[i] = int8(i)
	}
	T.MaskedEqual(int8(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(int8(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(int8(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(int8(1), int8(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(int8(1), int8(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(int8(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedGreaterEqual_I16(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Int16), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.int16s()
	for i := range data {
		data[i] = int16(i)
	}
	T.MaskedEqual(int16(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(int16(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(int16(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(int16(1), int16(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(int16(1), int16(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(int16(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedGreaterEqual_I32(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Int32), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.int32s()
	for i := range data {
		data[i] = int32(i)
	}
	T.MaskedEqual(int32(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(int32(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(int32(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(int32(1), int32(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(int32(1), int32(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(int32(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedGreaterEqual_I64(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Int64), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.int64s()
	for i := range data {
		data[i] = int64(i)
	}
	T.MaskedEqual(int64(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(int64(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(int64(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(int64(1), int64(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(int64(1), int64(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(int64(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedGreaterEqual_U(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Uint), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.uints()
	for i := range data {
		data[i] = uint(i)
	}
	T.MaskedEqual(uint(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(uint(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(uint(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(uint(1), uint(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(uint(1), uint(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(uint(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedGreaterEqual_U8(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Uint8), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.uint8s()
	for i := range data {
		data[i] = uint8(i)
	}
	T.MaskedEqual(uint8(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(uint8(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(uint8(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(uint8(1), uint8(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(uint8(1), uint8(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(uint8(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedGreaterEqual_U16(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Uint16), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.uint16s()
	for i := range data {
		data[i] = uint16(i)
	}
	T.MaskedEqual(uint16(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(uint16(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(uint16(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(uint16(1), uint16(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(uint16(1), uint16(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(uint16(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedGreaterEqual_U32(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Uint32), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.uint32s()
	for i := range data {
		data[i] = uint32(i)
	}
	T.MaskedEqual(uint32(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(uint32(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(uint32(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(uint32(1), uint32(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(uint32(1), uint32(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(uint32(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedGreaterEqual_U64(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Uint64), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.uint64s()
	for i := range data {
		data[i] = uint64(i)
	}
	T.MaskedEqual(uint64(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(uint64(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(uint64(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(uint64(1), uint64(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(uint64(1), uint64(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(uint64(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedGreaterEqual_F32(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Float32), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.float32s()
	for i := range data {
		data[i] = float32(i)
	}
	T.MaskedEqual(float32(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(float32(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(float32(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(float32(1), float32(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(float32(1), float32(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(float32(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedGreaterEqual_F64(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Float64), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.float64s()
	for i := range data {
		data[i] = float64(i)
	}
	T.MaskedEqual(float64(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(float64(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(float64(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(float64(1), float64(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(float64(1), float64(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(float64(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedGreaterEqual_Str(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(String), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.strings()
	for i := range data {
		data[i] = string(i)
	}
	T.MaskedEqual(string(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(string(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(string(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(string(1), string(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(string(1), string(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(string(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}

/* MaskedLess */

func TestDense_MaskedLess_I(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Int), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.ints()
	for i := range data {
		data[i] = int(i)
	}
	T.MaskedEqual(int(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(int(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(int(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(int(1), int(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(int(1), int(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(int(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedLess_I8(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Int8), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.int8s()
	for i := range data {
		data[i] = int8(i)
	}
	T.MaskedEqual(int8(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(int8(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(int8(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(int8(1), int8(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(int8(1), int8(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(int8(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedLess_I16(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Int16), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.int16s()
	for i := range data {
		data[i] = int16(i)
	}
	T.MaskedEqual(int16(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(int16(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(int16(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(int16(1), int16(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(int16(1), int16(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(int16(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedLess_I32(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Int32), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.int32s()
	for i := range data {
		data[i] = int32(i)
	}
	T.MaskedEqual(int32(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(int32(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(int32(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(int32(1), int32(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(int32(1), int32(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(int32(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedLess_I64(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Int64), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.int64s()
	for i := range data {
		data[i] = int64(i)
	}
	T.MaskedEqual(int64(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(int64(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(int64(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(int64(1), int64(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(int64(1), int64(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(int64(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedLess_U(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Uint), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.uints()
	for i := range data {
		data[i] = uint(i)
	}
	T.MaskedEqual(uint(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(uint(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(uint(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(uint(1), uint(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(uint(1), uint(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(uint(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedLess_U8(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Uint8), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.uint8s()
	for i := range data {
		data[i] = uint8(i)
	}
	T.MaskedEqual(uint8(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(uint8(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(uint8(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(uint8(1), uint8(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(uint8(1), uint8(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(uint8(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedLess_U16(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Uint16), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.uint16s()
	for i := range data {
		data[i] = uint16(i)
	}
	T.MaskedEqual(uint16(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(uint16(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(uint16(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(uint16(1), uint16(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(uint16(1), uint16(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(uint16(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedLess_U32(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Uint32), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.uint32s()
	for i := range data {
		data[i] = uint32(i)
	}
	T.MaskedEqual(uint32(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(uint32(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(uint32(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(uint32(1), uint32(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(uint32(1), uint32(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(uint32(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedLess_U64(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Uint64), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.uint64s()
	for i := range data {
		data[i] = uint64(i)
	}
	T.MaskedEqual(uint64(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(uint64(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(uint64(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(uint64(1), uint64(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(uint64(1), uint64(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(uint64(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedLess_F32(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Float32), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.float32s()
	for i := range data {
		data[i] = float32(i)
	}
	T.MaskedEqual(float32(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(float32(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(float32(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(float32(1), float32(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(float32(1), float32(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(float32(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedLess_F64(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Float64), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.float64s()
	for i := range data {
		data[i] = float64(i)
	}
	T.MaskedEqual(float64(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(float64(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(float64(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(float64(1), float64(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(float64(1), float64(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(float64(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedLess_Str(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(String), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.strings()
	for i := range data {
		data[i] = string(i)
	}
	T.MaskedEqual(string(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(string(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(string(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(string(1), string(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(string(1), string(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(string(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}

/* MaskedLessEqual */

func TestDense_MaskedLessEqual_I(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Int), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.ints()
	for i := range data {
		data[i] = int(i)
	}
	T.MaskedEqual(int(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(int(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(int(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(int(1), int(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(int(1), int(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(int(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedLessEqual_I8(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Int8), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.int8s()
	for i := range data {
		data[i] = int8(i)
	}
	T.MaskedEqual(int8(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(int8(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(int8(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(int8(1), int8(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(int8(1), int8(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(int8(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedLessEqual_I16(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Int16), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.int16s()
	for i := range data {
		data[i] = int16(i)
	}
	T.MaskedEqual(int16(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(int16(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(int16(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(int16(1), int16(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(int16(1), int16(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(int16(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedLessEqual_I32(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Int32), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.int32s()
	for i := range data {
		data[i] = int32(i)
	}
	T.MaskedEqual(int32(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(int32(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(int32(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(int32(1), int32(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(int32(1), int32(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(int32(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedLessEqual_I64(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Int64), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.int64s()
	for i := range data {
		data[i] = int64(i)
	}
	T.MaskedEqual(int64(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(int64(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(int64(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(int64(1), int64(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(int64(1), int64(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(int64(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedLessEqual_U(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Uint), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.uints()
	for i := range data {
		data[i] = uint(i)
	}
	T.MaskedEqual(uint(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(uint(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(uint(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(uint(1), uint(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(uint(1), uint(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(uint(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedLessEqual_U8(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Uint8), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.uint8s()
	for i := range data {
		data[i] = uint8(i)
	}
	T.MaskedEqual(uint8(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(uint8(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(uint8(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(uint8(1), uint8(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(uint8(1), uint8(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(uint8(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedLessEqual_U16(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Uint16), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.uint16s()
	for i := range data {
		data[i] = uint16(i)
	}
	T.MaskedEqual(uint16(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(uint16(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(uint16(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(uint16(1), uint16(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(uint16(1), uint16(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(uint16(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedLessEqual_U32(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Uint32), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.uint32s()
	for i := range data {
		data[i] = uint32(i)
	}
	T.MaskedEqual(uint32(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(uint32(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(uint32(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(uint32(1), uint32(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(uint32(1), uint32(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(uint32(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedLessEqual_U64(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Uint64), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.uint64s()
	for i := range data {
		data[i] = uint64(i)
	}
	T.MaskedEqual(uint64(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(uint64(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(uint64(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(uint64(1), uint64(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(uint64(1), uint64(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(uint64(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedLessEqual_F32(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Float32), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.float32s()
	for i := range data {
		data[i] = float32(i)
	}
	T.MaskedEqual(float32(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(float32(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(float32(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(float32(1), float32(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(float32(1), float32(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(float32(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedLessEqual_F64(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Float64), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.float64s()
	for i := range data {
		data[i] = float64(i)
	}
	T.MaskedEqual(float64(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(float64(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(float64(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(float64(1), float64(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(float64(1), float64(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(float64(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedLessEqual_Str(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(String), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.strings()
	for i := range data {
		data[i] = string(i)
	}
	T.MaskedEqual(string(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(string(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(string(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(string(1), string(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(string(1), string(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(string(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}

/* MaskedInside */

func TestDense_MaskedInside_I(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Int), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.ints()
	for i := range data {
		data[i] = int(i)
	}
	T.MaskedEqual(int(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(int(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(int(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(int(1), int(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(int(1), int(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(int(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedInside_I8(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Int8), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.int8s()
	for i := range data {
		data[i] = int8(i)
	}
	T.MaskedEqual(int8(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(int8(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(int8(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(int8(1), int8(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(int8(1), int8(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(int8(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedInside_I16(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Int16), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.int16s()
	for i := range data {
		data[i] = int16(i)
	}
	T.MaskedEqual(int16(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(int16(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(int16(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(int16(1), int16(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(int16(1), int16(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(int16(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedInside_I32(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Int32), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.int32s()
	for i := range data {
		data[i] = int32(i)
	}
	T.MaskedEqual(int32(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(int32(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(int32(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(int32(1), int32(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(int32(1), int32(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(int32(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedInside_I64(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Int64), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.int64s()
	for i := range data {
		data[i] = int64(i)
	}
	T.MaskedEqual(int64(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(int64(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(int64(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(int64(1), int64(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(int64(1), int64(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(int64(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedInside_U(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Uint), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.uints()
	for i := range data {
		data[i] = uint(i)
	}
	T.MaskedEqual(uint(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(uint(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(uint(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(uint(1), uint(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(uint(1), uint(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(uint(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedInside_U8(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Uint8), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.uint8s()
	for i := range data {
		data[i] = uint8(i)
	}
	T.MaskedEqual(uint8(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(uint8(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(uint8(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(uint8(1), uint8(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(uint8(1), uint8(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(uint8(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedInside_U16(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Uint16), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.uint16s()
	for i := range data {
		data[i] = uint16(i)
	}
	T.MaskedEqual(uint16(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(uint16(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(uint16(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(uint16(1), uint16(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(uint16(1), uint16(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(uint16(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedInside_U32(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Uint32), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.uint32s()
	for i := range data {
		data[i] = uint32(i)
	}
	T.MaskedEqual(uint32(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(uint32(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(uint32(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(uint32(1), uint32(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(uint32(1), uint32(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(uint32(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedInside_U64(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Uint64), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.uint64s()
	for i := range data {
		data[i] = uint64(i)
	}
	T.MaskedEqual(uint64(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(uint64(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(uint64(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(uint64(1), uint64(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(uint64(1), uint64(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(uint64(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedInside_F32(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Float32), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.float32s()
	for i := range data {
		data[i] = float32(i)
	}
	T.MaskedEqual(float32(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(float32(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(float32(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(float32(1), float32(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(float32(1), float32(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(float32(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedInside_F64(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Float64), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.float64s()
	for i := range data {
		data[i] = float64(i)
	}
	T.MaskedEqual(float64(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(float64(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(float64(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(float64(1), float64(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(float64(1), float64(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(float64(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedInside_Str(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(String), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.strings()
	for i := range data {
		data[i] = string(i)
	}
	T.MaskedEqual(string(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(string(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(string(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(string(1), string(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(string(1), string(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(string(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}

/* MaskedOutside */

func TestDense_MaskedOutside_I(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Int), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.ints()
	for i := range data {
		data[i] = int(i)
	}
	T.MaskedEqual(int(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(int(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(int(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(int(1), int(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(int(1), int(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(int(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedOutside_I8(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Int8), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.int8s()
	for i := range data {
		data[i] = int8(i)
	}
	T.MaskedEqual(int8(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(int8(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(int8(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(int8(1), int8(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(int8(1), int8(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(int8(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedOutside_I16(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Int16), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.int16s()
	for i := range data {
		data[i] = int16(i)
	}
	T.MaskedEqual(int16(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(int16(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(int16(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(int16(1), int16(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(int16(1), int16(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(int16(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedOutside_I32(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Int32), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.int32s()
	for i := range data {
		data[i] = int32(i)
	}
	T.MaskedEqual(int32(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(int32(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(int32(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(int32(1), int32(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(int32(1), int32(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(int32(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedOutside_I64(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Int64), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.int64s()
	for i := range data {
		data[i] = int64(i)
	}
	T.MaskedEqual(int64(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(int64(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(int64(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(int64(1), int64(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(int64(1), int64(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(int64(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedOutside_U(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Uint), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.uints()
	for i := range data {
		data[i] = uint(i)
	}
	T.MaskedEqual(uint(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(uint(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(uint(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(uint(1), uint(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(uint(1), uint(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(uint(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedOutside_U8(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Uint8), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.uint8s()
	for i := range data {
		data[i] = uint8(i)
	}
	T.MaskedEqual(uint8(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(uint8(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(uint8(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(uint8(1), uint8(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(uint8(1), uint8(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(uint8(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedOutside_U16(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Uint16), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.uint16s()
	for i := range data {
		data[i] = uint16(i)
	}
	T.MaskedEqual(uint16(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(uint16(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(uint16(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(uint16(1), uint16(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(uint16(1), uint16(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(uint16(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedOutside_U32(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Uint32), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.uint32s()
	for i := range data {
		data[i] = uint32(i)
	}
	T.MaskedEqual(uint32(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(uint32(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(uint32(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(uint32(1), uint32(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(uint32(1), uint32(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(uint32(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedOutside_U64(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Uint64), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.uint64s()
	for i := range data {
		data[i] = uint64(i)
	}
	T.MaskedEqual(uint64(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(uint64(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(uint64(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(uint64(1), uint64(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(uint64(1), uint64(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(uint64(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedOutside_F32(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Float32), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.float32s()
	for i := range data {
		data[i] = float32(i)
	}
	T.MaskedEqual(float32(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(float32(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(float32(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(float32(1), float32(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(float32(1), float32(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(float32(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedOutside_F64(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Float64), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.float64s()
	for i := range data {
		data[i] = float64(i)
	}
	T.MaskedEqual(float64(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(float64(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(float64(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(float64(1), float64(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(float64(1), float64(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(float64(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
func TestDense_MaskedOutside_Str(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(String), WithShape(2, 3, 4, 5))
	assert.False(T.IsMasked())
	data := T.strings()
	for i := range data {
		data[i] = string(i)
	}
	T.MaskedEqual(string(0))
	assert.True(T.IsMasked())
	T.MaskedEqual(string(1))
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(string(2))
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedInside(string(1), string(22))
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(string(1), string(22))
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])

	T.ResetMask()
	for i := 0; i < 5; i++ {
		T.MaskedEqual(string(i * 10))
	}
	it := IteratorFromDense(T)
	runtime.SetFinalizer(it, destroyIterator)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}

	it.Reset()
	assert.Equal(120, j)
	j = 0
	for _, _, err := it.NextValid(); err == nil; _, _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.Equal(115, j)
	j = 0
	for _, _, err := it.NextInvalid(); err == nil; _, _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.Equal(5, j)
}
