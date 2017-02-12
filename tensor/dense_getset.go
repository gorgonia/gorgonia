package tensor

import (
	"reflect"
	"unsafe"

	"github.com/pkg/errors"
)

/*
GENERATED FILE. DO NOT EDIT
*/

/* bool */

func (t *Dense) bools() []bool      { return *(*[]bool)(unsafe.Pointer(t.hdr)) }
func (t *Dense) setB(i int, x bool) { t.bools()[i] = x }
func (t *Dense) getB(i int) bool    { return t.bools()[i] }

/* int */

func (t *Dense) ints() []int       { return *(*[]int)(unsafe.Pointer(t.hdr)) }
func (t *Dense) setI(i int, x int) { t.ints()[i] = x }
func (t *Dense) getI(i int) int    { return t.ints()[i] }

/* int8 */

func (t *Dense) int8s() []int8       { return *(*[]int8)(unsafe.Pointer(t.hdr)) }
func (t *Dense) setI8(i int, x int8) { t.int8s()[i] = x }
func (t *Dense) getI8(i int) int8    { return t.int8s()[i] }

/* int16 */

func (t *Dense) int16s() []int16       { return *(*[]int16)(unsafe.Pointer(t.hdr)) }
func (t *Dense) setI16(i int, x int16) { t.int16s()[i] = x }
func (t *Dense) getI16(i int) int16    { return t.int16s()[i] }

/* int32 */

func (t *Dense) int32s() []int32       { return *(*[]int32)(unsafe.Pointer(t.hdr)) }
func (t *Dense) setI32(i int, x int32) { t.int32s()[i] = x }
func (t *Dense) getI32(i int) int32    { return t.int32s()[i] }

/* int64 */

func (t *Dense) int64s() []int64       { return *(*[]int64)(unsafe.Pointer(t.hdr)) }
func (t *Dense) setI64(i int, x int64) { t.int64s()[i] = x }
func (t *Dense) getI64(i int) int64    { return t.int64s()[i] }

/* uint */

func (t *Dense) uints() []uint      { return *(*[]uint)(unsafe.Pointer(t.hdr)) }
func (t *Dense) setU(i int, x uint) { t.uints()[i] = x }
func (t *Dense) getU(i int) uint    { return t.uints()[i] }

/* uint8 */

func (t *Dense) uint8s() []uint8      { return *(*[]uint8)(unsafe.Pointer(t.hdr)) }
func (t *Dense) setU8(i int, x uint8) { t.uint8s()[i] = x }
func (t *Dense) getU8(i int) uint8    { return t.uint8s()[i] }

/* uint16 */

func (t *Dense) uint16s() []uint16      { return *(*[]uint16)(unsafe.Pointer(t.hdr)) }
func (t *Dense) setU16(i int, x uint16) { t.uint16s()[i] = x }
func (t *Dense) getU16(i int) uint16    { return t.uint16s()[i] }

/* uint32 */

func (t *Dense) uint32s() []uint32      { return *(*[]uint32)(unsafe.Pointer(t.hdr)) }
func (t *Dense) setU32(i int, x uint32) { t.uint32s()[i] = x }
func (t *Dense) getU32(i int) uint32    { return t.uint32s()[i] }

/* uint64 */

func (t *Dense) uint64s() []uint64      { return *(*[]uint64)(unsafe.Pointer(t.hdr)) }
func (t *Dense) setU64(i int, x uint64) { t.uint64s()[i] = x }
func (t *Dense) getU64(i int) uint64    { return t.uint64s()[i] }

/* uintptr */

func (t *Dense) uintptrs() []uintptr         { return *(*[]uintptr)(unsafe.Pointer(t.hdr)) }
func (t *Dense) setUintptr(i int, x uintptr) { t.uintptrs()[i] = x }
func (t *Dense) getUintptr(i int) uintptr    { return t.uintptrs()[i] }

/* float32 */

func (t *Dense) float32s() []float32     { return *(*[]float32)(unsafe.Pointer(t.hdr)) }
func (t *Dense) setF32(i int, x float32) { t.float32s()[i] = x }
func (t *Dense) getF32(i int) float32    { return t.float32s()[i] }

/* float64 */

func (t *Dense) float64s() []float64     { return *(*[]float64)(unsafe.Pointer(t.hdr)) }
func (t *Dense) setF64(i int, x float64) { t.float64s()[i] = x }
func (t *Dense) getF64(i int) float64    { return t.float64s()[i] }

/* complex64 */

func (t *Dense) complex64s() []complex64   { return *(*[]complex64)(unsafe.Pointer(t.hdr)) }
func (t *Dense) setC64(i int, x complex64) { t.complex64s()[i] = x }
func (t *Dense) getC64(i int) complex64    { return t.complex64s()[i] }

/* complex128 */

func (t *Dense) complex128s() []complex128   { return *(*[]complex128)(unsafe.Pointer(t.hdr)) }
func (t *Dense) setC128(i int, x complex128) { t.complex128s()[i] = x }
func (t *Dense) getC128(i int) complex128    { return t.complex128s()[i] }

/* string */

func (t *Dense) strings() []string      { return *(*[]string)(unsafe.Pointer(t.hdr)) }
func (t *Dense) setStr(i int, x string) { t.strings()[i] = x }
func (t *Dense) getStr(i int) string    { return t.strings()[i] }

/* unsafe.Pointer */

func (t *Dense) unsafePointers() []unsafe.Pointer         { return *(*[]unsafe.Pointer)(unsafe.Pointer(t.hdr)) }
func (t *Dense) setUnsafePointer(i int, x unsafe.Pointer) { t.unsafePointers()[i] = x }
func (t *Dense) getUnsafePointer(i int) unsafe.Pointer    { return t.unsafePointers()[i] }

func (t *Dense) makeArray(size int) {
	switch t.t.Kind() {
	case reflect.Bool:
		arr := make([]bool, size)
		t.fromSlice(arr)
	case reflect.Int:
		arr := make([]int, size)
		t.fromSlice(arr)
	case reflect.Int8:
		arr := make([]int8, size)
		t.fromSlice(arr)
	case reflect.Int16:
		arr := make([]int16, size)
		t.fromSlice(arr)
	case reflect.Int32:
		arr := make([]int32, size)
		t.fromSlice(arr)
	case reflect.Int64:
		arr := make([]int64, size)
		t.fromSlice(arr)
	case reflect.Uint:
		arr := make([]uint, size)
		t.fromSlice(arr)
	case reflect.Uint8:
		arr := make([]uint8, size)
		t.fromSlice(arr)
	case reflect.Uint16:
		arr := make([]uint16, size)
		t.fromSlice(arr)
	case reflect.Uint32:
		arr := make([]uint32, size)
		t.fromSlice(arr)
	case reflect.Uint64:
		arr := make([]uint64, size)
		t.fromSlice(arr)
	case reflect.Uintptr:
		arr := make([]uintptr, size)
		t.fromSlice(arr)
	case reflect.Float32:
		arr := make([]float32, size)
		t.fromSlice(arr)
	case reflect.Float64:
		arr := make([]float64, size)
		t.fromSlice(arr)
	case reflect.Complex64:
		arr := make([]complex64, size)
		t.fromSlice(arr)
	case reflect.Complex128:
		arr := make([]complex128, size)
		t.fromSlice(arr)

	case reflect.String:
		arr := make([]string, size)
		t.fromSlice(arr)

	case reflect.UnsafePointer:
		arr := make([]unsafe.Pointer, size)
		t.fromSlice(arr)
	default:

	}
}

// Set sets the value of the underlying array at the index i.
func (t *Dense) Set(i int, x interface{}) {
	switch t.t.Kind() {
	case reflect.Bool:
		xv := x.(bool)
		t.setB(i, xv)
	case reflect.Int:
		xv := x.(int)
		t.setI(i, xv)
	case reflect.Int8:
		xv := x.(int8)
		t.setI8(i, xv)
	case reflect.Int16:
		xv := x.(int16)
		t.setI16(i, xv)
	case reflect.Int32:
		xv := x.(int32)
		t.setI32(i, xv)
	case reflect.Int64:
		xv := x.(int64)
		t.setI64(i, xv)
	case reflect.Uint:
		xv := x.(uint)
		t.setU(i, xv)
	case reflect.Uint8:
		xv := x.(uint8)
		t.setU8(i, xv)
	case reflect.Uint16:
		xv := x.(uint16)
		t.setU16(i, xv)
	case reflect.Uint32:
		xv := x.(uint32)
		t.setU32(i, xv)
	case reflect.Uint64:
		xv := x.(uint64)
		t.setU64(i, xv)
	case reflect.Uintptr:
		xv := x.(uintptr)
		t.setUintptr(i, xv)
	case reflect.Float32:
		xv := x.(float32)
		t.setF32(i, xv)
	case reflect.Float64:
		xv := x.(float64)
		t.setF64(i, xv)
	case reflect.Complex64:
		xv := x.(complex64)
		t.setC64(i, xv)
	case reflect.Complex128:
		xv := x.(complex128)
		t.setC128(i, xv)
	case reflect.String:
		xv := x.(string)
		t.setStr(i, xv)
	case reflect.UnsafePointer:
		xv := x.(unsafe.Pointer)
		t.setUnsafePointer(i, xv)
	default:
		xv := reflect.ValueOf(x)
		ptr := uintptr(t.data)
		want := ptr + uintptr(i)*t.t.Size()
		val := reflect.NewAt(t.t, unsafe.Pointer(want))
		val = reflect.Indirect(val)
		val.Set(xv)
	}
}

// Get returns the ith element of the underlying array of the *Dense tensor.
func (t *Dense) Get(i int) interface{} {
	switch t.t.Kind() {
	case reflect.Bool:
		return t.getB(i)
	case reflect.Int:
		return t.getI(i)
	case reflect.Int8:
		return t.getI8(i)
	case reflect.Int16:
		return t.getI16(i)
	case reflect.Int32:
		return t.getI32(i)
	case reflect.Int64:
		return t.getI64(i)
	case reflect.Uint:
		return t.getU(i)
	case reflect.Uint8:
		return t.getU8(i)
	case reflect.Uint16:
		return t.getU16(i)
	case reflect.Uint32:
		return t.getU32(i)
	case reflect.Uint64:
		return t.getU64(i)
	case reflect.Uintptr:
		return t.getUintptr(i)
	case reflect.Float32:
		return t.getF32(i)
	case reflect.Float64:
		return t.getF64(i)
	case reflect.Complex64:
		return t.getC64(i)
	case reflect.Complex128:
		return t.getC128(i)
	case reflect.String:
		return t.getStr(i)
	case reflect.UnsafePointer:
		return t.getUnsafePointer(i)
	default:
		at := uintptr(t.data) + uintptr(i)*t.t.Size()
		val := reflect.NewAt(t.t, unsafe.Pointer(at))
		val = reflect.Indirect(val)
		return val.Interface()
	}
}

// Memset sets all values in the *Dense tensor to x.
func (t *Dense) Memset(x interface{}) error {
	if t.IsMaterializable() {
		return t.memsetIter(x)
	}
	switch t.t.Kind() {
	case reflect.Bool:
		xv, ok := x.(bool)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.bools()
		for i := range data {
			data[i] = xv
		}
	case reflect.Int:
		xv, ok := x.(int)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.ints()
		for i := range data {
			data[i] = xv
		}
	case reflect.Int8:
		xv, ok := x.(int8)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.int8s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Int16:
		xv, ok := x.(int16)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.int16s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Int32:
		xv, ok := x.(int32)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.int32s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Int64:
		xv, ok := x.(int64)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.int64s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Uint:
		xv, ok := x.(uint)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.uints()
		for i := range data {
			data[i] = xv
		}
	case reflect.Uint8:
		xv, ok := x.(uint8)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.uint8s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Uint16:
		xv, ok := x.(uint16)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.uint16s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Uint32:
		xv, ok := x.(uint32)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.uint32s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Uint64:
		xv, ok := x.(uint64)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.uint64s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Uintptr:
		xv, ok := x.(uintptr)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.uintptrs()
		for i := range data {
			data[i] = xv
		}
	case reflect.Float32:
		xv, ok := x.(float32)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.float32s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Float64:
		xv, ok := x.(float64)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.float64s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Complex64:
		xv, ok := x.(complex64)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.complex64s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Complex128:
		xv, ok := x.(complex128)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.complex128s()
		for i := range data {
			data[i] = xv
		}
	case reflect.String:
		xv, ok := x.(string)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.strings()
		for i := range data {
			data[i] = xv
		}
	case reflect.UnsafePointer:
		xv, ok := x.(unsafe.Pointer)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.unsafePointers()
		for i := range data {
			data[i] = xv
		}
	default:
		xv := reflect.ValueOf(x)
		ptr := uintptr(t.data)
		for i := 0; i < t.hdr.Len; i++ {
			want := ptr + uintptr(i)*t.t.Size()
			val := reflect.NewAt(t.t, unsafe.Pointer(want))
			val = reflect.Indirect(val)
			val.Set(xv)
		}
	}
	return nil
}

func (t *Dense) memsetIter(x interface{}) (err error) {
	it := NewFlatIterator(t.AP)
	var i int
	switch t.t.Kind() {
	case reflect.Bool:
		xv, ok := x.(bool)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.bools()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Int:
		xv, ok := x.(int)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.ints()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Int8:
		xv, ok := x.(int8)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.int8s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Int16:
		xv, ok := x.(int16)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.int16s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Int32:
		xv, ok := x.(int32)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.int32s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Int64:
		xv, ok := x.(int64)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.int64s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Uint:
		xv, ok := x.(uint)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.uints()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Uint8:
		xv, ok := x.(uint8)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.uint8s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Uint16:
		xv, ok := x.(uint16)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.uint16s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Uint32:
		xv, ok := x.(uint32)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.uint32s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Uint64:
		xv, ok := x.(uint64)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.uint64s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Uintptr:
		xv, ok := x.(uintptr)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.uintptrs()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Float32:
		xv, ok := x.(float32)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.float32s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Float64:
		xv, ok := x.(float64)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.float64s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Complex64:
		xv, ok := x.(complex64)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.complex64s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Complex128:
		xv, ok := x.(complex128)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.complex128s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.String:
		xv, ok := x.(string)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.strings()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.UnsafePointer:
		xv, ok := x.(unsafe.Pointer)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.unsafePointers()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	default:
		xv := reflect.ValueOf(x)
		ptr := uintptr(t.data)
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			want := ptr + uintptr(i)*t.t.Size()
			val := reflect.NewAt(t.t, unsafe.Pointer(want))
			val = reflect.Indirect(val)
			val.Set(xv)
		}
		err = handleNoOp(err)
	}
	return
}

// Zero zeroes out the underlying array of the *Dense tensor
func (t *Dense) Zero() {
	if t.IsMaterializable() {
		if err := t.zeroIter(); err != nil {
			panic(err)
		}
	}
	switch t.t.Kind() {
	case reflect.Bool:
		data := t.bools()
		for i := range data {
			data[i] = false

		}
	case reflect.Int:
		data := t.ints()
		for i := range data {
			data[i] = 0
		}
	case reflect.Int8:
		data := t.int8s()
		for i := range data {
			data[i] = 0
		}
	case reflect.Int16:
		data := t.int16s()
		for i := range data {
			data[i] = 0
		}
	case reflect.Int32:
		data := t.int32s()
		for i := range data {
			data[i] = 0
		}
	case reflect.Int64:
		data := t.int64s()
		for i := range data {
			data[i] = 0
		}
	case reflect.Uint:
		data := t.uints()
		for i := range data {
			data[i] = 0
		}
	case reflect.Uint8:
		data := t.uint8s()
		for i := range data {
			data[i] = 0
		}
	case reflect.Uint16:
		data := t.uint16s()
		for i := range data {
			data[i] = 0
		}
	case reflect.Uint32:
		data := t.uint32s()
		for i := range data {
			data[i] = 0
		}
	case reflect.Uint64:
		data := t.uint64s()
		for i := range data {
			data[i] = 0
		}
	case reflect.Uintptr:
		data := t.uintptrs()
		for i := range data {
			data[i] = 0
		}
	case reflect.Float32:
		data := t.float32s()
		for i := range data {
			data[i] = 0
		}
	case reflect.Float64:
		data := t.float64s()
		for i := range data {
			data[i] = 0
		}
	case reflect.Complex64:
		data := t.complex64s()
		for i := range data {
			data[i] = 0
		}
	case reflect.Complex128:
		data := t.complex128s()
		for i := range data {
			data[i] = 0
		}
	case reflect.String:
		data := t.strings()
		for i := range data {
			data[i] = ""

		}
	case reflect.UnsafePointer:
		data := t.unsafePointers()
		for i := range data {
			data[i] = nil

		}
	default:
		ptr := uintptr(t.data)
		for i := 0; i < t.hdr.Len; i++ {
			want := ptr + uintptr(i)*t.t.Size()
			val := reflect.NewAt(t.t, unsafe.Pointer(want))
			val = reflect.Indirect(val)
			val.Set(reflect.Zero(t.t))
		}
	}
}

func (t *Dense) zeroIter() (err error) {
	it := NewFlatIterator(t.AP)
	var i int
	switch t.t.Kind() {
	case reflect.Bool:
		data := t.bools()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = false

		}
		err = handleNoOp(err)
	case reflect.Int:
		data := t.ints()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.Int8:
		data := t.int8s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.Int16:
		data := t.int16s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.Int32:
		data := t.int32s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.Int64:
		data := t.int64s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.Uint:
		data := t.uints()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.Uint8:
		data := t.uint8s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.Uint16:
		data := t.uint16s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.Uint32:
		data := t.uint32s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.Uint64:
		data := t.uint64s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.Uintptr:
		data := t.uintptrs()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.Float32:
		data := t.float32s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.Float64:
		data := t.float64s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.Complex64:
		data := t.complex64s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.Complex128:
		data := t.complex128s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.String:
		data := t.strings()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = ""

		}
		err = handleNoOp(err)
	case reflect.UnsafePointer:
		data := t.unsafePointers()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = nil

		}
		err = handleNoOp(err)
	default:
		ptr := uintptr(t.data)
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			want := ptr + uintptr(i)*t.t.Size()
			val := reflect.NewAt(t.t, unsafe.Pointer(want))
			val = reflect.Indirect(val)
			val.Set(reflect.Zero(t.t))
		}
		err = handleNoOp(err)
	}
	return
}

func copyDense(dest, src *Dense) int {
	if dest.t != src.t {
		err := errors.Errorf(dtypeMismatch, src.t, dest.t)
		panic(err.Error())
	}
	switch dest.t.Kind() {
	case reflect.Bool:
		return copy(dest.bools(), src.bools())
	case reflect.Int:
		return copy(dest.ints(), src.ints())
	case reflect.Int8:
		return copy(dest.int8s(), src.int8s())
	case reflect.Int16:
		return copy(dest.int16s(), src.int16s())
	case reflect.Int32:
		return copy(dest.int32s(), src.int32s())
	case reflect.Int64:
		return copy(dest.int64s(), src.int64s())
	case reflect.Uint:
		return copy(dest.uints(), src.uints())
	case reflect.Uint8:
		return copy(dest.uint8s(), src.uint8s())
	case reflect.Uint16:
		return copy(dest.uint16s(), src.uint16s())
	case reflect.Uint32:
		return copy(dest.uint32s(), src.uint32s())
	case reflect.Uint64:
		return copy(dest.uint64s(), src.uint64s())
	case reflect.Uintptr:
		return copy(dest.uintptrs(), src.uintptrs())
	case reflect.Float32:
		return copy(dest.float32s(), src.float32s())
	case reflect.Float64:
		return copy(dest.float64s(), src.float64s())
	case reflect.Complex64:
		return copy(dest.complex64s(), src.complex64s())
	case reflect.Complex128:
		return copy(dest.complex128s(), src.complex128s())

	case reflect.String:
		return copy(dest.strings(), src.strings())

	case reflect.UnsafePointer:
		return copy(dest.unsafePointers(), src.unsafePointers())
	default:
		dv := reflect.ValueOf(dest.v)
		sv := reflect.ValueOf(src.v)
		return reflect.Copy(dv, sv)
	}
}

func copySliced(dest *Dense, dstart, dend int, src *Dense, sstart, send int) int {
	if dest.t != src.t {
		panic("Cannot copy arrays of different types")
	}
	switch dest.t.Kind() {
	case reflect.Bool:
		return copy(dest.bools()[dstart:dend], src.bools()[sstart:send])
	case reflect.Int:
		return copy(dest.ints()[dstart:dend], src.ints()[sstart:send])
	case reflect.Int8:
		return copy(dest.int8s()[dstart:dend], src.int8s()[sstart:send])
	case reflect.Int16:
		return copy(dest.int16s()[dstart:dend], src.int16s()[sstart:send])
	case reflect.Int32:
		return copy(dest.int32s()[dstart:dend], src.int32s()[sstart:send])
	case reflect.Int64:
		return copy(dest.int64s()[dstart:dend], src.int64s()[sstart:send])
	case reflect.Uint:
		return copy(dest.uints()[dstart:dend], src.uints()[sstart:send])
	case reflect.Uint8:
		return copy(dest.uint8s()[dstart:dend], src.uint8s()[sstart:send])
	case reflect.Uint16:
		return copy(dest.uint16s()[dstart:dend], src.uint16s()[sstart:send])
	case reflect.Uint32:
		return copy(dest.uint32s()[dstart:dend], src.uint32s()[sstart:send])
	case reflect.Uint64:
		return copy(dest.uint64s()[dstart:dend], src.uint64s()[sstart:send])
	case reflect.Uintptr:
		return copy(dest.uintptrs()[dstart:dend], src.uintptrs()[sstart:send])
	case reflect.Float32:
		return copy(dest.float32s()[dstart:dend], src.float32s()[sstart:send])
	case reflect.Float64:
		return copy(dest.float64s()[dstart:dend], src.float64s()[sstart:send])
	case reflect.Complex64:
		return copy(dest.complex64s()[dstart:dend], src.complex64s()[sstart:send])
	case reflect.Complex128:
		return copy(dest.complex128s()[dstart:dend], src.complex128s()[sstart:send])

	case reflect.String:
		return copy(dest.strings()[dstart:dend], src.strings()[sstart:send])

	case reflect.UnsafePointer:
		return copy(dest.unsafePointers()[dstart:dend], src.unsafePointers()[sstart:send])
	default:
		dv := reflect.ValueOf(dest.v)
		dv = dv.Slice(dstart, dend)
		sv := reflect.ValueOf(src.v)
		sv = sv.Slice(sstart, send)
		return reflect.Copy(dv, sv)
	}
}

func copyDenseIter(dest, src *Dense, diter, siter *FlatIterator) (int, error) {
	if dest.t != src.t {
		panic("Cannot copy arrays of different types")
	}

	if diter == nil && siter == nil && !dest.IsMaterializable() && !src.IsMaterializable() {
		return copyDense(dest, src), nil
	}

	if diter == nil {
		diter = NewFlatIterator(dest.AP)
	}
	if siter == nil {
		siter = NewFlatIterator(src.AP)
	}

	k := dest.t.Kind()
	var i, j, count int
	var err error
	for {
		if i, err = diter.Next(); err != nil {
			if err = handleNoOp(err); err != nil {
				return count, err
			}
			break
		}
		if j, err = siter.Next(); err != nil {
			if err = handleNoOp(err); err != nil {
				return count, err
			}
			break
		}
		switch k {
		case reflect.Bool:
			dest.setB(i, src.getB(j))
		case reflect.Int:
			dest.setI(i, src.getI(j))
		case reflect.Int8:
			dest.setI8(i, src.getI8(j))
		case reflect.Int16:
			dest.setI16(i, src.getI16(j))
		case reflect.Int32:
			dest.setI32(i, src.getI32(j))
		case reflect.Int64:
			dest.setI64(i, src.getI64(j))
		case reflect.Uint:
			dest.setU(i, src.getU(j))
		case reflect.Uint8:
			dest.setU8(i, src.getU8(j))
		case reflect.Uint16:
			dest.setU16(i, src.getU16(j))
		case reflect.Uint32:
			dest.setU32(i, src.getU32(j))
		case reflect.Uint64:
			dest.setU64(i, src.getU64(j))
		case reflect.Uintptr:
			dest.setUintptr(i, src.getUintptr(j))
		case reflect.Float32:
			dest.setF32(i, src.getF32(j))
		case reflect.Float64:
			dest.setF64(i, src.getF64(j))
		case reflect.Complex64:
			dest.setC64(i, src.getC64(j))
		case reflect.Complex128:
			dest.setC128(i, src.getC128(j))
		case reflect.String:
			dest.setStr(i, src.getStr(j))
		case reflect.UnsafePointer:
			dest.setUnsafePointer(i, src.getUnsafePointer(j))
		default:
			dest.Set(i, src.Get(j))
		}
		count++
	}
	return count, err
}

// the method assumes the AP and metadata has already been set and this is simply slicing the values
func (t *Dense) slice(start, end int) {
	switch t.t.Kind() {
	case reflect.Bool:
		data := t.bools()[start:end]
		t.fromSlice(data)
	case reflect.Int:
		data := t.ints()[start:end]
		t.fromSlice(data)
	case reflect.Int8:
		data := t.int8s()[start:end]
		t.fromSlice(data)
	case reflect.Int16:
		data := t.int16s()[start:end]
		t.fromSlice(data)
	case reflect.Int32:
		data := t.int32s()[start:end]
		t.fromSlice(data)
	case reflect.Int64:
		data := t.int64s()[start:end]
		t.fromSlice(data)
	case reflect.Uint:
		data := t.uints()[start:end]
		t.fromSlice(data)
	case reflect.Uint8:
		data := t.uint8s()[start:end]
		t.fromSlice(data)
	case reflect.Uint16:
		data := t.uint16s()[start:end]
		t.fromSlice(data)
	case reflect.Uint32:
		data := t.uint32s()[start:end]
		t.fromSlice(data)
	case reflect.Uint64:
		data := t.uint64s()[start:end]
		t.fromSlice(data)
	case reflect.Uintptr:
		data := t.uintptrs()[start:end]
		t.fromSlice(data)
	case reflect.Float32:
		data := t.float32s()[start:end]
		t.fromSlice(data)
	case reflect.Float64:
		data := t.float64s()[start:end]
		t.fromSlice(data)
	case reflect.Complex64:
		data := t.complex64s()[start:end]
		t.fromSlice(data)
	case reflect.Complex128:
		data := t.complex128s()[start:end]
		t.fromSlice(data)

	case reflect.String:
		data := t.strings()[start:end]
		t.fromSlice(data)

	case reflect.UnsafePointer:
		data := t.unsafePointers()[start:end]
		t.fromSlice(data)
	default:
		v := reflect.ValueOf(t.v)
		v = v.Slice(start, end)
		t.fromSlice(v.Interface())
	}
}

// Eq checks that any two things are equal. If the shapes are the same, but the strides are not the same, it's will still be considered the same
func (t *Dense) Eq(other interface{}) bool {
	if ot, ok := other.(*Dense); ok {
		if ot == t {
			return true
		}

		if ot.len() != t.len() {
			return false
		}

		if t.t != ot.t {
			return false
		}

		if !t.Shape().Eq(ot.Shape()) {
			return false
		}

		switch t.t.Kind() {
		case reflect.Bool:
			for i, v := range t.bools() {
				if ot.getB(i) != v {
					return false
				}
			}
		case reflect.Int:
			for i, v := range t.ints() {
				if ot.getI(i) != v {
					return false
				}
			}
		case reflect.Int8:
			for i, v := range t.int8s() {
				if ot.getI8(i) != v {
					return false
				}
			}
		case reflect.Int16:
			for i, v := range t.int16s() {
				if ot.getI16(i) != v {
					return false
				}
			}
		case reflect.Int32:
			for i, v := range t.int32s() {
				if ot.getI32(i) != v {
					return false
				}
			}
		case reflect.Int64:
			for i, v := range t.int64s() {
				if ot.getI64(i) != v {
					return false
				}
			}
		case reflect.Uint:
			for i, v := range t.uints() {
				if ot.getU(i) != v {
					return false
				}
			}
		case reflect.Uint8:
			for i, v := range t.uint8s() {
				if ot.getU8(i) != v {
					return false
				}
			}
		case reflect.Uint16:
			for i, v := range t.uint16s() {
				if ot.getU16(i) != v {
					return false
				}
			}
		case reflect.Uint32:
			for i, v := range t.uint32s() {
				if ot.getU32(i) != v {
					return false
				}
			}
		case reflect.Uint64:
			for i, v := range t.uint64s() {
				if ot.getU64(i) != v {
					return false
				}
			}
		case reflect.Uintptr:
			for i, v := range t.uintptrs() {
				if ot.getUintptr(i) != v {
					return false
				}
			}
		case reflect.Float32:
			for i, v := range t.float32s() {
				if ot.getF32(i) != v {
					return false
				}
			}
		case reflect.Float64:
			for i, v := range t.float64s() {
				if ot.getF64(i) != v {
					return false
				}
			}
		case reflect.Complex64:
			for i, v := range t.complex64s() {
				if ot.getC64(i) != v {
					return false
				}
			}
		case reflect.Complex128:
			for i, v := range t.complex128s() {
				if ot.getC128(i) != v {
					return false
				}
			}
		case reflect.String:
			for i, v := range t.strings() {
				if ot.getStr(i) != v {
					return false
				}
			}
		case reflect.UnsafePointer:
			for i, v := range t.unsafePointers() {
				if ot.getUnsafePointer(i) != v {
					return false
				}
			}
		default:
			for i := 0; i < t.len(); i++ {
				if !reflect.DeepEqual(t.Get(i), ot.Get(i)) {
					return false
				}
			}
		}
		return true
	}
	return false
}
