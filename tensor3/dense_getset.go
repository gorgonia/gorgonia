package tensor

import (
	"reflect"
	"unsafe"
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

func (t *Dense) set(i int, x interface{}) {
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

func (t *Dense) get(i int) interface{} {
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

func copyDense(dest, src *Dense) int {
	if dest.t != src.t {
		panic("Cannot copy arrays of different types")
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

func copyDenseIter(dest, src *Dense) int {
	if dest.t != src.t {
		panic("Cannot copy arrays of different types")
	}

	siter := NewFlatIterator(src.AP)
	diter := NewFlatIterator(dest.AP)

	k := dest.t.Kind()
	var i, j, count int
	var err error
	for {
		if i, err = diter.Next(); err != nil {
			if _, ok := err.(NoOpError); !ok {
				panic(err)
			}
			err = nil
			break
		}
		if j, err = siter.Next(); err != nil {
			if _, ok := err.(NoOpError); !ok {
				panic(err)
			}
			err = nil
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
			dest.set(i, src.get(j))
		}
		count++
	}
	return count
}
