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

func (h *header) Bools() []bool      { return *(*[]bool)(unsafe.Pointer(h)) }
func (h *header) SetB(i int, x bool) { h.Bools()[i] = x }
func (h *header) GetB(i int) bool    { return h.Bools()[i] }

/* int */

func (h *header) Ints() []int       { return *(*[]int)(unsafe.Pointer(h)) }
func (h *header) SetI(i int, x int) { h.Ints()[i] = x }
func (h *header) GetI(i int) int    { return h.Ints()[i] }

/* int8 */

func (h *header) Int8s() []int8       { return *(*[]int8)(unsafe.Pointer(h)) }
func (h *header) SetI8(i int, x int8) { h.Int8s()[i] = x }
func (h *header) GetI8(i int) int8    { return h.Int8s()[i] }

/* int16 */

func (h *header) Int16s() []int16       { return *(*[]int16)(unsafe.Pointer(h)) }
func (h *header) SetI16(i int, x int16) { h.Int16s()[i] = x }
func (h *header) GetI16(i int) int16    { return h.Int16s()[i] }

/* int32 */

func (h *header) Int32s() []int32       { return *(*[]int32)(unsafe.Pointer(h)) }
func (h *header) SetI32(i int, x int32) { h.Int32s()[i] = x }
func (h *header) GetI32(i int) int32    { return h.Int32s()[i] }

/* int64 */

func (h *header) Int64s() []int64       { return *(*[]int64)(unsafe.Pointer(h)) }
func (h *header) SetI64(i int, x int64) { h.Int64s()[i] = x }
func (h *header) GetI64(i int) int64    { return h.Int64s()[i] }

/* uint */

func (h *header) Uints() []uint      { return *(*[]uint)(unsafe.Pointer(h)) }
func (h *header) SetU(i int, x uint) { h.Uints()[i] = x }
func (h *header) GetU(i int) uint    { return h.Uints()[i] }

/* uint8 */

func (h *header) Uint8s() []uint8      { return *(*[]uint8)(unsafe.Pointer(h)) }
func (h *header) SetU8(i int, x uint8) { h.Uint8s()[i] = x }
func (h *header) GetU8(i int) uint8    { return h.Uint8s()[i] }

/* uint16 */

func (h *header) Uint16s() []uint16      { return *(*[]uint16)(unsafe.Pointer(h)) }
func (h *header) SetU16(i int, x uint16) { h.Uint16s()[i] = x }
func (h *header) GetU16(i int) uint16    { return h.Uint16s()[i] }

/* uint32 */

func (h *header) Uint32s() []uint32      { return *(*[]uint32)(unsafe.Pointer(h)) }
func (h *header) SetU32(i int, x uint32) { h.Uint32s()[i] = x }
func (h *header) GetU32(i int) uint32    { return h.Uint32s()[i] }

/* uint64 */

func (h *header) Uint64s() []uint64      { return *(*[]uint64)(unsafe.Pointer(h)) }
func (h *header) SetU64(i int, x uint64) { h.Uint64s()[i] = x }
func (h *header) GetU64(i int) uint64    { return h.Uint64s()[i] }

/* uintptr */

func (h *header) Uintptrs() []uintptr         { return *(*[]uintptr)(unsafe.Pointer(h)) }
func (h *header) SetUintptr(i int, x uintptr) { h.Uintptrs()[i] = x }
func (h *header) GetUintptr(i int) uintptr    { return h.Uintptrs()[i] }

/* float32 */

func (h *header) Float32s() []float32     { return *(*[]float32)(unsafe.Pointer(h)) }
func (h *header) SetF32(i int, x float32) { h.Float32s()[i] = x }
func (h *header) GetF32(i int) float32    { return h.Float32s()[i] }

/* float64 */

func (h *header) Float64s() []float64     { return *(*[]float64)(unsafe.Pointer(h)) }
func (h *header) SetF64(i int, x float64) { h.Float64s()[i] = x }
func (h *header) GetF64(i int) float64    { return h.Float64s()[i] }

/* complex64 */

func (h *header) Complex64s() []complex64   { return *(*[]complex64)(unsafe.Pointer(h)) }
func (h *header) SetC64(i int, x complex64) { h.Complex64s()[i] = x }
func (h *header) GetC64(i int) complex64    { return h.Complex64s()[i] }

/* complex128 */

func (h *header) Complex128s() []complex128   { return *(*[]complex128)(unsafe.Pointer(h)) }
func (h *header) SetC128(i int, x complex128) { h.Complex128s()[i] = x }
func (h *header) GetC128(i int) complex128    { return h.Complex128s()[i] }

/* string */

func (h *header) Strings() []string      { return *(*[]string)(unsafe.Pointer(h)) }
func (h *header) SetStr(i int, x string) { h.Strings()[i] = x }
func (h *header) GetStr(i int) string    { return h.Strings()[i] }

/* unsafe.Pointer */

func (h *header) UnsafePointers() []unsafe.Pointer         { return *(*[]unsafe.Pointer)(unsafe.Pointer(h)) }
func (h *header) SetUnsafePointer(i int, x unsafe.Pointer) { h.UnsafePointers()[i] = x }
func (h *header) GetUnsafePointer(i int) unsafe.Pointer    { return h.UnsafePointers()[i] }

// Set sets the value of the underlying array at the index i.
func (a *array) Set(i int, x interface{}) {
	switch a.t.Kind() {
	case reflect.Bool:
		xv := x.(bool)
		a.SetB(i, xv)
	case reflect.Int:
		xv := x.(int)
		a.SetI(i, xv)
	case reflect.Int8:
		xv := x.(int8)
		a.SetI8(i, xv)
	case reflect.Int16:
		xv := x.(int16)
		a.SetI16(i, xv)
	case reflect.Int32:
		xv := x.(int32)
		a.SetI32(i, xv)
	case reflect.Int64:
		xv := x.(int64)
		a.SetI64(i, xv)
	case reflect.Uint:
		xv := x.(uint)
		a.SetU(i, xv)
	case reflect.Uint8:
		xv := x.(uint8)
		a.SetU8(i, xv)
	case reflect.Uint16:
		xv := x.(uint16)
		a.SetU16(i, xv)
	case reflect.Uint32:
		xv := x.(uint32)
		a.SetU32(i, xv)
	case reflect.Uint64:
		xv := x.(uint64)
		a.SetU64(i, xv)
	case reflect.Uintptr:
		xv := x.(uintptr)
		a.SetUintptr(i, xv)
	case reflect.Float32:
		xv := x.(float32)
		a.SetF32(i, xv)
	case reflect.Float64:
		xv := x.(float64)
		a.SetF64(i, xv)
	case reflect.Complex64:
		xv := x.(complex64)
		a.SetC64(i, xv)
	case reflect.Complex128:
		xv := x.(complex128)
		a.SetC128(i, xv)
	case reflect.String:
		xv := x.(string)
		a.SetStr(i, xv)
	case reflect.UnsafePointer:
		xv := x.(unsafe.Pointer)
		a.SetUnsafePointer(i, xv)
	default:
		xv := reflect.ValueOf(x)
		ptr := uintptr(a.ptr)
		want := ptr + uintptr(i)*a.t.Size()
		val := reflect.NewAt(a.t, unsafe.Pointer(want))
		val = reflect.Indirect(val)
		val.Set(xv)
	}
}

// Get returns the ith element of the underlying array of the *Dense tensor.
func (a *array) Get(i int) interface{} {
	switch a.t.Kind() {
	case reflect.Bool:
		return a.GetB(i)
	case reflect.Int:
		return a.GetI(i)
	case reflect.Int8:
		return a.GetI8(i)
	case reflect.Int16:
		return a.GetI16(i)
	case reflect.Int32:
		return a.GetI32(i)
	case reflect.Int64:
		return a.GetI64(i)
	case reflect.Uint:
		return a.GetU(i)
	case reflect.Uint8:
		return a.GetU8(i)
	case reflect.Uint16:
		return a.GetU16(i)
	case reflect.Uint32:
		return a.GetU32(i)
	case reflect.Uint64:
		return a.GetU64(i)
	case reflect.Uintptr:
		return a.GetUintptr(i)
	case reflect.Float32:
		return a.GetF32(i)
	case reflect.Float64:
		return a.GetF64(i)
	case reflect.Complex64:
		return a.GetC64(i)
	case reflect.Complex128:
		return a.GetC128(i)
	case reflect.String:
		return a.GetStr(i)
	case reflect.UnsafePointer:
		return a.GetUnsafePointer(i)
	default:
		at := uintptr(a.ptr) + uintptr(i)*a.t.Size()
		val := reflect.NewAt(a.t, unsafe.Pointer(at))
		val = reflect.Indirect(val)
		return val.Interface()
	}
}

// Memset sets all values in the array.
func (a *array) Memset(x interface{}) error {
	switch a.t.Kind() {
	case reflect.Bool:
		xv, ok := x.(bool)
		if !ok {
			return errors.Errorf(dtypeMismatch, a.t, x)
		}
		data := a.Bools()
		for i := range data {
			data[i] = xv
		}
	case reflect.Int:
		xv, ok := x.(int)
		if !ok {
			return errors.Errorf(dtypeMismatch, a.t, x)
		}
		data := a.Ints()
		for i := range data {
			data[i] = xv
		}
	case reflect.Int8:
		xv, ok := x.(int8)
		if !ok {
			return errors.Errorf(dtypeMismatch, a.t, x)
		}
		data := a.Int8s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Int16:
		xv, ok := x.(int16)
		if !ok {
			return errors.Errorf(dtypeMismatch, a.t, x)
		}
		data := a.Int16s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Int32:
		xv, ok := x.(int32)
		if !ok {
			return errors.Errorf(dtypeMismatch, a.t, x)
		}
		data := a.Int32s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Int64:
		xv, ok := x.(int64)
		if !ok {
			return errors.Errorf(dtypeMismatch, a.t, x)
		}
		data := a.Int64s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Uint:
		xv, ok := x.(uint)
		if !ok {
			return errors.Errorf(dtypeMismatch, a.t, x)
		}
		data := a.Uints()
		for i := range data {
			data[i] = xv
		}
	case reflect.Uint8:
		xv, ok := x.(uint8)
		if !ok {
			return errors.Errorf(dtypeMismatch, a.t, x)
		}
		data := a.Uint8s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Uint16:
		xv, ok := x.(uint16)
		if !ok {
			return errors.Errorf(dtypeMismatch, a.t, x)
		}
		data := a.Uint16s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Uint32:
		xv, ok := x.(uint32)
		if !ok {
			return errors.Errorf(dtypeMismatch, a.t, x)
		}
		data := a.Uint32s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Uint64:
		xv, ok := x.(uint64)
		if !ok {
			return errors.Errorf(dtypeMismatch, a.t, x)
		}
		data := a.Uint64s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Uintptr:
		xv, ok := x.(uintptr)
		if !ok {
			return errors.Errorf(dtypeMismatch, a.t, x)
		}
		data := a.Uintptrs()
		for i := range data {
			data[i] = xv
		}
	case reflect.Float32:
		xv, ok := x.(float32)
		if !ok {
			return errors.Errorf(dtypeMismatch, a.t, x)
		}
		data := a.Float32s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Float64:
		xv, ok := x.(float64)
		if !ok {
			return errors.Errorf(dtypeMismatch, a.t, x)
		}
		data := a.Float64s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Complex64:
		xv, ok := x.(complex64)
		if !ok {
			return errors.Errorf(dtypeMismatch, a.t, x)
		}
		data := a.Complex64s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Complex128:
		xv, ok := x.(complex128)
		if !ok {
			return errors.Errorf(dtypeMismatch, a.t, x)
		}
		data := a.Complex128s()
		for i := range data {
			data[i] = xv
		}
	case reflect.String:
		xv, ok := x.(string)
		if !ok {
			return errors.Errorf(dtypeMismatch, a.t, x)
		}
		data := a.Strings()
		for i := range data {
			data[i] = xv
		}
	case reflect.UnsafePointer:
		xv, ok := x.(unsafe.Pointer)
		if !ok {
			return errors.Errorf(dtypeMismatch, a.t, x)
		}
		data := a.UnsafePointers()
		for i := range data {
			data[i] = xv
		}
	default:
		xv := reflect.ValueOf(x)
		ptr := uintptr(a.ptr)
		for i := 0; i < a.l; i++ {
			want := ptr + uintptr(i)*a.t.Size()
			val := reflect.NewAt(a.t, unsafe.Pointer(want))
			val = reflect.Indirect(val)
			val.Set(xv)
		}
	}
	return nil
}

// Eq checks that any two arrays are equal
func (a array) Eq(other interface{}) bool {
	if oa, ok := other.(array); ok {
		if oa.t != a.t {
			return false
		}

		if oa.l != a.l {
			return false
		}

		if oa.c != a.c {
			return false
		}

		// same exact thing
		if uintptr(oa.ptr) == uintptr(a.ptr) {
			return true
		}

		switch a.t.Kind() {
		case reflect.Bool:
			for i, v := range a.Bools() {
				if oa.GetB(i) != v {
					return false
				}
			}
		case reflect.Int:
			for i, v := range a.Ints() {
				if oa.GetI(i) != v {
					return false
				}
			}
		case reflect.Int8:
			for i, v := range a.Int8s() {
				if oa.GetI8(i) != v {
					return false
				}
			}
		case reflect.Int16:
			for i, v := range a.Int16s() {
				if oa.GetI16(i) != v {
					return false
				}
			}
		case reflect.Int32:
			for i, v := range a.Int32s() {
				if oa.GetI32(i) != v {
					return false
				}
			}
		case reflect.Int64:
			for i, v := range a.Int64s() {
				if oa.GetI64(i) != v {
					return false
				}
			}
		case reflect.Uint:
			for i, v := range a.Uints() {
				if oa.GetU(i) != v {
					return false
				}
			}
		case reflect.Uint8:
			for i, v := range a.Uint8s() {
				if oa.GetU8(i) != v {
					return false
				}
			}
		case reflect.Uint16:
			for i, v := range a.Uint16s() {
				if oa.GetU16(i) != v {
					return false
				}
			}
		case reflect.Uint32:
			for i, v := range a.Uint32s() {
				if oa.GetU32(i) != v {
					return false
				}
			}
		case reflect.Uint64:
			for i, v := range a.Uint64s() {
				if oa.GetU64(i) != v {
					return false
				}
			}
		case reflect.Uintptr:
			for i, v := range a.Uintptrs() {
				if oa.GetUintptr(i) != v {
					return false
				}
			}
		case reflect.Float32:
			for i, v := range a.Float32s() {
				if oa.GetF32(i) != v {
					return false
				}
			}
		case reflect.Float64:
			for i, v := range a.Float64s() {
				if oa.GetF64(i) != v {
					return false
				}
			}
		case reflect.Complex64:
			for i, v := range a.Complex64s() {
				if oa.GetC64(i) != v {
					return false
				}
			}
		case reflect.Complex128:
			for i, v := range a.Complex128s() {
				if oa.GetC128(i) != v {
					return false
				}
			}
		case reflect.String:
			for i, v := range a.Strings() {
				if oa.GetStr(i) != v {
					return false
				}
			}
		case reflect.UnsafePointer:
			for i, v := range a.UnsafePointers() {
				if oa.GetUnsafePointer(i) != v {
					return false
				}
			}
		default:
			for i := 0; i < a.l; i++ {
				if !reflect.DeepEqual(a.Get(i), oa.Get(i)) {
					return false
				}
			}
		}
		return true
	}
	return false
}
